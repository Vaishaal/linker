import shapefile
import argparse
from collections import defaultdict
from scipy.spatial import KDTree
import numpy as np
import json
import utm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


BUILDING = 0
INTERSECTION = 1
ROAD = 2
pipes =  shapefile.Writer(shapefile.POLYLINE)
pipes.field('ID', 'N', '10')
class Linker(object):
    def __init__(self, buildings_path, intersections_path, roads_path):
        build_sf = shapefile.Reader(buildings_path) # Polygons.
        int_sf = shapefile.Reader(intersections_path) # Points.
        road_sf = shapefile.Reader(roads_path) # Series of line segments.
        self.irecords = int_sf.records()
        self.rrecords = road_sf.records()
        self.buildings = map(
                            lambda building: \
                                [to_utm(tuple(corner)) for corner in building.points],
                            build_sf.shapes()
                            )
        self.intersections = map(
                                lambda intersection: \
                                    to_utm(tuple(intersection.points[0])),
                                    int_sf.shapes()
                                )
        self.roads = map(
                        lambda road: [to_utm(tuple(corner)) for corner in road.points],
                        road_sf.shapes()
                        )
        self.blocks = self.grab_blocks()
        self.segment_map, self.r_segment_map = self.grab_road_segments()
        self.segment_lookup = self.organize_road() # Find segments from points
        self.building_lookup = self.organize_buildings() # Find buildings from segments
        self.nodes = self.build_nodes()
        self.node_map = {n["key"]:n for n in self.nodes}
        self.edges = defaultdict(list)

    def build_nodes(self):
        nodes = []
        keys = set()
        for i,b in enumerate(self.buildings):
            node = {}
            attrs = {}
            cx,cy = centroid(b)
            #cx,cy,_,__ = utm.from_latlon(cy,cx)
            node["x"] = cx
            node["y"] = cy
            node["key"] = "b{0}".format(i)
            assert(node["key"] not in keys)
            keys.add(i)
            attrs["nodeType"] = BUILDING
            fill_attr(attrs)
            node["attr"] = attrs
            nodes.append(node)

        for i,ix in enumerate(self.intersections):
            node = {}
            attrs = {}
            cx,cy = ix[0],ix[1]
            #cx,cy,_,__ = utm.from_latlon(ix[1],ix[0])
            node["x"] = cx
            node["y"] = cy
            node["key"] = "i{0}".format(i)
            assert(node["key"] not in keys)
            keys.add(node["key"])
            attrs["nodeType"] = INTERSECTION
            fill_attr(attrs)
            node["attr"] = attrs
            nodes.append(node)
        return nodes
    def organize_road(self):
        '''Returns a function that takes in two parameters,
        a point and k, this will then return atleast the k nearest neighbor 
        road segments to that point 
        '''
        segment_mp_map = defaultdict(list)
        for seg in self.r_segment_map.keys():
            xm = (seg[0][0] + seg[1][0])/2.0
            ym = (seg[0][1] + seg[1][1])/2.0
            segment_mp_map[(xm,ym)].append(seg)
        mp_array = np.array(segment_mp_map.keys())
        tree = KDTree(mp_array)
        def lookup(pt, k=5):
            midpoints = tree.query(pt,k)
            ret = []
            for mpt in mp_array[midpoints[1]]:
                ret += segment_mp_map[tuple(mpt)]
            return ret
        return lookup

    def organize_buildings(self):
        '''Returns a function that takes in a road segment and gives nearby
           buildings
        '''
        segment_build_map = defaultdict(list)
        for i,building in enumerate(self.buildings):
            cx,cy = centroid(building)
            segments = self.segment_lookup([cx,cy])
            for seg in segments:
                segment_build_map[tuple(seg)].append(i)
        def lookup(segment):
            return segment_build_map[segment]
        return lookup

    def grab_blocks(self):
        ''' Generate blocks from intersections and roads 
            every block is an tuple of two ints each of which correspond to an
            intersection. A pipe will then drawn on both sides of this road.
        '''
        all_blocks = []
        int_map = dict([i[::-1] for i in enumerate(self.intersections)])
        for i,road in enumerate(self.roads):
            block_list = map(lambda x: int_map[x],
                             filter(lambda x: x in int_map, road))
            blocks = zip(block_list[:-1],block_list[1:])
            all_blocks += blocks
        return all_blocks

    def grab_road_segments(self):
        ''' Return a map from roadid to a list of segments, 
            each of which belong on that road, returns reverse
            map to
        '''
        segment_map = {}
        r_segment_map = {}
        for i,road in enumerate(self.roads):
            segments = zip(road[:-1],road[1:])
            segment_map[self.rrecords[i][0]] = segments
            for segment in segments:
                r_segment_map[tuple(segment)] = self.rrecords[i][0]
        return segment_map,r_segment_map



    def calculate_pipe_width(self, segment):
        ''' Given a particular segment (a pair of points),
            calculate the optimum linking pipe width.
        '''
        return 20

    def link(self):
        for osm_id,road in self.segment_map.items():
            reverse_segments = map(lambda x: x[::-1], road)
            all_segs = reverse_segments + road
            for segment in all_segs:
                width = self.calculate_pipe_width(segment)
                pipe = Pipe(width, segment,self)
                nearby_buildings = self.building_lookup(segment)
                in_pipe = filter(lambda x: self.buildings[x] in pipe, nearby_buildings)
                pt1 = np.array(segment[0])
                pt2 = np.array(segment[1])
                cos = pt1.dot(pt2)/(np.linalg.norm(pt1)*np.linalg.norm(pt2))
                seg_angle = np.arccos(cos)
                centroids = np.array(map(lambda x: centroid(self.buildings[x]), in_pipe))
                centroid_projections = map(lambda x: np.linalg.norm(x)*np.cos(seg_angle),
                                           centroids)
                sorted_buildings = map(lambda x: x[1], sorted(enumerate(in_pipe), key=lambda x: centroid_projections[x[0]]))
                edges = zip(sorted_buildings[:-1],sorted_buildings[1:])
                for e1,e2 in edges:
                    self.edges["b{0}".format(e1)].append("b{0}".format(e2))
                    self.edges["b{0}".format(e2)].append("b{0}".format(e1))

    def write(self):
        nodes = open("graph.nodes.json","w+")
        edges = open("graph.edges.json","w+")
        nodes.write(json.dumps(self.nodes))
        edges.write(json.dumps(self.edges))
        pipes.save("pipes")
        nodes.close()
        edges.close()


class Pipe(object):
    def __init__(self, width, segment, linker):
        p1 = np.array(segment[0])
        p4 = np.array(segment[1])
        rot_matrix = np.array([[0,-1],[1,0]])
        rot = rot_matrix.dot((p4 - p1))
        p2 = p1 + rot/(np.linalg.norm(rot))*width
        p3 = p4 + rot/(np.linalg.norm(rot))*width
        pts = map(to_latlon,map(list,[p1,p2,p3,p4,p1]))
        pipes.record(0)
        pipes.poly(shapeType=shapefile.POLYLINE, parts=zip(list(pts[:-1]),list(pts[1:])))
        self.polygon = Polygon([p1,p2,p3,p4])
    def __contains__(self, building):
        building_poly = Polygon(building)
        return self.polygon.intersects(building_poly)


ATTRIBUTES = ["roadClass", "degree","nodeType","height","width"]

def fill_attr(attr):
    for k in ATTRIBUTES:
        if k not in attr:
            attr[k] = -1

def to_utm(pt):
    return utm.from_latlon(pt[1],pt[0])[:2]

def to_latlon(pt):
    return utm.to_latlon(pt[0],pt[1],37,'S')[::-1]
def centroid(vertices):
    x,y = zip(*vertices)
    centroid = (sum(x) / len(vertices), sum(y) / len(vertices))
    return centroid

if __name__ == "__main__":
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description='Links buildings and intersections for the database.')
    parser.add_argument('buildings', type=str, help='Shapefile containing building footprints.')
    parser.add_argument('intersections', type=str, help='Shapefile containing intersection points.')
    parser.add_argument('roads', type=str, help='Shapefile containing roads.')
    args = parser.parse_args()
    linker = Linker(args.buildings, args.intersections, args.roads)
    linker.link()
    linker.write()
