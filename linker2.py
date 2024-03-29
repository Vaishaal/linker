import shapefile
import argparse
from collections import defaultdict
from scipy.spatial import KDTree
import numpy as np
import json
import utm
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


BUILDING = 0
INTERSECTION = 1
ROAD = 2
pipe_count = 0
pipes =  shapefile.Writer(shapefile.POLYLINE)
pipes.field('ID', 'N', '10')

road_lines =  shapefile.Writer(shapefile.POLYLINE)
road_lines.field('ID', 'N', '10')

road_angles = shapefile.Writer(shapefile.POLYLINE)
road_angles.field('ID','N','10')

b = 0
i = 2**20
r = 2**30
NORTH = np.array([0, 1])
class Linker(object):
    def __init__(self, buildings_path, intersections_path, roads_path):
        build_sf = shapefile.Reader(buildings_path) # Polygons.
        int_sf = shapefile.Reader(intersections_path) # Points.
        road_sf = shapefile.Reader(roads_path) # Series of line segments.
        self.road_sf = road_sf
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
        self.edge_weights = defaultdict(list)
        self.int_map = dict([i[::-1] for i in enumerate(self.intersections)])

    def build_nodes(self):
        nodes = []
        keys = set()

        all_roads = map(
                    lambda road: [tuple(corner) for corner in road.points],
                    self.road_sf.shapes()
                    )
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
            attrs["degree"] = self.irecords[i][1]
            attrs["nodeType"] = INTERSECTION
            fill_attr(attrs)
            node["attr"] = attrs
            nodes.append(node)

        for i,r in enumerate(self.roads):
            node = {}
            attrs = {}
            cx,cy = -1,-1
            node["x"] = cx
            node["y"] = cy
            seg_vector = np.array(r[0]) - np.array(r[-1])
            pt1 = np.array(r[0])
            pt2 = np.array(r[-1])
            seg_vector[0]
            print pt1, pt2
            if (np.linalg.norm(seg_vector) != 0):
                angle = np.arccos(pt1.dot(pt2)/(np.linalg.norm(pt1)*np.linalg.norm(pt2)))
                angle = math.atan2(seg_vector[1],seg_vector[0])
                angle = ((angle * 180)/math.pi)%360
            else:
                angle = 0
            node["key"] = "r{0}".format(self.rrecords[i][0])

            road = all_roads[i]
            road_angles.poly(shapeType=shapefile.POLYLINE, parts=zip(road[:-1], road[1:]))
            road_angles.record(int(angle))

            assert(node["key"] not in keys)
            keys.add(node["key"])
            attrs["nodeType"] = ROAD
            attrs["roadType"] = self.irecords[i][1]
            attrs["angle"] = angle
            fill_attr(attrs)
            node["attr"] = attrs
            nodes.append(node)
        return nodes
    def organize_road(self):
        '''Returns a function that takes in two parameters,
        a point and k, this will then return atleast the k nearest neighbor 
        road segments to that point 
        '''
        segment_pts_map = defaultdict(list)
        for seg in self.r_segment_map.keys():
            pts = generate_points(seg,3)
            xm = (seg[0][0] + seg[1][0])/2.0
            ym = (seg[0][1] + seg[1][1])/2.0
            for pt in pts:
                segment_pts_map[tuple(pt)].append(seg)
        mp_array = np.array(segment_pts_map.keys())
        tree = KDTree(mp_array)
        def lookup(pt, k=5):
            points = tree.query(pt,k)
            ret = []
            for pt in mp_array[points[1]]:
                ret += segment_pts_map[tuple(pt)]
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
        return 15
    def add_edge(self, key1, key2, reverse=True, weight=1):
        self.edges[key1].append(key2)
        self.edge_weights[key1].append(weight)
        if (reverse):
            self.edges[key2].append(key1)
            self.edge_weights[key2].append(weight)

    def link(self):
        for osm_id,road in self.segment_map.items():
            segment_link_map = {}
            for segment in road:
                segment_links = self.link_segment(segment,osm_id)
                segment_link_map[segment] = segment_links
            for (segment1,segment2) in zip(road[:-1],road[1:]):
                assert(segment1[1] == segment2[0])
                buildings1 = segment_link_map[segment1]
                buildings2 = segment_link_map[segment2]
                b1_id = "b{0}".format(buildings1[0][-1])
                b1r_id = "b{0}".format(buildings1[1][-1])
                b2_id = "b{0}".format(buildings2[0][0])
                b2r_id = "b{0}".format(buildings2[1][0])

                if segment1[1] in self.int_map:
                    i_id = "i{0}".format(self.int_map[segment1[1]])
                    self.add_edge(b1_id,i_id)
                    self.add_edge(b2_id,i_id)
                    self.add_edge(b1r_id,i_id)
                    self.add_edge(b2r_id,i_id)
                    self.add_edge(b1_id,b2_id, weight=0.5)
                    self.add_edge(b1r_id,b2r_id, weight=0.5)
                else:
                    self.add_edge(b1_id,b2_id)
                    self.add_edge(b1r_id,b2r_id)
            start = road[0]
            end = road[-1]

            buildings1 = segment_link_map[start]
            b1_id = "b{0}".format(buildings1[0][0])
            b1r_id = "b{0}".format(buildings1[1][0])
            if start[0] in self.int_map:
                i_id = "i{0}".format(self.int_map[start[0]])
                self.add_edge(b1_id,i_id)
                self.add_edge(b1r_id,i_id)

            buildings2 = segment_link_map[end]
            b2_id = "b{0}".format(buildings2[0][-1])
            b2r_id = "b{0}".format(buildings2[1][-1])
            if end[1] in self.int_map:
                i_id = "i{0}".format(self.int_map[end[1]])
                self.add_edge(b2_id,i_id)
                self.add_edge(b2r_id,i_id)

    def link_segment(self,segment,osm_id):
        road_id = "r{0}".format(osm_id)
        width = self.calculate_pipe_width(segment)
        pipe = Pipe(width, segment,self)
        r_pipe = Pipe(width, segment[::-1],self) # Other side of road
        nearby_buildings = self.building_lookup(segment)
        in_pipe = filter(lambda x: self.buildings[x] in pipe, nearby_buildings)
        in_r_pipe = filter(lambda x: self.buildings[x] in r_pipe, nearby_buildings)
        xm = (segment[0][0] + segment[1][0])/2.0
        ym = (segment[0][1] + segment[1][1])/2.0
        p1 = to_latlon([xm,ym])
        if segment[0] in self.int_map:
            p2 = to_latlon(segment[0])
            road_lines.record(0)
            road_lines.poly(shapeType=shapefile.POLYLINE, parts=[[p1,p2]])
            i_id = "i{0}".format(self.int_map[segment[0]])
            self.add_edge(i_id,road_id, reverse=False)
        if segment[-1] in self.int_map:
            p2 = to_latlon(segment[-1])
            road_lines.record(0)
            road_lines.poly(shapeType=shapefile.POLYLINE, parts=[[p1,p2]])
            i_id = "i{0}".format(self.int_map[segment[-1]])
            self.add_edge(i_id,road_id, reverse=False)



        for b in (in_pipe + in_r_pipe):
            p2 = to_latlon(list(centroid(self.buildings[b])))
            road_lines.record(0)
            road_lines.poly(shapeType=shapefile.POLYLINE, parts=[[p1,p2]])
        pt1 = np.array(segment[0])
        pt2 = np.array(segment[1])
        cos = pt1.dot(pt2)/(np.linalg.norm(pt1)*np.linalg.norm(pt2))
        seg_angle = np.arccos(cos)
        centroids = np.array(map(lambda x: centroid(self.buildings[x]), in_pipe))
        centroid_projections = map(lambda x: np.linalg.norm(x)*np.cos(seg_angle),
                                   centroids)
        sorted_buildings = map(lambda x: x[1], sorted(enumerate(in_pipe), key=lambda x: centroid_projections[x[0]]))
        centroids_r = np.array(map(lambda x: centroid(self.buildings[x]), in_r_pipe))
        centroid_r_projections = map(lambda x: np.linalg.norm(x)*np.cos(seg_angle),
                                   centroids_r)
        sorted_r_buildings = map(lambda x: x[1], sorted(enumerate(in_r_pipe), key=lambda x: centroid_r_projections[x[0]]))

        edges = zip(sorted_buildings[:-1],sorted_buildings[1:]) + zip(sorted_r_buildings[:-1],sorted_r_buildings[1:])

        for e1,e2 in edges:
            b1 = "b{0}".format(e1)
            b2 = "b{0}".format(e2)
            self.add_edge(b1,b2)
            self.add_edge(b1,road_id)
            self.add_edge(b2,road_id)
        if sorted_r_buildings == []:
            sorted_r_buildings = [-1]
        if sorted_buildings == []:
            sorted_buildings = [-1]
        return (sorted_buildings, sorted_r_buildings)

    def write(self):
        nodes = open("graph.nodes.json","w+")
        edges = open("graph.edges.json","w+")
        weights = open("graph.weights.json","w+")
        change_key = lambda x: eval(x[0]) + long(x[1:])
        for node in self.nodes:
            old_key = node["key"]
            node["key"] = change_key(old_key)
        new_edges = {}
        for k,v in self.edges.items():
            new_edges[str(change_key(k))] = map(change_key, v)
        new_edge_weights = {}
        for k,v in self.edge_weights.items():
            new_edge_weights[str(change_key(k))] = v
        nodes.write(json.dumps(self.nodes))
        zipped_edge_weights = {k: zip(v, new_edge_weights[k]) for k,v in new_edges.items()}
        filtered_edges_and_weights = {k:list(set(v)) for k,v in zipped_edge_weights.items()}
        filtered_edges = {}
        filtered_weights = {}
        for k,v in filtered_edges_and_weights.items():
            edge_weights = dict(v)
            these_edges = edge_weights.keys()
            filtered_edges[k] = list(set(these_edges))
            filtered_weights[k] = map(lambda x: edge_weights[x], filtered_edges[k])

        weights.write(json.dumps(filtered_weights))
        edges.write(json.dumps(filtered_edges))
        pipes.save("/home/vaishaal/Dropbox/pipes")
        road_lines.save("/home/vaishaal/Dropbox/road_lines")
        road_angles.save("/home/vaishaal/Dropbox/road_angles")
        nodes.close()
        edges.close()


class Pipe(object):
    def __init__(self, width, segment, linker):
        global pipe_count
        p1 = np.array(segment[0])
        p4 = np.array(segment[1])
        rot_matrix = np.array([[0,-1],[1,0]])
        rot = rot_matrix.dot((p4 - p1))
        p2 = p1 + rot/(np.linalg.norm(rot))*width
        p3 = p4 + rot/(np.linalg.norm(rot))*width
        pts = map(to_latlon,map(list,[p1,p2,p3,p4,p1]))
        self.pipe_count = pipe_count
        pipe_count += 1
        pipes.poly(shapeType=shapefile.POLYLINE, parts=zip(list(pts[:-1]),list(pts[1:])))
        pipes.record(self.pipe_count)
        self.polygon = Polygon([p1,p2,p3,p4])
    def __contains__(self, building):
        building_poly = Polygon(building)
        intersection_area = self.polygon.intersection(building_poly).area
        return intersection_area/building_poly.area > 0.1


ATTRIBUTES = ["roadClass", "degree","nodeType","height","width"]

def fill_attr(attr):
    for k in ATTRIBUTES:
        if k not in attr:
            attr[k] = -1

def generate_points(seg,n):
    '''Will generate 2**n - 1 pts'''
    if n == 0:
        return []
    xm = (seg[0][0] + seg[1][0])/2.0
    ym = (seg[0][1] + seg[1][1])/2.0
    return [[xm,ym]] + generate_points([[xm,ym],seg[1]], n - 1) + generate_points([seg[0],[xm,ym]], n - 1)


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
