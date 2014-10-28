import shapefile
import argparse
from collections import defaultdict
from scipy.spatial import KDTree
import numpy as np
import json
import utm
import math
from math import atan2, degrees, pi
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LineString


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

        self.buildingCenters = self.buildBuildingCenters() #NEW
        self.buildingKDTree = self.buildKDTreeBuildings() #NEW
        self.roadToBuildings = self.categorizeBuildings() #NEW

        self.node_map = {n["key"]:n for n in self.nodes}
        self.edges = defaultdict(list)

        self.pipeEdges = defaultdict(list) #NEW

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
            if (np.linalg.norm(seg_vector) != 0):
                if pt1.dot(pt2)/(np.linalg.norm(pt1)*np.linalg.norm(pt2)) > 1.0:
                    angle = np.arccos(1.0)
                else:
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
        def lookup(pt, k=3):
            ret = []
            retSet = []
            ret2 = []
            i = k
            while len(ret) < k:
                points = tree.query(pt,i)
                for pt in mp_array[points[1]]:
                    if segment_pts_map[tuple(pt)] not in retSet:
                        ret+=(segment_pts_map[tuple(pt)])
                        retSet.append(segment_pts_map[tuple(pt)])
                i += 1
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

    def perpIntersectPoint(self, point, line_start, line_end):
        ''' Calc minimum distance from a point and a line segment and intersection'''
        # sqrDist of the line (PyQGIS function = magnitude (length) of a line **2)
        sx = line_start[0]
        sy = line_start[1]
        ex = line_end[0]
        ey = line_end[1]
        magnitude2 = (ex-sx)**2 + (ey-sy)**2
        # minimum distance
        u = ((point[0] - line_start[0]) * (line_end[0] - line_start[0]) + (point[1] - line_start[1]) * (line_end[1] - line_start[1]))/(magnitude2)
        # intersection point on the line
        ix = line_start[0] + u * (line_end[0] - line_start[0])
        iy = line_start[1]+ u * (line_end[1] - line_start[1])
        
        def distance(a,b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        def is_between(a,b,c):
            return -0.00001 < (distance(a, c) + distance(c, b) - distance(a, b)) < 0.00001

        pt = (ix,iy)
        if is_between(line_start,line_end,pt):
            return pt
        else:
            return None

    def buildBuildingCenters(self):
        nodes = {}
        keys = set()
        for i,b in enumerate(self.buildings):
            node = {}
            attrs = {}
            cx,cy = centroid(b)
            key = cx,cy
            value = [i, b]
            nodes[key] = value
        return nodes

    def buildKDTreeBuildings(self):
        return KDTree(self.buildingCenters.keys())

    def categorizeBuildings(self):
        def isLeft(a,b,c):
            return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0;

        def getIntersectionArea(bldPoly,line,perpLine,bldID):
            if not line.intersects(bldPoly):
                return 0.0
            pt1 = list(line.coords)[0]
            pt2 = list(line.coords)[1]
            perppt1 = list(perpLine.coords)[0]
            perppt2 = list(perpLine.coords)[1]
            dx = perppt2[0] - perppt1[0]
            dy = perppt2[1] - perppt1[1] 
            pt3 = (pt1[0]-dx,pt1[1]-dy)
            pt4 = (pt2[0]-dx,pt2[1]-dy)
            linePoly = Polygon([pt1,pt3,pt4,pt2])
            
            try:
                intersection_area = linePoly.intersection(bldPoly).area
                return intersection_area/bldPoly.area
            except:
                return -1

        def overlapsNeighbor(bldPoly,neighborPoly):
            try:
                intersection_area = bldPoly.intersection(neighborPoly).area
                return intersection_area/bldPoly.area > 0.05
            except:
                return False

        Rd2BldLeft = {}
        Rd2BldRight = {}
        for bld in self.buildingCenters.keys():
            bldID = self.buildingCenters[bld][0]

            #if bldID < 3700 or bldID > 3800:
            #if bldID != 3751:
                #continue

            roads = self.segment_lookup([bld[0],bld[1]],10)
            p1 = (bld[0], bld[1])
            p1LatLon = to_latlon(p1)
            res = self.buildingKDTree.query([bld[0],bld[1]], 20)
            neighbors = []
            for item in res[1][1:]: #start from index 1 because index 0 is always the original building bld
                buildingID = self.buildingCenters[self.buildingCenters.keys()[item]][0]
                neighbor = self.buildings[buildingID]
                #this part removes a bug that some neighbor shapes have extra information, which will result in Polygon error later on.
                start = neighbor[0]
                for i in range(1,len(neighbor)):
                    if neighbor[i] == start and i > 2:
                        break
                neighbor = neighbor[:i+1]
                neighbors.append(neighbor)
            roadDict = {}

            bldVertices = self.buildingCenters[bld][1]
            bldPolygon = Polygon(bldVertices)

            for rd in roads: 
                rdLine = LineString([rd[0],rd[1]])   
                intersectsSomething = False
                p3Intersect = self.perpIntersectPoint(p1,rd[0],rd[1])
                if not p3Intersect:
                    continue

                bldVertices.append(p1)
                for vertex in bldVertices:
                    if intersectsSomething:
                        break
                    p3 = self.perpIntersectPoint(vertex,rd[0],rd[1])
                    vertexLatLon = to_latlon(vertex)

                    if p3:
                        p3LatLon = to_latlon(p3)
                        perpLine = LineString([p1,p3])

                        for neighbor in neighbors:
                            neighborPoly = Polygon(neighbor)
                            if overlapsNeighbor(bldPolygon,neighborPoly):
                                continue
                            if perpLine.intersects(neighborPoly):
                                intersectRd = False
                                intersectionRdArea = 0
                                if neighborPoly.intersects(rdLine):
                                    intersectRd = True
                                    intersectionRdArea = getIntersectionArea(neighborPoly,rdLine,perpLine,bldID)
                                if intersectionRdArea > 0.4 or not intersectRd:
                                    #if bldID == 4287 or bldID == 4288:
                                        #print intersectionRdArea
                                    #road_lines.record(0)
                                    #road_lines.poly(shapeType=shapefile.POLYLINE, parts=[[vertexLatLon,p3LatLon]])
                                    intersectsSomething = True
                                    break
                        for rd2 in roads:
                            if rd2 == rd:
                                continue
                            roadLine2 = LineString([rd2[0],rd2[1]])
                            if perpLine.intersects(roadLine2):
                                intersectsSomething = True
                                break
            
                if not intersectsSomething:
                    perpLine = LineString([p1,p3Intersect])
                    if perpLine.length > (3*bldPolygon.length)/4:
                        continue
                    #if rdLine.length < (bldPolygon.length/3):
                    #        continue
                    p3IntersectLatLon = to_latlon(p3Intersect)
                    road_lines.record(0)
                    road_lines.poly(shapeType=shapefile.POLYLINE, parts=[[p3IntersectLatLon,p1LatLon]])
                    if isLeft(rd[0],rd[1],p1):
                        if rd not in Rd2BldLeft:
                            Rd2BldLeft[rd] = [bldID]
                        else:
                            Rd2BldLeft[rd].append(bldID)  
                    else:
                        if rd not in Rd2BldRight:
                            Rd2BldRight[rd] = [bldID]
                        else:
                            Rd2BldRight[rd].append(bldID)

        return (Rd2BldLeft, Rd2BldRight)              

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
        dist = self.getDistance(segment[0][0], segment[0][1], segment[1][0], segment[1][1])
        return 10 + dist/10

    def add_edge(self, key1, key2, reverse=True, weight=1):
        self.edges[key1].append(key2)
        self.edge_weights[key1].append(weight)
        if (reverse):
            self.edges[key2].append(key1)
            self.edge_weights[key2].append(weight)

    def getAngle(self, x1, y1, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            rads = atan2(-dy,dx)
            rads %= 2*pi
            degs = degrees(rads)
            return degs

    def getDistance(self, x1, y1, x2, y2):
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)


    def findMinDistanceBuilding(self, source, sourceX, sourceY, buildings):
        minDist = 9999999.9
        target = source
        for building in buildings:
            bld = self.nodes[building]
            if bld['x'] == sourceX and bld['y'] == sourceY:
                continue
            dist = self.getDistance(sourceX, sourceY, bld['x'], bld['y']) 
            if dist < minDist:
                minDist = dist
                target = building
        return target

    def link(self):
        for osm_id,road in self.segment_map.items():
            road_id = "r{0}".format(osm_id)
            segment_link_map = {}
            for segment in road:
                segment_links = self.link_segment(segment,osm_id)
                segment_link_map[segment] = segment_links
            for (segment1,segment2) in zip(road[:-1],road[1:]):
                assert(segment1[1] == segment2[0])
                buildings1 = segment_link_map[segment1]
                buildings2 = segment_link_map[segment2]

                if segment1[1] in self.int_map:
                    i_id = "i{0}".format(self.int_map[segment1[1]])

                    self.add_edge(i_id,road_id, reverse=False)
                    minDistBuilding1 = self.findMinDistanceBuilding(self.int_map[segment1[1]],segment1[1][0],segment1[1][1],buildings1[0])
                    b1_id = "b{0}".format(minDistBuilding1)

                    minDistBuilding1R = self.findMinDistanceBuilding(self.int_map[segment1[1]],segment1[1][0],segment1[1][1],buildings1[1])
                    b1r_id = "b{0}".format(minDistBuilding1R)

                    minDistBuilding2 = self.findMinDistanceBuilding(self.int_map[segment1[1]],segment1[1][0],segment1[1][1],buildings2[0])
                    b2_id = "b{0}".format(minDistBuilding2)

                    minDistBuilding2R = self.findMinDistanceBuilding(self.int_map[segment1[1]],segment1[1][0],segment1[1][1],buildings2[1])
                    b2r_id = "b{0}".format(minDistBuilding2R)

                    #self.add_edge(b1_id,i_id)
                    #self.add_edge(b2_id,i_id)
                    #self.add_edge(b1r_id,i_id)
                    #self.add_edge(b2r_id,i_id)
                    self.add_edge(b1_id,b2_id, weight=0.5)
                    self.add_edge(b1r_id,b2r_id, weight=0.5)

                    nb = self.nodes[minDistBuilding1]
                    cp = (nb['x'],nb['y'])
                    #cpLatlon = to_latlon(cp)
                    nbR = self.nodes[minDistBuilding1R]
                    cpR = (nbR['x'],nbR['y'])
                    #cpRLatlon = to_latlon(cpR)
                    nb2 = self.nodes[minDistBuilding2]
                    cp2 = (nb2['x'],nb2['y'])
                    #cp2Latlon = to_latlon(cp2)
                    nb2R = self.nodes[minDistBuilding2R]
                    cp2R = (nb2R['x'],nb2R['y'])
                    #cp2RLatlon = to_latlon(cp2R)
                    edgeLine = LineString([cp, cp2])
                    edgeLine2 = LineString([cpR, cp2R])

                    roads = self.segment_lookup([cp[0],cp[1]],5)
                    for rd in roads:
                        rdLine = LineString([rd[0],rd[1]])   
                        if edgeLine.intersects(rdLine):
                            self.add_edge(b1_id,i_id)
                            self.add_edge(b2_id,i_id)
                        if edgeLine2.intersects(rdLine):
                            self.add_edge(b1r_id,i_id)
                            self.add_edge(b2r_id,i_id)

                else:

                    minDist = 99999999.9
                    bldPair = (buildings1[0][0],buildings2[0][-1])
                    for building in buildings1[0]:
                        bld = self.nodes[building]
                        minDistBuilding = self.findMinDistanceBuilding(building,bld['x'],bld['y'],buildings2[0])
                        minBld = self.nodes[minDistBuilding]
                        dist = self.getDistance(bld['x'],bld['y'],minBld['x'],minBld['y'])
                        if dist < minDist:
                            minDist = dist
                            bldPair = (building,minDistBuilding)
                    b1_id = "b{0}".format(bldPair[0])
                    b2_id = "b{0}".format(bldPair[1])

                    minDist = 99999999.9
                    bldPair = (buildings1[1][0],buildings2[1][-1])
                    for building in buildings1[1]:
                        bld = self.nodes[building]
                        minDistBuilding = self.findMinDistanceBuilding(building,bld['x'],bld['y'],buildings2[1])
                        minBld = self.nodes[minDistBuilding]
                        dist = self.getDistance(bld['x'],bld['y'],minBld['x'],minBld['y'])
                        if dist < minDist:
                            minDist = dist
                            bldPair = (building,minDistBuilding)
                    b1r_id = "b{0}".format(bldPair[0])
                    b2r_id = "b{0}".format(bldPair[1])

                    self.add_edge(b1_id,b2_id)
                    self.add_edge(b1r_id,b2r_id)

            start = road[0]
            end = road[-1]

            
            buildings1 = segment_link_map[start]
            buildings2 = segment_link_map[end]

            if start[0] in self.int_map:
                i_id = "i{0}".format(self.int_map[start[0]])

                minDistBuilding = self.findMinDistanceBuilding(self.int_map[start[0]],start[0][0],start[0][1],buildings1[0])
                b1_id = "b{0}".format(minDistBuilding)

                minDistBuildingR = self.findMinDistanceBuilding(self.int_map[start[0]],start[0][0],start[0][1],buildings1[1])
                b1r_id = "b{0}".format(minDistBuildingR)

                self.add_edge(b1_id,i_id)
                self.add_edge(b1r_id,i_id)
                self.add_edge(b1_id,b1r_id, weight=0.5)

            if end[1] in self.int_map:
                i_id = "i{0}".format(self.int_map[end[1]])

                minDistBuilding = self.findMinDistanceBuilding(self.int_map[end[1]],end[1][0],end[1][1],buildings2[0])
                b2_id = "b{0}".format(minDistBuilding)

                minDistBuildingR = self.findMinDistanceBuilding(self.int_map[end[1]],end[1][0],end[1][1],buildings2[1])
                b2r_id = "b{0}".format(minDistBuildingR)
                
                self.add_edge(b2_id,i_id)
                self.add_edge(b2r_id,i_id)
                self.add_edge(b2_id,b2r_id, weight=0.5)

    def link_segment(self,segment,osm_id):
        road_id = "r{0}".format(osm_id)
        left_pipe = []
        right_pipe = []
        if segment in self.roadToBuildings[0]:
            left_pipe = self.roadToBuildings[0][segment]
        if segment in self.roadToBuildings[1]:
            right_pipe = self.roadToBuildings[1][segment]
        in_pipe = left_pipe
        in_r_pipe = right_pipe

        for building in in_pipe:
            #check for buildings in both pipes and keep it in the pipe with more intersection area
            if building in in_r_pipe:
                building_poly = Polygon(self.buildings[building])
                intersection_area = pipe.polygon.intersection(building_poly).area
                intersection_area_other = r_pipe.polygon.intersection(building_poly).area
                if intersection_area >= intersection_area_other:
                    in_r_pipe.remove(building)
                    #sorted_r_buildings.remove(building)
                    sorted_r_buildings = [x for x in sorted_r_buildings if x != building]
                else:
                    in_pipe.remove(building)
                    #sorted_buildings.remove(building)
                    sorted_buildings = [x for x in sorted_buildings if x != building]
                    continue

            bld1 = self.nodes[building]
            b1 = "b{0}".format(building)
            self.add_edge(b1,road_id, reverse=False)
            b2 = self.findMinDistanceBuilding(building,bld1['x'],bld1['y'],in_pipe)
            b2Bld = self.nodes[b2]
            b2Angle = self.getAngle(bld1['x'],bld1['y'],b2Bld['x'],b2Bld['y'])
            b2ID = "b{0}".format(b2)
            self.add_edge(b2ID,road_id, reverse=False)
            self.add_edge(b1,b2ID)

            new_pipe = [x for x in in_pipe if x != b2]
            b2Next = self.findMinDistanceBuilding(building,bld1['x'],bld1['y'],new_pipe)
            b2NextBld = self.nodes[b2Next]
            b2NextAngle = self.getAngle(bld1['x'],bld1['y'],b2NextBld['x'],b2NextBld['y'])
            b2NextID = "b{0}".format(b2Next)
            
            if abs(b2Angle - b2NextAngle) > 135 and abs(b2Angle - b2NextAngle) < 225:
                self.add_edge(b1,b2NextID)

        for building in in_r_pipe:
            bld1 = self.nodes[building]
            b1 = "b{0}".format(building)
            self.add_edge(b1,road_id, reverse=False)
            b2 = self.findMinDistanceBuilding(building,bld1['x'],bld1['y'],in_r_pipe)
            b2Bld = self.nodes[b2]
            b2Angle = self.getAngle(bld1['x'],bld1['y'],b2Bld['x'],b2Bld['y'])
            b2ID = "b{0}".format(b2)
            self.add_edge(b2ID,road_id, reverse=False)
            self.add_edge(b1,b2ID)

            new_pipe = [x for x in in_r_pipe if x != b2]
            b2Next = self.findMinDistanceBuilding(building,bld1['x'],bld1['y'],new_pipe)
            b2NextBld = self.nodes[b2Next]
            b2NextAngle = self.getAngle(bld1['x'],bld1['y'],b2NextBld['x'],b2NextBld['y'])
            b2NextID = "b{0}".format(b2Next)

            if abs(b2Angle - b2NextAngle) > 135 and abs(b2Angle - b2NextAngle) < 225:
                self.add_edge(b1,b2NextID)

        sorted_buildings = in_pipe
        sorted_r_buildings = in_r_pipe
        if sorted_r_buildings == []:
            sorted_r_buildings = [-1]
        if sorted_buildings == []:
            sorted_buildings = [-1]
        return (sorted_buildings, sorted_r_buildings)

    def write(self):
        nodes = open("../graph.nodes.json","w+")
        edges = open("../graph.edges.json","w+")
        weights = open("../graph.weights.json","w+")
        change_key = lambda x: eval(x[0]) + long(x[1:])
        for node in self.nodes:
            old_key = node["key"]
            node["key"] = change_key(old_key)
        new_edges = {}
        print "EDGES: ", len(self.edges)
        print "NODES: ", len(self.nodes)
        print "BUILDINGS: ", len(self.buildings)
        print "INTERSECTIONS: ", len(self.intersections)
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
        #pipes.save("Finderpipes")
        road_lines.save("Finderroad_lines")
        #road_angles.save("Finderroad_angles")
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
        try:
            intersection_area = self.polygon.intersection(building_poly).area
            return intersection_area/building_poly.area > 0.1
        except:
            return False


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
    return utm.to_latlon(pt[0],pt[1],36,'N')[::-1]
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
