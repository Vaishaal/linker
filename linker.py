import argparse
import shapefile
import numpy as np
from math import pi, sqrt
from scipy.spatial import *
import utm
from collections import defaultdict
import json
import pickle
import time

PRELIM_DIST = 40 # Range of preliminary edges in meters.
PIPE_WIDTH = 20 # Max width of the pipe in meters.
ANGLE_TOLERANCE = pi/8 # Angle of tolerance that we can call straight for determining T intersections.
DEGREE = 1 # Field of the intersections record that contains degree.
OVERLAP_THRESHOLD = 5 # Number of meters the centroids of two buildings have to be apart to count as non-overlapping.
ROT90 = np.matrix([[0, -1], [1, 0]]) # Counter-clockwise 90 degree rotation matrix.
NORTH = np.array([0, 1]) # North pointing unit vector.

# Matcher node types.
BUILDING = 0
INTERSECTION = 1
ROAD = 2

# Intersection shapefile record fields.
INTERSECTING_ROADS = 0

# Road shapefile record fields.
ROAD_ID = 0
CLASS = 1
CLASS_MAP = {'service': 0,
        'residential': 1,
        'unclassified': 2,
        'tertiary': 3,
        'secondary': 4,
        'primary': 5,
        'trunk': 6,
        'motorway': 7,
        'track': 8,
        'primary_link': 9}

def link(building_sf_path, intersection_sf_path, road_sf_path, ignore_service = True):
    ###############
    # PRE-PROCESS #
    ###############

    # Read in the shapefiles.
    build_sf = shapefile.Reader(building_sf_path) # Polygons.
    int_sf = shapefile.Reader(intersection_sf_path) # Points.
    road_sf = shapefile.Reader(road_sf_path) # Series of line segments.
    irecords = int_sf.records()
    rrecords = road_sf.records()

    # Convert everything to UTM. Find building centroids.
    buildings_utm = map(lambda building: [lonlat_to_utm(corner) for corner in building.points], build_sf.shapes())
    intersections_utm = map(lambda intersection: lonlat_to_utm(intersection.points[0]), int_sf.shapes())
    roads_utm = map(lambda road: [lonlat_to_utm(corner) for corner in road.points], road_sf.shapes())
    building_centroids = np.zeros([len(build_sf.shapes()), 2])
    centroid_to_building = {}
    for i in range(len(build_sf.shapes())):
        centroid = compute_centroid(buildings_utm[i])
        building_centroids[i] = centroid
        centroid_to_building[centroid] = buildings_utm[i]

    # Generate a KDTree of buildings and intersections for preliminary edges.
    buildings_intersections = np.concatenate((building_centroids, np.array(intersections_utm)))
    point_tree = KDTree(buildings_intersections)

    # Preliminary edges.
    edges = defaultdict(set)
    for pair in point_tree.query_pairs(PRELIM_DIST):
        edges[pair[0]].add(pair[1])
        edges[pair[1]].add(pair[0])

    # Compute radius to search for road points.
    (segment_kdtree, kd_to_segment, segment_search_radius, service_set, segment_to_osm) = process_segments(roads_utm, rrecords)

    #########
    # Prune #
    #########
    print 'Pruning edges that cross roads.'
    x = time.time()
    prune_cross_roads(edges, buildings_intersections, segment_kdtree, kd_to_segment, segment_search_radius, service_set, ignore_service)
    y = time.time()
    print "%0.1f seconds\n" % (y - x)

    print 'Pruning edges that belong to interior buildings.'
    building_to_segments = {}
    x = time.time()
    prune_interior(edges, building_centroids, buildings_utm, segment_kdtree, kd_to_segment, segment_search_radius, building_to_segments)
    y = time.time()
    print "%0.1f seconds\n" % (y - x)

    print 'Pruning extra building-intersection edges.'
    x = time.time()
    prune_building_intersection(edges, building_centroids, intersections_utm)
    y = time.time()
    print "%0.1f seconds\n" % (y - x)

    print 'Pruning extra building-building edges.'
    x = time.time()
    prune_building_building(edges, building_centroids)
    y = time.time()
    print "%0.1f seconds\n" % (y - x)

    ### Add Road Nodes ###
    for key in building_to_segments:
        building_to_segments[key].sort(key=lambda x: x[1])
        intersection_road_ids = set()
        for n in edges[key]:
            if n >= len(building_centroids):
                intersection_road_ids.update(filter(lambda x: x != -1,\
                        map(int, irecords[n - len(building_centroids)][INTERSECTING_ROADS].split(','))))
        if intersection_road_ids:
            for seg in building_to_segments[key]:
                # Allow all roads inside the pipe width
                osm_id = segment_to_osm[seg[0]]
                if osm_id in intersection_road_ids:
                    edges[osm_id].add(key)
        elif len(building_to_segments[key]) > 0:
            osm_id = segment_to_osm[building_to_segments[key][0][0]] # Just add closest segment.

    ### OUTPUT ###
    for key in edges:
        edges[key] = list(edges[key])
    output_db(edges, building_centroids, intersections_utm, roads_utm, rrecords, irecords)


###################
# Pruning Modules #
###################
def prune_cross_roads(edges, buildings_intersections, segment_kdtree, kd_to_segment, segment_search_radius, service_set, ignore_service):
    # Edges should never cross roads unless we are choosing to ignore service roads.
    for u in edges:
        start = buildings_intersections[u]
        nearby_segment_indices = segment_kdtree.query_ball_point(start, segment_search_radius)
        nearby_segments = set(map(lambda i: kd_to_segment[i], nearby_segment_indices))
        remaining = set()
        for v in edges[u]:
            end = buildings_intersections[v]
            edge_vector = end - start
            for segment in nearby_segments:
                seg_start = segment[0]
                seg_vector = np.array(segment[1]) - seg_start
                # Solve for how long along the vector directions the segment and edge intersection (or not).
                solution = np.linalg.pinv(np.matrix([seg_vector, -edge_vector]).T)*(np.matrix(start - seg_start).T)
                if min(solution) < 0 or max(solution) > 1 or (segment in service_set and ignore_service):
                    remaining.add(v)
        edges[u] = remaining
    pass

def prune_interior(edges, building_centroids, buildings_utm, segment_kdtree, kd_to_segment, segment_search_radius, building_to_segments):
    # Buildings whose closest corner to road distance is more than PIPE_WIDTH should not have any edges.
    to_remove = set()
    for index in edges:
        if index < len(building_centroids):
            point = building_centroids[index]
            nearby_segment_indices = segment_kdtree.query_ball_point(point, segment_search_radius)
            nearby_segments = set(map(lambda i: kd_to_segment[i], nearby_segment_indices))
            keep = False
            seg_distances = []
            for segment in nearby_segments:
                seg_start = segment[0]
                seg_vector = (np.matrix(segment[1]) - seg_start).T
                perp_vector = (ROT90*seg_vector)/np.linalg.norm(seg_vector)
                solution = np.linalg.pinv(np.concatenate([seg_vector, -perp_vector], axis=1))*((np.matrix(point) - np.matrix(seg_start)).T)
                seg_distances.append((segment, float(abs(solution[1]))))
                for corner in buildings_utm[index]:
                    solution = np.linalg.pinv(np.concatenate([seg_vector, -perp_vector], axis=1))*((np.matrix(corner) - np.matrix(seg_start)).T)
                    # abs(solution[1]) is the distance from building to segment
                    if abs(solution[1]) <= PIPE_WIDTH:
                        keep = True
                        break
                if keep: #TODO: Python has no labeled continues. Maybe a cleaner way to do this?
                    break
            building_to_segments[index] = seg_distances
            if keep:
                continue
            to_remove.add(index)
    for index in to_remove:
        edges.pop(index, None)
    for u in edges:
        remaining = set()
        for v in edges[u]:
            if not v in to_remove:
                remaining.add(v)
        edges[u] = remaining

def prune_building_intersection(edges, building_centroids, intersections_utm):
    # Only buildings that are on the corners of blocks can have edges to intersections.
    for index in edges:
        if index >= len(building_centroids): # If it's an intersection.
            int_point = intersections_utm[index - len(building_centroids)]
            checked = set()
            keep = set()
            for n in edges[index]: # n for neighbor
                if n >= len(building_centroids):
                    continue
                elif not n in checked: # DFS to find connected buildings.
                    fringe = [n]
                    cc = set() # Connected component.
                    while fringe:
                        a = fringe.pop()
                        cc.add(a)
                        for b in edges[a]:
                            if b < len(building_centroids) and not b in cc:
                                fringe.append(b)
                    dist_list = []
                    for m in edges[index]:
                        if m in cc:
                            dist_list.append((m, np.linalg.norm(np.array(int_point) - np.array(building_centroids[m]))))
                    dist_list.sort(key=lambda x: x[1])
                    keep.add(dist_list[0][0])
                    for tmp in dist_list:
                        checked.add(tmp[0])
            for k in keep:
                edges[k].remove(index)

def prune_building_building(edges, building_centroids):
    # A building can only have edges to up to two buildings.
    # Pick the closest two buildings. Must run AFTER prune_interior and prune_cross_roads.
    for key in edges:
        if key < len(building_centroids):
            a = building_centroids[key]
            dist_list = []
            for b in edges[key]:
                if b < len(building_centroids):
                    dist_list.append((b, np.linalg.norm(a - building_centroids[b])))
            dist_list.sort(key=lambda x: x[1])
            keep = set()
            if len(dist_list) > 0:
                keep.add(dist_list[0][0])
            if len(dist_list) > 1:
                keep.add(dist_list[1][0])
            edges[key] = keep

####################
# Helper Functions #
####################
def lonlat_to_utm(point):
    output = utm.conversion.from_latlon(point[1], point[0]) # Swap indices for lonlat.
    return [output[0], output[1]]

def utm_to_lonlat(point):
    output = utm.to_latlon(point[0], point[1], 37, 'S')
    return [output[1], output[0]]

def process_segments(roads_utm, rrecords):
    # Find the longest road segment.
    max_segment_length = -float('inf')
    num_segments = reduce(lambda x,y: x + y, map(lambda road: len(road) - 1, roads_utm))
    segment_kd_points = np.zeros([5*num_segments, 2])
    kd_to_segment = []
    service_set = set()
    segment_to_osm = {}
    counter = 0
    for road_ind, road in enumerate(roads_utm):
        if rrecords[road_ind][CLASS] == 'service':
            is_service = True
        else:
            is_service = False
        for i in range(len(road)-1):
            start = np.array(road[i])
            end = np.array(road[i+1])
            segment = (tuple(road[i]), tuple(road[i+1]))
            segment_to_osm[segment] = rrecords[road_ind][ROAD_ID]
            segment_vector = end - start
            segment_length = np.linalg.norm(segment_vector)
            max_segment_length = segment_length if segment_length > max_segment_length else max_segment_length
            for i in range(5):
                segment_kd_points[counter] = start + (i/4)*segment_vector
                kd_to_segment.append(segment)
                if is_service:
                    service_set.add(segment)
                counter += 1
    # Return KD tree of segments, list to map from KDTree indices to segments and the maximum distance
    # a point can be from any end or midpoint of a segment + 0.1 as floating point tolerance.
    return (KDTree(segment_kd_points), kd_to_segment, sqrt((max_segment_length/8)**2 + PRELIM_DIST**2) + 0.1, service_set, segment_to_osm)

def compute_centroid(polygon):
    # Implements the formula on the wikipedia page.
    A = 0 # A for Area
    Cx = 0 # Centroid x coordinate
    Cy = 0 # y coordinate
    for i in range(len(polygon) - 1):
        tmp = polygon[i][0]*polygon[i+1][1] - polygon[i+1][0]*polygon[i][1]
        A += tmp
        Cx += (polygon[i][0] + polygon[i+1][0])*tmp
        Cy += (polygon[i][1] + polygon[i+1][1])*tmp
    A = A/2
    Cx = Cx/(6*A)
    Cy = Cy/(6*A)
    return (Cx, Cy)

def get_angle(u, v):
    return abs(float(np.arccos((u/np.linalg.norm(u)).dot(v/np.linalg.norm(v)))))

##########
# Output #
##########

def output_db(edges, building_centroids, intersections_utm, roads_utm, rrecords, irecords):
    output_edges = json.dumps(edges)
    fe = open('db.edges.json', 'w+')
    fe.write(output_edges)
    fe.close()
    node_data = []
    for i, centroid in enumerate(building_centroids):
        node_data.append({'key': i,\
                'attr': {'length': -1,\
                    'height': -1,\
                    'anglel': -1,\
                    'angleh': -1,\
                    'roadClass': -1,\
                    'degree': -1,\
                    'nodeType': BUILDING},\
                'x': centroid[0],\
                'y': centroid[1]})
    for i, intersection in enumerate(intersections_utm):
        node_data.append({'key': len(building_centroids) + i,\
                'attr': {'length': -1,\
                    'height': -1,\
                    'anglel': -1,\
                    'angleh': -1,\
                    'roadClass': -1,\
                    'degree': irecords[i][DEGREE],\
                    'nodeType': INTERSECTION},\
                'x': intersection[0],\
                'y': intersection[1]})
    for i, road_attrs in enumerate(rrecords):
        road = roads_utm[i]
        angles = []
        for j in range(len(road) - 1):
            seg_vector = np.array(road[j+1]) - np.array(road[j])
            seg_vector = seg_vector/np.linalg.norm(seg_vector)
            angles.append(np.arcsin(np.linalg.norm(np.cross(NORTH, seg_vector))))
        node_data.append({'key': road_attrs[ROAD_ID],\
                'attr': {'length': -1,\
                    'height': -1,\
                    'anglel': min(angles),\
                    'angleh': max(angles),\
                    'roadClass': CLASS_MAP[road_attrs[CLASS]],\
                    'degree': -1,\
                    'nodeType': ROAD},\
                'x': -1,\
                'y': -1})
    output_nodes = json.dumps(node_data)
    fn = open('db.nodes.json', 'w+')
    fn.write(output_nodes)
    fn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Links buildings and intersections for the database.')
    parser.add_argument('buildings', type=str, help='Shapefile containing building footprints.')
    parser.add_argument('intersections', type=str, help='Shapefile containing intersection points.')
    parser.add_argument('roads', type=str, help='Shapefile containing roads.')
    parser.add_argument('-s', help='Process service road intersections as normal.', action='store_true', default=False)
    args = parser.parse_args()
    link(args.buildings, args.intersections, args.roads, ignore_service=args.s)
