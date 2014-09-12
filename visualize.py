import shapefile
import argparse
import json
import utm

def visualize(name,nodes, edges, utm_zone, latlon):
    nodes_shp =  shapefile.Writer(shapefile.POINT)
    edges_shp =  shapefile.Writer(shapefile.POLYLINE)
    nodes_shp.field('ID', 'N', '10')
    edges_shp.field('ID', 'N', '10')
    points = {}
    roads = {}
    lines = []
    seen_edges = set()
    count = 0
    for n in nodes:
        if (n["attr"]["nodeType"] != 2):
            if (not latlon):
                point = utm.to_latlon(n["x"],n["y"],utm_zone[0],utm_zone[1])
            else:
                point = n["y"],n["x"]
            points[n["key"]] = point
            nodes_shp.point(point[1],point[0])
            nodes_shp.record(count)
            count += 1
    node_map = {n["key"]:n for n in nodes}
    for k,edge in edges.items():
        count +=1
        if int(k) not in node_map or int(k) not in points: continue
        node1  = node_map[int(k)]
        for e in edge:
            if e not in node_map or e not in points: continue
            node2 = node_map[e]
            line = [points[node2["key"]][::-1],points[node1["key"]][::-1]]
            if (tuple(set(line)) not in seen_edges):
                seen_edges.add(tuple(set(line)))
                edges_shp.record(count)
                edges_shp.poly(shapeType=shapefile.POLYLINE, parts=[line])

    nodes_shp.save('/home/vaishaal/Dropbox/nodes')
    if (edges):
        edges_shp.save('/home/vaishaal/Dropbox/edges')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate node and edge shapefile')
    parser.add_argument('graph', help='path to graph (omitting the .nodes/.edges)')
    parser.add_argument('--latlon', dest='latlon', action='store_true')
    parser.add_argument('--utm_zone', help='utm zone graph is located in (default 37,N)', default="36,N")
    args = parser.parse_args()
    utm_zone = args.utm_zone.split(",")
    utm_number, utm_dir = int(utm_zone[0]), utm_zone[1]
    nodes = json.loads(open(args.graph +".nodes.json").read())
    edges = json.loads(open(args.graph + ".edges.json").read())
    print edges
    visualize(args.graph,nodes,edges, (utm_number, utm_dir), args.latlon)
