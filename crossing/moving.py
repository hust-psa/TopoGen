import math
import numpy as np
import networkx as nx
from shapely.geometry import Point, MultiPoint, LineString, Polygon as shpPoly
from skgeom import Point2, Segment2, Ray2, arrangement as argmt, intersection
from utils.graph import irrelevant_edges
from utils.arrangement import representative_point, is_left
from utils.plotting import plot_arrangement as plt_argmt, plot_dual_graph as plt_dg


def create_arrangement(graph, node, **kwargs):
    """
    Create visibility region arrangement
    :param graph: networkx.Graph
    :param node: str, the name of the node being moved
    :param kwargs:
        bfs_depth: integer, depth of the BFS-tree used in the simplified heuristic
        simplified: boolean, whether to use the simplified heuristic
        limit_region: numpy.array or None, the boundary of the graph
            if not given, infer a border box from node coordinates
        min_edge_ratio: minimal number of nodes needed to be covered by the limited region
        expansion_ratio: expasion radius = expansion_ratio * average edge length
        box_pad_ratio: if not limit_region is given, use the border box with this padding ratio
    :return: tuple (arrangement, half_edge_deltas, half_edge_info)
        - arrangement: skgeom.arrangement.Arrangement, visibility region arrangement
        - half_edge_deltas: dict<skgeom.arrangement.Halfedge, integer>
            key: a half edge in the arrangement,
            value: the delta value of the half edge;
        - half_edge_info: dict<skgeom.arrangement.Halfedge, str>
            key: a half edge in the arrangement,
            value: the information of the half edge
    """
    simplified = kwargs.get('simplified', True)
    bfs_depth = kwargs.get('bfs_depth', 3)
    limit_region = kwargs.get('limit_region', None)
    expansion_ratio = kwargs.get('expansion_ratio', 0.2)
    box_pad_ratio = kwargs.get('box_pad_ratio', 0.3)
    if limit_region is not None:
        # convert numpy.array into shapely.Polygon
        limit_region = shpPoly(limit_region)
    else:
        pos = np.array([[attr['x'], attr['y']] for node, attr in graph.nodes(data=True)])
        x_min, y_min = np.min(pos, axis=0)
        x_max, y_max = np.max(pos, axis=0)
        padding_x, padding_y = box_pad_ratio * (x_max - x_min), box_pad_ratio * (y_max - y_min)
        left, right = x_min - padding_x, x_max + padding_x
        bottom, top = y_min - padding_y, y_max + padding_y
        limit_region = shpPoly([(left, top), (right, top), (right, bottom), (left, bottom)])

    # compute (n_f, n_g) values
    v = node
    nf_ng_values = dict()
    for u in graph.neighbors(v):
        for e in irrelevant_edges(graph, v):
            # segments on the visibility boundary could be the edge e=(x,y) or the rays connecting ux and uy
            x, y = e
            if (x, y, False) not in nf_ng_values:
                n_f, n_g = get_nf_ng(graph, v, e, is_ray=False)
                nf_ng_values[(x, y, False)] = (n_f, n_g)
                nf_ng_values[(y, x, False)] = (n_g, n_f)
            for z in e:
                if u == z:
                    continue
                if (u, z, True) not in nf_ng_values:
                    n_f, n_g = get_nf_ng(graph, v, (u, z), is_ray=True)
                    nf_ng_values[(u, z, True)] = (n_f, n_g)

    if simplified:
        # if using the simplification heuristic
        # create the arrangement for the subgraph surrounding the node
        # building a breadth-first-search tree rooted at <node> with specified <bfs_depth>
        bfs_tree = nx.bfs_tree(graph, node, depth_limit=bfs_depth)
        tree_pts = MultiPoint([(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in bfs_tree.nodes])
        edge_lengths = []
        for u, v in bfs_tree.edges:
            u_pt, v_pt = [(graph.nodes[i]['x'], graph.nodes[i]['y']) for i in (u, v)]
            edge_lengths.append(math.dist(u_pt, v_pt))
        expansion_radius = expansion_ratio * np.average(edge_lengths)
        coverage = tree_pts.convex_hull.buffer(expansion_radius)
        limit_region = limit_region.intersection(coverage)
        # find all incident edges to the nodes in the limited region
        subgraph_edges, internal_nodes = [], []
        for n, attr in graph.nodes(data=True):
            if limit_region.contains(Point(attr['x'], attr['y'])):
                internal_nodes.append(n)
                for incident_edge in graph.edges(n):
                    subgraph_edges.append(incident_edge)
        # only consider the subgraph composed of the nearest neighbors
        graph = nx.edge_subgraph(graph, subgraph_edges).copy()
        graph.remove_nodes_from(list(nx.isolates(graph)))
    else:
        internal_nodes = list(graph.nodes)

    # initialize the visibility region arrangement for node v
    v = node
    arrangement = argmt.Arrangement()
    if limit_region.geom_type == 'MultiPolygon':
        # find the largest subpolygon
        max_poly, max_poly_area = None, 0
        for poly in limit_region.geoms:
            if poly.area > max_poly_area:
                max_poly = poly
                max_poly_area = poly.area
        limit_region = max_poly
    limit_bnd_node_coords = limit_region.exterior.coords
    for i in range(len(limit_bnd_node_coords) - 1):
        seg = Segment2(Point2(*limit_bnd_node_coords[i]), Point2(*limit_bnd_node_coords[i + 1]))
        arrangement.insert(seg)

    # save the matching between line segments to the (n_f, n_g) values
    segment_to_delta = []
    # for each neighboring node u of v, find its visibility boundary towards each irrelevant edge e
    # add the boundary segments into the visibility region arrangement
    for u in graph.neighbors(v):
        for e in irrelevant_edges(graph, v):
            u_pt = (graph.nodes[u]['x'], graph.nodes[u]['y'])
            # segments on the visibility boundary could be the edge e=(x,y) or the rays connecting ux and uy
            x, y = e
            n_f, n_g = nf_ng_values[(x, y, False)]
            x_pt = (graph.nodes[x]['x'], graph.nodes[x]['y'])
            y_pt = (graph.nodes[y]['x'], graph.nodes[y]['y'])
            if x in internal_nodes and y in internal_nodes:
                seg = Segment2(Point2(*x_pt), Point2(*y_pt))
            else:
                intx_pt = segment_intersection_with_polygon(limit_region, LineString([x_pt, y_pt]))
                if x in internal_nodes:
                    seg = Segment2(Point2(*x_pt), Point2(*intx_pt))
                else:
                    seg = Segment2(Point2(*intx_pt), Point2(*y_pt))
            segment_to_delta.append((seg, e, n_f - n_g))
            segment_to_delta.append((seg.opposite(), e[::-1], n_g - n_f))
            arrangement.insert(seg)

            for z in e:
                if u == z or z not in internal_nodes:
                    continue
                n_f, n_g = nf_ng_values[(u, z, True)]
                # find the intersections of line uz and the boundary of the limited region
                z_pt = (graph.nodes[z]['x'], graph.nodes[z]['y'])
                intx_pt = ray_intersection_with_polygon(limit_region, LineString([u_pt, z_pt]))
                if intx_pt:
                    seg = Segment2(Point2(*z_pt), Point2(*intx_pt))
                    segment_to_delta.append((seg, (u, z + '→'), n_g - n_f))
                    segment_to_delta.append((seg.opposite(), (z + '→', u), n_f - n_g))
                    arrangement.insert(seg)

    # traverse each half edge in the overall arrangement and find the collinear edge segment that has been stored
    # if it is find, assign the delta of the edge segment to the half edge
    half_edge_deltas = dict()
    half_edge_info = dict()
    for he in arrangement.halfedges:
        for seg, info, delta in segment_to_delta:
            he_seg = Segment2(he.source().point(), he.target().point())
            if seg.has_on(he_seg.source()) and seg.has_on(he_seg.target()) and he_seg.direction() == seg.direction():
                half_edge_deltas[he] = delta
                half_edge_info[he] = info
                break
    return arrangement, half_edge_deltas, half_edge_info


def ray_intersection_with_polygon(polygon: shpPoly, segment: LineString):
    """
    Return the intersections of a ray and a polygon
    :param polygon: shapely.Polygon
    :param segment: shapely.LineString, (anchor, source)
    :return: coordinates of intersection
    """
    anchor, source = Point2(*segment.coords[0]), Point2(*segment.coords[1])
    ray = Ray2(source, source - anchor)
    poly_pts = list(polygon.exterior.coords)
    for i in range(len(poly_pts) - 1):
        poly_seg = Segment2(Point2(*poly_pts[i]), Point2(*poly_pts[i + 1]))
        intx = intersection(ray, poly_seg)
        if isinstance(intx, Point2):
            return intx.x(), intx.y()
    return None


def segment_intersection_with_polygon(polygon: shpPoly, segment: LineString):
    """
    Return the intersection of a line segment and a polygon
    :param polygon: shapely.Polygon
    :param segment: shapely.LineString, (source, target)
    :return: coordinates of intersection
    """
    source, target = Point2(*segment.coords[0]), Point2(*segment.coords[1])
    seg = Segment2(source, target)
    poly_pts = list(polygon.exterior.coords)
    for i in range(len(poly_pts) - 1):
        poly_seg = Segment2(Point2(*poly_pts[i]), Point2(*poly_pts[i + 1]))
        intx = intersection(seg, poly_seg)
        if isinstance(intx, Point2):
            return intx.x(), intx.y()
    return None


def get_nf_ng(graph, node, block_line, is_ray):
    """
    Compute the numbers n_f and n_g with respect to each edge or ray in an arrangement
    See M. Radermacher et al., “Geometric heuristics for rectilinear crossing minimization" for details
    :param graph: networkx.Graph, the graph model
    :param node: str, the name of the node being moved
    :param block_line: (str, str), names of nodes determining the blocking line for which n_f and n_g are computed
        if is_ray is True, nodes correspond to (directional origin node, starting node)
        if is_ray is False, nodes correspond to the terminal nodes of the edge
    :param is_ray: boolean, whether nodes correspond to a ray
    :return: (n_f, n_g) values
    """
    # Hf denotes the half-plane to the left side of the line
    # Hg denotes the half-plane to the right side of the line
    cnt_f = cnt_g = 0
    if not is_ray:
        source, target = block_line
        source_pt = (graph.nodes[source]['x'], graph.nodes[source]['y'])
        target_pt = (graph.nodes[target]['x'], graph.nodes[target]['y'])
        for neighbor in graph.neighbors(node):
            if neighbor in block_line:
                continue
            neighbor_pt = (graph.nodes[neighbor]['x'], graph.nodes[neighbor]['y'])
            if is_left(neighbor_pt, (source_pt, target_pt)) > 0:
                cnt_f += 1
            elif is_left(neighbor_pt, (source_pt, target_pt)) < 0:
                cnt_g += 1
    else:
        # for each neighboring node u and each node z != u
        # count the neighbors of z in both sides of the line uz
        anchor, source = block_line
        anchor_pt = (graph.nodes[anchor]['x'], graph.nodes[anchor]['y'])
        source_pt = (graph.nodes[source]['x'], graph.nodes[source]['y'])
        for neighbor in graph.neighbors(source):
            if neighbor != node and neighbor != anchor:
                neighbor_pt = (graph.nodes[neighbor]['x'], graph.nodes[neighbor]['y'])
                if is_left(neighbor_pt, (anchor_pt, source_pt)) > 0:
                    cnt_f += 1
                elif is_left(neighbor_pt, (anchor_pt, source_pt)) < 0:
                    cnt_g += 1
    return cnt_f, cnt_g


def build_dual_graph(arrangement, half_edge_deltas, half_edge_info):
    """
    Build the dual graph of the visibility region arrangement
    :param arrangement: skgeom.arrangement.Arrangement, the visibility region arrangement
    :param half_edge_deltas: dict<skgeom.arrangement.Halfedge, integer>
                                key: a half edge in the arrangement,
                                value: the delta value of the half edge
    :param half_edge_info: dict<skgeom.arrangement.Halfedge, str>
                                key: a half edge in the arrangement,
                                value: the information of the half edge
    :return networkx.DiGraph, the dual graph
    """
    dual_g = nx.DiGraph()
    for face in arrangement.faces:
        if not face.is_unbounded():
            dual_g.add_node(face)
            first_half_edge = None
            half_edge_circulator = face.outer_ccb
            next_half_edge = next(half_edge_circulator)
            while next_half_edge is not first_half_edge:
                if first_half_edge is None:
                    first_half_edge = next_half_edge
                neighbor_face = next_half_edge.twin().face()
                if not neighbor_face.is_unbounded():
                    dual_g.add_node(neighbor_face)
                    dual_g.add_edge(face, neighbor_face, delta=half_edge_deltas.get(next_half_edge, 0),
                                    info=half_edge_info.get(next_half_edge, ('', '')))
                next_half_edge = next(half_edge_circulator)
    return dual_g


def find_optimal_position(graph, node, plot_arrangement=False, plot_dual_graph=False, **kwargs):
    """
    Find the optimal position to mode the node
    :param graph: networkx.Graph
    :param node: str, the name of the node to be moved
    :param plot_arrangement: boolean, whether to draw the visibility region arrangement
    :param plot_dual_graph: boolean, whether to draw the dual graph of the visibility region arrangement
    :return: (float, float), coordinates of the optimal position
    """
    arrangement, he_deltas, he_info = create_arrangement(graph, node, **kwargs)
    if arrangement is None:
        return graph.nodes[node]['x'], graph.nodes[node]['y']
    dual_graph = build_dual_graph(arrangement, he_deltas, he_info)
    # find the starting face
    node_pt = Point2(graph.nodes[node]['x'], graph.nodes[node]['y'])
    node_location = arrangement.find(node_pt)
    start_face = None
    if isinstance(node_location, argmt.Face):
        if node_location.has_outer_ccb():
            start_face = node_location
    elif isinstance(node_location, argmt.Halfedge):
        if node_location.face().has_outer_ccb():
            start_face = node_location.face()
        else:
            start_face = node_location.twin().face()
    elif isinstance(node_location, argmt.Vertex):
        he_circulator = node_location.incident_halfedges
        first_he = None
        next_he = next(he_circulator)
        start_face = None
        while next_he != first_he:
            if first_he is None:
                first_he = next_he
            if next_he.face().has_outer_ccb():
                start_face = next_he.face()
                break
            next_he = next(he_circulator)
    if start_face is None:
        return (float(node_pt.x()), float(node_pt.y())), 0
    # breadth-first traverse of the dual graph to find the face with minimal crossings
    n_crossings = {start_face: 0}
    for head_face, tail_face in nx.bfs_edges(dual_graph, start_face):
        delta = dual_graph.edges[(head_face, tail_face)]['delta']
        n_crossings[tail_face] = n_crossings[head_face] + delta
    optimal_face = sorted(n_crossings.items(), key=lambda crs: crs[1])[0][0]
    if plot_dual_graph:
        plt_dg(dual_graph, start_face, optimal_face, n_crossings, save_figure=False)
    if plot_arrangement:
        plt_argmt(graph, node, arrangement, optimal_face, n_crossings, save_figure=False)
    if optimal_face is not start_face:
        return representative_point(optimal_face), n_crossings[optimal_face]
    else:
        return (float(node_pt.x()), float(node_pt.y())), 0
