import math
import networkx as nx
from ground.base import get_context
from bentley_ottmann.planar import segments_intersections

context = get_context()
Point, Segment = context.point_cls, context.segment_cls


def find_crossings(graph):
    """
    Find all crossings in a networkx.Graph
    :param graph: networkx.Graph, the graph with crossings
    :return: list of tuples, [((crossed edge a, crossed edge b), coordinates of crossing)...]
    """
    # Convert graph edges into line segments
    segments = []
    segment_idx_to_edge = []
    for edge in graph.edges:
        node_i, node_j = (graph.nodes[label] for label in edge)
        (node_i_x, node_i_y), (node_j_x, node_j_y) = ((n.get('x'), n.get('y')) for n in (node_i, node_j))
        segments.append(Segment(Point(node_i_x, node_i_y), Point(node_j_x, node_j_y)))
        segment_idx_to_edge.append(edge)
    # Find crossings of line segments
    crossings = []
    for x_segment_idxs, x_points in segments_intersections(segments).items():
        seg_i, seg_j = segments[x_segment_idxs[0]], segments[x_segment_idxs[1]]
        x_pt = x_points[0]
        if x_pt != seg_i.start and x_pt != seg_i.end and x_pt != seg_j.start and x_pt != seg_j.end:
            edge_i, edge_j = segment_idx_to_edge[x_segment_idxs[0]], segment_idx_to_edge[x_segment_idxs[1]]
            crossings.append(((edge_i, edge_j), (float(x_pt.x), float(x_pt.y))))
    return crossings


def count_crossings(graph):
    """
    Return the number of crossings in a networkx.Graph
    :param graph: networkx.Graph, the graph with crossings
    :return: int, the number of crossings
    """
    return len(find_crossings(graph))


def crossings_by_edge(graph, crossings=None):
    """
    Get crossings by edge
    :param graph: networkx.Graph, the graph with crossings
    :param crossings: list of tuples, pre-computed crossing data
    :return:
    """
    if not crossings:
        crossings = find_crossings(graph)
    results = dict()
    for x_edges, _ in crossings:
        for i in [0, 1]:
            edge_i, edge_j = x_edges[i], x_edges[i - 1]
            if edge_i not in results.keys():
                results[edge_i] = [edge_j]
            else:
                crossed_edges = results[edge_i]
                results[edge_i] = crossed_edges + [edge_j]
    return results
