import numpy as np
from skgeom import Point2, Segment2
from skgeom import arrangement as argmt


def incident_edges_by_order(graph, node):
    """
    Return incident edges of a node by the counterclockwise order
    :param graph: networkx.Graph
    :param node: str, the name of the node
    :return: list of tuples, a list of incident edges in order
    """
    incident_edges = dict()
    for edge in graph.edges(node):
        (ux, uy), (vx, vy) = ((graph.nodes[n]['x'], graph.nodes[n]['y']) for n in edge)
        rad = np.arctan2((vy - uy), (vx - ux))
        if rad < 0:
            rad = rad % (2 * np.pi)
        incident_edges[edge] = rad
    return [e[0] for e in sorted(incident_edges.items(), key=lambda x: x[1])]


def radian_between_vectors(vector_i, vector_j):
    """
    Compute the radian of the included angle between two vectors
    :param vector_i: tuple of floats, coordinates of vector 1
    :param vector_j: tuple of floats, coordinates of vector 2
    :return: float, radian of the included angle
    """
    rad = np.arctan2(vector_j[1], vector_j[0]) - np.arctan2(vector_i[1], vector_i[0])
    if rad < 0:
        rad = rad % (2 * np.pi)
    return rad


def node_direction(u_pos, v_pos, k):
    """
    Find the sector where node v locates with respect to node u
    :param u_pos: tuple of floats, coordinates of node u
    :param v_pos: tuple of floats, coordinates of node v
    :param k: int, the number of directions
    :return: int, the sector where node v locates with respect to node u
    """
    (ux, uy), (vx, vy) = u_pos, v_pos
    rad = np.arctan2((vy - uy), (vx - ux)) / np.pi
    return int(np.floor(k * rad + 0.5) % (2 * k))


def non_adjacent_nodes_in_faces(graph):
    non_adjacent_node_pairs = []
    node_pt_map = []
    for node, attr in graph.nodes(data=True):
        pt = Point2(attr['x'], attr['y'])
        node_pt_map.append((node, pt))

    arrangement = argmt.Arrangement()
    for edge in graph.edges:
        seg = Segment2(*(Point2(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in edge))
        arrangement.insert(seg)
    for face in arrangement.faces:
        nodes_in_face = []
        if face.is_unbounded():
            he_circulator = list(face.holes)[0]
        else:
            he_circulator = face.outer_ccb
        first_he = None
        next_he = next(he_circulator)
        while next_he != first_he:
            if first_he is None:
                first_he = next_he
            for node, pt in node_pt_map:
                if pt == next_he.source().point():
                    if node not in nodes_in_face:
                        nodes_in_face.append(node)
                    break
            for node, pt in node_pt_map:
                if pt == next_he.target().point():
                    if node not in nodes_in_face:
                        nodes_in_face.append(node)
                    break
            next_he = next(he_circulator)
        for i in range(len(nodes_in_face) - 1):
            for j in range(i + 1, len(nodes_in_face)):
                node_i, node_j = nodes_in_face[i], nodes_in_face[j]
                if not graph.has_edge(node_i, node_j):
                    if ((node_i, node_j) not in non_adjacent_node_pairs and
                            (node_j, node_i) not in non_adjacent_node_pairs):
                        non_adjacent_node_pairs.append((node_i, node_j))
    return non_adjacent_node_pairs


def non_adjacent_edge_pairs_in_faces(graph):
    """
    Return non-adjacent edge pairs in all faces of a graph
    :param graph: networkx.Graph
    :return: list of tuples, non-adjacent edge pairs
    """
    non_adjacent_edge_pairs = []
    arrangement = argmt.Arrangement()
    edge_seg_map = []
    deg_1_nodes = [node for node in graph.nodes if graph.degree(node) == 1]
    oppo_nodes = [list(graph.edges(node))[0][1] for node in deg_1_nodes]
    for edge in graph.edges:
        if edge[0] in deg_1_nodes or edge[1] in deg_1_nodes:
            continue
        seg = Segment2(*(Point2(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in edge))
        edge_seg_map.append((edge, seg))
        edge_seg_map.append((edge, seg.opposite()))
        arrangement.insert(seg)
    for face in arrangement.faces:
        edges_ccb = []
        nodes_ccb = []
        if face.is_unbounded():
            half_edge_circulator = list(face.holes)[0]
        else:
            half_edge_circulator = face.outer_ccb
        first_half_edge = None
        next_half_edge = next(half_edge_circulator)
        while next_half_edge != first_half_edge:
            if first_half_edge is None:
                first_half_edge = next_half_edge
            for edge, seg in edge_seg_map:
                if seg == next_half_edge.curve():
                    if edge not in edges_ccb and edge[::-1] not in edges_ccb:
                        edges_ccb.append(edge)
                    nodes_ccb.extend(list(edge))
                    break
            next_half_edge = next(half_edge_circulator)
        nodes_ccb = list(set(nodes_ccb))
        for i in range(len(edges_ccb) - 2):
            for j in range(i + 2, len(edges_ccb)):
                if j - i != -1 % len(edges_ccb):
                    if len(set(edges_ccb[i] + edges_ccb[j])) != 4:
                        continue
                    if ((edges_ccb[i], edges_ccb[j]) not in non_adjacent_edge_pairs and
                            (edges_ccb[j], edges_ccb[i]) not in non_adjacent_edge_pairs):
                        non_adjacent_edge_pairs.append((edges_ccb[i], edges_ccb[j]))
        for deg_1_node, oppo_node in zip(deg_1_nodes, oppo_nodes):
            if oppo_node in nodes_ccb:
                pendant_edge = (deg_1_node, oppo_node)
                for ccb_edge in edges_ccb:
                    if oppo_node not in ccb_edge:
                        if ((ccb_edge, pendant_edge) not in non_adjacent_edge_pairs and
                                (pendant_edge, ccb_edge) not in non_adjacent_edge_pairs):
                            non_adjacent_edge_pairs.append((pendant_edge, ccb_edge))
    return non_adjacent_edge_pairs
