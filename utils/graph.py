import json
import numpy as np


def cnt_cut_edges(graph):
    """
    Return the total number of cut edges
    :param graph: networkx.Graph, the graph model
    :return: int, the total number of cut edges
    """
    return np.sum([1 for (u, v) in graph.edges if graph.nodes[u]['area'] != graph.nodes[v]['area']])


def cnt_nodes_by_area(graph):
    """
    Return the number of nodes in each area
    :param graph: networkx.Graph, the graph model
    :return: dict[area] = the number of nodes
    """
    cnt_nodes = dict()
    for node, attr in graph.nodes(data=True):
        if attr['area'] not in cnt_nodes:
            cnt_nodes[attr['area']] = 0
        cnt_nodes[attr['area']] += 1
    return cnt_nodes


def weights_by_area(graph):
    """
    Return the total weight of nodes in each area
    :param graph: networkx.Graph, the graph model
    :return: dict[area] = the total weight of nodes
    """
    weights = dict()
    for node, attr in graph.nodes(data=True):
        if attr['area'] not in weights:
            weights[attr['area']] = 0
        weights[attr['area']] += attr['w']
    return weights


def degrees_by_area(graph, node):
    """
    Return the degree of a node, divided by area
    :param graph: networkx.Graph, the graph model
    :param node: str, the name of the node
    :return: dict[area] = degree
    """
    degrees = dict()
    adj_nodes = list(graph.neighbors(node))
    for adj_node in adj_nodes:
        adj_area = graph.nodes[adj_node]['area']
        if adj_area not in degrees:
            degrees[adj_area] = 0
        degrees[adj_area] += 1
    return degrees


def irrelevant_edges(graph, node):
    """
    Return the edges irrelevant to a node
    :param graph: networkx.Graph, the graph model
    :param node: str, the name of the node
    :return: list of tuples, [(node_i, node_j)...], all irrelevant edges
    """
    results = graph.edges
    for active_edge in graph.edges(node):
        results = results - {active_edge} - {active_edge[::-1]}
    return results


def add_dummy_node(graph, point, edges, label=None):
    """
    Add a dummy node to the specified edges
    :param graph: networkx.Graph
    :param point: tuple of floats, coordinates of the dummy node
    :param edges: list of edges, incident edges of the dummy node
    :param label: node label
    :return:
        - networkx.Graph, the graph after adding the dummy node
        - str, the label of the dummy node
    """
    if len(edges) == 1:
        if label is None:
            n_inter = 0
            for n in graph.nodes:
                if n.startswith('I'):
                    n_inter += 1
            node_label = 'I%d' % n_inter
        else:
            node_label = 'I%s' % label
    else:
        if label is None:
            n_dummy = 0
            for n in graph.nodes:
                if n.startswith('D'):
                    n_dummy += 1
            node_label = 'D%d' % n_dummy
        else:
            node_label = 'D%s' % label
    new_graph = graph.copy()
    x, y = point
    new_graph.add_node(node_label, x=x, y=y, w=1)
    edge_pairs = []
    for edge in edges:
        node_i, node_j = edge
        if 'area' in new_graph.nodes[node_i]:
            new_graph.nodes[node_label]['area'] = new_graph.nodes[node_i]['area']
        new_graph.remove_edge(node_i, node_j)
        new_graph.add_edge(node_i, node_label)
        new_graph.add_edge(node_label, node_j)
        for oppo_node in edge:
            if oppo_node.startswith('D'):
                oppo_node_edge_pairs = new_graph.nodes[oppo_node]['ep']
                old_node = node_j if oppo_node == node_i else node_i
                new_edge_pairs = []
                for edge_1, edge_2 in json.loads(oppo_node_edge_pairs):
                    if old_node in edge_1:
                        edge_1[edge_1.index(old_node)] = node_label
                    if old_node in edge_2:
                        edge_2[edge_2.index(old_node)] = node_label
                    new_edge_pairs.append([edge_1, edge_2])
                new_graph.nodes[oppo_node]['ep'] = json.dumps(new_edge_pairs)
        edge_pairs.append([[node_i, node_label], [node_label, node_j]])
    new_graph.nodes[node_label]['ep'] = json.dumps(edge_pairs)
    return new_graph, node_label
