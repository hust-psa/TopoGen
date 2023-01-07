import os
import math
import gzip
import pickle
import pandas as pd
import networkx as nx
import xml.etree.ElementTree as eT


def create_graph_from_ppnet(pp_net):
    """
    Create a networkx.Graph from a pandapower.network
    :param pp_net: pandapower.network, the network model to be converted
    :return: networkx.Graph
    """
    graph = nx.Graph()
    # add all real vertices, i.e., buses, to the graph
    pp_bus_aug = pp_net.bus.join(pp_net.bus_geodata)
    for i, row in pp_bus_aug.iterrows():
        graph.add_node(str(i), x=row['x'], y=row['y'], w=1)
    # for each three-winding transformer, add a virtual vertex corresponding to its neutral point
    n_real_v = len(graph.nodes)
    for i, row in pp_net.trafo3w.iterrows():
        hv_node, mv_node, lv_node = str(row['hv_bus']), str(row['mv_bus']), str(row['lv_bus'])
        hv_node_x, hv_node_y = graph.nodes[hv_node]['x'], graph.nodes[hv_node]['y']
        mv_node_x, mv_node_y = graph.nodes[mv_node]['x'], graph.nodes[mv_node]['y']
        lv_node_x, lv_node_y = graph.nodes[lv_node]['x'], graph.nodes[lv_node]['y']
        vir_node = str(n_real_v + i)
        vir_node_x, vir_node_y = (hv_node_x + mv_node_x + lv_node_x) / 3, (hv_node_y + mv_node_y + lv_node_y) / 3
        graph.add_node(vir_node, x=vir_node_x, y=vir_node_y, w=1)
        # add three equivalent lines connecting the hv, mv, and lv buses and the virtual vertex
        for term_node in [hv_node, mv_node, lv_node]:
            vir_node_pos = (vir_node_x, vir_node_y)
            term_node_pos = (graph.nodes[term_node]['x'], graph.nodes[term_node]['y'])
            weight = 1 / math.dist(vir_node_pos, term_node_pos)
            graph.add_edge(vir_node, term_node, w=weight)
    # add all real edges, i.e., transmission lines and two-winding transformers
    for i, row in pp_net.line.iterrows():
        f_node, t_node = str(row['from_bus']), str(row['to_bus'])
        if not graph.has_edge(f_node, t_node):
            f_node_pos = (graph.nodes[f_node]['x'], graph.nodes[f_node]['y'])
            t_node_pos = (graph.nodes[t_node]['x'], graph.nodes[t_node]['y'])
            weight = 1 / math.dist(f_node_pos, t_node_pos)
            graph.add_edge(f_node, t_node, w=weight)
    for i, row in pp_net.trafo.iterrows():
        hv_node, lv_node = str(row['hv_bus']), str(row['lv_bus'])
        if not graph.has_edge(hv_node, lv_node):
            hv_node_pos = (graph.nodes[hv_node]['x'], graph.nodes[hv_node]['y'])
            lv_node_pos = (graph.nodes[lv_node]['x'], graph.nodes[lv_node]['y'])
            weight = 1 / math.dist(hv_node_pos, lv_node_pos)
            graph.add_edge(hv_node, lv_node, w=weight)
    return graph


def create_graph_from_cimg(cimg_path):
    """
    Create a networkx.Graph from a CIMG file
    :param cimg_path: str, path to the CIMG file
    :return: networkx.Graph
    """
    graph = nx.Graph()
    cimg_tree = eT.parse(cimg_path)
    for substation in cimg_tree.findall('./Layer/Substation'):
        node_id = substation.get('id')
        x, y = float(substation.get('x')), float(substation.get('y'))
        graph.add_node(node_id, x=x, y=y, w=1)
    for acline in cimg_tree.findall('.Layer/ACLine'):
        head_node, tail_node = acline.get('link').split(';')
        head_node_pos = (graph.nodes[head_node]['x'], graph.nodes[head_node]['y'])
        tail_node_pos = (graph.nodes[tail_node]['x'], graph.nodes[tail_node]['y'])
        weight = 1 / math.dist(head_node_pos, tail_node_pos)
        graph.add_edge(head_node, tail_node, w=weight)
    name_map = dict()
    n_node = 0
    for n in graph.nodes:
        name_map[n] = str(n_node)
        n_node += 1
    nx.relabel_nodes(graph, name_map, copy=False)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def create_graph_from_entso_csvs(bus_csv_path, line_csv_path):
    """
    Create a networkx.Graph from ENTSO-E buses and lines csv files
    :param bus_csv_path: str, path to the buses csv file
    :param line_csv_path: str, path to the lines csv file
    :return: networkx.Graph
    """
    graph = nx.Graph()
    bus_csv = pd.read_csv(bus_csv_path)
    line_csv = pd.read_csv(line_csv_path)
    for idx, row in bus_csv.iterrows():
        node_id = row['bus_id']
        x, y = row['x'], row['y']
        graph.add_node(node_id, x=x, y=y, w=1)
    for idx, row in line_csv.iterrows():
        head_node, tail_node = row['bus0'], row['bus1']
        head_node_pos = (graph.nodes[head_node]['x'], graph.nodes[head_node]['y'])
        tail_node_pos = (graph.nodes[tail_node]['x'], graph.nodes[tail_node]['y'])
        weight = 1 / math.dist(head_node_pos, tail_node_pos)
        graph.add_edge(head_node, tail_node, w=weight)
    name_map = dict()
    n_node = 0
    for n in graph.nodes:
        name_map[n] = str(n_node)
        n_node += 1
    nx.relabel_nodes(graph, name_map, copy=False)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def load_nx_graph(path):
    """
    Load a previously saved networkx.Graph model from the specified path
    :param path: str, the path to read
    :return: networkx.Graph, the model dumped to the path before
    """
    if os.path.isfile(path) and os.path.splitext(path)[-1] == '.gexf':
        return nx.read_gexf(path=path)
    else:
        return None


def save_nx_graph(graph, path):
    """
    Save a networkx.Graph model to the specified path
    :param graph: networkx.Graph, the model to dump
    :param path: str, the path to dump
    :return: None
    """
    return nx.write_gexf(graph, path)


def load_model(path):
    """
    Load a previously saved object from the specified path
    :param path: str, the path to read
    :return: object
    """
    if not os.path.isfile(path):
        return None
    with gzip.open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_model(obj, path):
    """
    Save an object to the specified path
    :param obj: object, the model to dump
    :param path: str, the path to dump
    :return: None
    """
    split = os.path.split(path)
    if split[0] != '':
        os.makedirs(split[0], exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump(obj, f)
    pass
