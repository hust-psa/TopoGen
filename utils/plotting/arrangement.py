import networkx as nx
import matplotlib.pyplot as plt
from skgeom.draw import draw
from utils.arrangement import representative_point
from .save import save_plot


def plot_dual_graph(dual_graph, start_face, optimal_face, face_metrics, save_figure=True, save_path='dual_graph.pdf'):
    """
    Draw the dual graph of an skgeom.arrangement.Arrangement
    :param dual_graph: networkx.Graph, the dual graph
    :param start_face: skgeom.arrangement.Face, the node corresponding to the starting face is colored red
    :param optimal_face: skgeom.arrangement.Face, the node corresponding to the optimal face is colored green
    :param face_metrics: list of integers, tests of all faces
    :param save_figure: boolean, whether to save the drawing to an .pdf file
    :param save_path: str, the path to save the figure
    :return: None
    """
    plt.figure(1, figsize=(15, 12), dpi=120)
    pos = nx.spring_layout(dual_graph)
    node_colors = []
    for n in dual_graph.nodes:
        if n is start_face:
            node_colors.append('tab:red')
        elif n is optimal_face:
            node_colors.append('tab:green')
        else:
            node_colors.append('tab:blue')
    nx.draw(dual_graph, pos, node_color=node_colors, node_size=50,
            with_labels=True, labels=face_metrics, font_size=8, font_color='white')
    edge_deltas = nx.get_edge_attributes(dual_graph, 'delta')
    edge_info = nx.get_edge_attributes(dual_graph, 'info')
    edge_labels = dict()
    for e in edge_deltas:
        e_name = '%s: %d' % (edge_info[e][1], edge_deltas[e])
        edge_labels[(e[0], e[1])] = e_name
    nx.draw_networkx_edge_labels(dual_graph, pos, edge_labels=edge_labels, label_pos=0.25, font_size=6)
    if save_figure:
        figure = plt.gcf()
        save_plot(figure, save_path)
    plt.show()
    pass


def plot_arrangement(graph, node, arrangement, optimal_face, face_metrics, save_figure=True, save_path='argmt.pdf'):
    """
    Draw the arrangement created from visibility region analysis on moving a node
    :param graph: networkx.Graph, the graph model
    :param node: str, the name of the node being moved
    :param arrangement: skgeom.arrangement.Arrangement, the created arrangement
    :param optimal_face: skgeom.arrangement.Face, the optimal face for moving the node
    :param face_metrics: list of integers, tests of all nodes
    :param save_figure: boolean, whether to save the drawing to an .pdf file
    :param save_path: str, the path to save the figure
    :return: None
    """
    plt.figure(1, figsize=(15, 12), dpi=120)
    if face_metrics is not None:
        for face in arrangement.faces:
            if not face.has_outer_ccb():
                continue
            plt.text(*representative_point(face), face_metrics[face], size=6, color='red')
    for half_edge in arrangement.halfedges:
        draw(half_edge.curve(), visible_point=False, color='gray', linewidth=0.5)
    if optimal_face is not None:
        half_edge_circulator = optimal_face.outer_ccb
        first_half_edge = None
        next_half_edge = next(half_edge_circulator)
        while next_half_edge != first_half_edge:
            if first_half_edge is None:
                first_half_edge = next_half_edge
                draw(next_half_edge.curve(), visible_point=False, color='tab:orange', linewidth=1.5)
            else:
                draw(next_half_edge.curve(), visible_point=False, color='tab:orange', linewidth=1.5)
            next_half_edge = next(half_edge_circulator)
    if graph is not None:
        pos_x = nx.get_node_attributes(graph, 'x')
        pos_y = nx.get_node_attributes(graph, 'y')
        pos = dict()
        for idx, x in pos_x.items():
            pos[idx] = (x, pos_y[idx])
        node_colors = ['tab:red' if n == node else 'tab:olive' for n in graph.nodes]
        nx.draw(graph, pos, node_size=40, with_labels=True, font_size=6,
                node_color=node_colors, edge_color='tab:olive')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    if save_figure:
        figure = plt.gcf()
        save_plot(figure, save_path)
    plt.show()
    pass
