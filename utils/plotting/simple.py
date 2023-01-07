import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skgeom import Point2
from skgeom.draw import draw
from matplotlib.patches import Polygon as PolygonPatch
from utils.crossing import find_crossings
from .save import save_plot


def plot_graph(graph, figsize=10, node_size=80, highlight_node=None, draw_crossing=True,
               draw_area=False, highlight_area=None, draw_cell=False, cell=None, area=None, draw_label=True,
               save_figure=True, save_path='graph.pdf'):
    """
    Draw a networkx.Graph and optionally save it to a file
    :param graph: networkx.Graph
    :param figsize: figure size in inches
    :param node_size: node size
    :param highlight_node: list of str, names of the nodes to be highlighted
    :param draw_crossing: boolean, whether to highlight crossings
    :param draw_area: boolean, whether to draw nodes in different areas with different colors
    :param highlight_area: list of int, areas not in the list are drawn as gray
    :param draw_cell: boolean, whether to draw the partition cell
    :param cell: numpy.array of shape (n, 2), coordinates of cell nodes
    :param area: int, index of area
    :param draw_label: boolean, whether to draw labels
    :param save_figure: boolean, whether to save the drawing to an .pdf file
    :param save_path: str, the path to save the figure
    :return: None
    """
    if not highlight_node:
        highlight_node = []
    pos, labels, sizes, colors = dict(), dict(), [], []
    n_dummy = 0
    for node in graph.nodes:
        pos[node] = (graph.nodes[node]['x'], graph.nodes[node]['y'])
        labels[node] = node
        if str(node).lower().startswith('d') or str(node).lower().startswith('i'):
            sizes.append(0.8 * node_size)
        else:
            sizes.append(node_size)
        if node in highlight_node:
            colors.append('tab:red')
        else:
            if not draw_area:
                if str(node).lower().startswith('d'):
                    colors.append('tab:brown')
                    n_dummy += 1
                elif str(node).lower().startswith('i'):
                    colors.append('tab:gray')
                else:
                    colors.append('tab:blue')
            else:
                if highlight_area is None:
                    colors.append(graph.nodes[node]['area'])
                else:
                    if graph.nodes[node]['area'] == highlight_area:
                        colors.append('tab:blue')
                    else:
                        colors.append('tab:gray')

    if cell is None:
        x_min, y_min = np.min(np.array(list(pos.values())), axis=0)
        x_max, y_max = np.max(np.array(list(pos.values())), axis=0)
    else:
        x_min, y_min = np.min(cell, axis=0)
        x_max, y_max = np.max(cell, axis=0)
    x_delta, y_delta = (x_max - x_min) / 15, (y_max - y_min) / 15
    x_min, x_max = x_min - x_delta, x_max + x_delta
    y_min, y_max = y_min - y_delta, y_max + y_delta
    figsize = (figsize, figsize * (y_max - y_min) / (x_max - x_min) + 0.8)
    plt.figure(1, figsize=figsize, dpi=120)
    nx.draw(graph, pos, node_size=sizes, with_labels=draw_label, labels=labels, font_size=node_size / 16,
            font_color='white', node_color=colors)
    annotation = []
    if area is not None:
        annotation.append('Area = %s' % str(area))
    if draw_crossing:
        crossings = find_crossings(graph)
        for _, (x, y) in crossings:
            draw(Point2(x, y), color='tab:orange', s=20, zorder=2)
        annotation.append('Crossings = %d' % (len(crossings) + n_dummy))
    if draw_cell and cell is not None:
        ax = plt.gca()
        ax.add_patch(PolygonPatch(cell, alpha=0.2))
    anno = ', '.join(annotation)
    plt.text(0.05, 0.95, s=anno, ha='left', va='top', transform=plt.gca().transAxes, color='tab:orange')
    if save_figure:
        figure = plt.gcf()
        save_plot(figure, save_path)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    pass


def plot_comparison_graph(raw_graph, crossing_reduced_graph, optimized_graph, force_directed_graph,
                          save_figure=True, save_path='graph.pdf'):
    """
    Plot a 2x2 diagram for comparison
    :param raw_graph: upper left graph
    :param crossing_reduced_graph: upper right graph
    :param optimized_graph: lower left graph
    :param force_directed_graph: lower right graph
    :param save_figure: boolean, whether to save the figure as a file
    :param save_path: str, path to save the figure
    :return: None
    """
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 7.5), layout='constrained', linewidth=4,
                            gridspec_kw={'hspace': 0, 'wspace': 0})

    def _plot_graph(graph, ax):
        plt.sca(ax)
        ax.margins(0)
        pos, sizes, colors = dict(), [], []
        dummy_pos, inter_pos = [], []
        for node in graph.nodes:
            pos[node] = (graph.nodes[node]['x'], graph.nodes[node]['y'])
            colors.append('tab:blue')
            if str(node).lower().startswith('d'):
                sizes.append(0)
                dummy_pos.append(pos[node])
            elif str(node).lower().startswith('i'):
                sizes.append(0)
                inter_pos.append(pos[node])
            else:
                sizes.append(10)
        nx.draw(graph, pos, node_size=sizes, node_color=colors, with_labels=False, width=0.5)
        crossings = find_crossings(graph)
        for _, (x, y) in crossings:
            draw(Point2(x, y), color='tab:orange', s=4, zorder=2)
        for x, y in dummy_pos:
            draw(Point2(x, y), color='tab:brown', s=4, zorder=2)
        plt.axis('on')
        plt.setp(ax.spines.values(), linewidth=1)
        pass

    _plot_graph(raw_graph, axs[0][0])
    _plot_graph(crossing_reduced_graph, axs[0][1])
    _plot_graph(optimized_graph, axs[1][0])
    _plot_graph(force_directed_graph, axs[1][1])

    if save_figure:
        save_plot(fig, save_path)
    plt.show()
    pass
