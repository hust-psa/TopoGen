import os
import time
import networkx as nx
from utils.interface import save_model, load_model
from utils.plotting import plot_graph


def force_directed_diagram(graph, iteration=1000, save_diagram=True, save_dir='files', **kwargs):
    """
    Plot the force-directed layout of a graph
    :param graph: networkx.Graph to be plotted
    :param iteration: int, number of iterations of the force-directed algorithm
    :param save_diagram: boolean, whether to save the diagram
    :param save_dir: str, path to save
    :return: None
    """
    saved_model = load_model(os.path.join(save_dir, 'force.mdl'))
    if saved_model is not None:
        fd_graph, fd_time = saved_model
    else:
        start = time.time()
        fd_graph = graph.copy()
        pos = nx.spring_layout(fd_graph, iterations=iteration)
        for node, (x, y) in pos.items():
            fd_graph.nodes[node]['x'] = x
            fd_graph.nodes[node]['y'] = y
        end = time.time()
        fd_time = end - start
        if save_diagram:
            save_model((fd_graph, fd_time), path=os.path.join(save_dir, 'force.mdl'))
    print('Force-direction layout time: %g s' % fd_time)
    plot_graph(fd_graph, save_figure=save_diagram, save_path=os.path.join(save_dir, 'force.pdf'), **kwargs)
    pass
