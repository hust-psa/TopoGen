import math
import time
from utils.interface import save_model as save_graph_model
from utils.crossing import count_crossings, crossings_by_edge
from utils.graph import add_dummy_node
from utils.plotting import plot_graph
from crossing.moving import find_optimal_position


def vertex_movement(graph, order_method='sq', descending=True, plot_intermediate_graphs=False,
                    save_model=True, save_path='cross.mdl', **kwargs):
    """
    Move all vertices in the graph
    :param graph: networkx.Graph
    :param order_method: str, ordering method, can be 'sum', 'sq', or 'log'
    :param descending: boolean, move vertices according to the descending order of the selected metric
    :param plot_intermediate_graphs: boolean, plot graph after moving each node
    :param save_model: boolean, whether to save the reduced-crossing graph model
    :param save_path: str, the path to save the model
    :return: networkx.Graph, the graph after crossing reduction
    """
    graph = graph.copy()
    order_method = order_method.lower()
    if order_method not in ['sum', 'sq', 'log']:
        raise TypeError('Invalid vertex movement ordering method')

    start = time.time()
    # move all nodes in the graph
    nodes_to_move = list(graph.nodes)
    # determine the order of node moving
    total_crossings = count_crossings(graph)
    crossings = crossings_by_edge(graph)
    node_crossing_metrics = dict()
    for node in nodes_to_move:
        metric = 0
        for incident_edge in graph.edges(node):
            if incident_edge in crossings:
                if order_method == 'sum':
                    metric += len(crossings[incident_edge])
                elif order_method == 'sq':
                    metric += len(crossings[incident_edge]) ** 2
                else:
                    metric += math.log2(len(crossings[incident_edge]) + 1)
        node_crossing_metrics[node] = metric

    # move vertices to their optimal positions
    print('Initial number of crossings: %d' % total_crossings)
    n_cnt = 0
    for node, _ in sorted(node_crossing_metrics.items(), key=lambda i: i[1], reverse=descending):
        (x, y), delta = find_optimal_position(graph, node, plot_arrangement=plot_intermediate_graphs,
                                              plot_dual_graph=plot_intermediate_graphs, **kwargs)
        graph.nodes[node]['x'] = x
        graph.nodes[node]['y'] = y

        n_cnt += 1
        total_crossings = count_crossings(graph)
        print('  Interation %d: %d' % (n_cnt, total_crossings))
        if plot_intermediate_graphs:
            plot_graph(graph, highlight_node=node, save_figure=False)
    end = time.time()

    total_time = end - start
    if save_model:
        save_graph_model((graph, total_time), save_path)
    return graph, total_time


def edge_insertion(graph, move_crossed_neighbors=True, order_method='sq', descending=True,
                   plot_intermediate_graphs=False, save_model=True, save_path='cross.mdl', **kwargs):
    """
    Reinsert crossed edges one by one
    :param graph: networkx.Graph
    :param move_crossed_neighbors: boolean, whether to move terminal nodes of crossed edges
    :param order_method: str, ordering method, can be 'sum', 'sq', or 'log'
    :param descending: boolean, move vertices according to the descending order of the selected metric
    :param plot_intermediate_graphs: boolean, plot graph after moving each node
    :param save_model: boolean, whether to save the reduced-crossing graph model
    :param save_path: str, the path to save the model
    :return: networkx.Graph, the graph after crossing reduction
    """
    graph = graph.copy()
    order_method = order_method.lower()
    if order_method not in ['sum', 'sq', 'log']:
        raise TypeError('Invalid vertex movement ordering method')

    start = time.time()
    # find all crossings in the graph
    total_crossings = count_crossings(graph)
    crossings = crossings_by_edge(graph)
    updated_crossings = crossings
    print('Initial number of crossings: %d' % total_crossings)
    n_cnt = 0
    while len(crossings) > 0:
        edge, x_edges = sorted(crossings.items(), key=lambda e: len(e[1]), reverse=True)[0]
        if edge in updated_crossings:
            # find all relevant nodes to be moved
            nodes_to_move = {edge[0], edge[1]}
            if move_crossed_neighbors:
                for x_edge in x_edges:
                    nodes_to_move = nodes_to_move.union({x_edge[0], x_edge[1]})
            # determine the order of node movement
            node_crossing_metrics = dict()
            for node in nodes_to_move:
                metric = 0
                for incident_edge in graph.edges(node):
                    if incident_edge in crossings:
                        if order_method == 'sum':
                            metric += len(crossings[incident_edge])
                        elif order_method == 'sq':
                            metric += len(crossings[incident_edge]) ** 2
                        else:
                            metric += math.log2(len(crossings[incident_edge]) + 1)
                node_crossing_metrics[node] = metric
            # move relevent vertices to their optimal positions
            for node, _ in sorted(node_crossing_metrics.items(), key=lambda i: i[1], reverse=descending):
                (x, y), delta = find_optimal_position(graph, node, plot_arrangement=plot_intermediate_graphs,
                                                      plot_dual_graph=plot_intermediate_graphs, **kwargs)
                graph.nodes[node]['x'] = x
                graph.nodes[node]['y'] = y
                total_crossings += delta

            n_cnt += 1
            total_crossings = count_crossings(graph)
            updated_crossings = crossings_by_edge(graph)
            print('  Interation %d: %d' % (n_cnt, total_crossings))
            if plot_intermediate_graphs:
                plot_graph(graph, save_figure=False)

        del crossings[edge]
        for x_edge in x_edges:
            crossed_edges = crossings[x_edge]
            crossed_edges.remove(edge)
            if len(crossed_edges) > 0:
                crossings[x_edge] = crossed_edges
            else:
                del crossings[x_edge]
    end = time.time()

    total_time = (end - start)
    if save_model:
        save_graph_model((graph, total_time), save_path)
    return graph, total_time


def move_inter_node(graph, plot_intermediate_graphs=False, order_method='sq', descending=True,
                    save_model=True, save_path='cross.mdl', **kwargs):
    """
    Add an intemediate node to each crossing edge and move it along with terminal nodes
    :param graph: networkx.Graph
    :param plot_intermediate_graphs: boolean, plot graph after moving each node
    :param order_method: str, ordering method, can be 'sum', 'sq', or 'log'
    :param descending: boolean, move vertices according to the descending order of the selected metric
    :param save_model: boolean, whether to save the reduced-crossing graph model
    :param save_path: str, the path to save the model
    :return: networkx.Graph, the graph after crossing reduction
    """
    graph = graph.copy()
    order_method = order_method.lower()
    if order_method not in ['sum', 'sq', 'log']:
        raise TypeError('Invalid vertex movement ordering method')

    start = time.time()
    # find all crossings in the graph
    total_crossings = count_crossings(graph)
    crossings = crossings_by_edge(graph)
    updated_crossings = crossings
    print('Initial number of crossings: %d' % total_crossings)
    n_cnt = 1
    n_inter_node_added = 0
    total_time = 0
    while len(crossings) > 0:
        edge, x_edges = sorted(crossings.items(), key=lambda e: len(e[1]), reverse=True)[0]
        if edge in updated_crossings:
            print('  Iteration %d: Moving %s -> ' % (n_cnt, str(edge)), end='')
            # find all relevant nodes to be moved
            nodes_to_move = []
            for terminal in edge:
                if 'type' in graph.nodes[terminal]:
                    if graph.nodes[terminal]['type'] == 'internal':
                        nodes_to_move.append(terminal)
                else:
                    nodes_to_move.append(terminal)
            # determine the order of node movement
            node_crossing_metrics = dict()
            for node in nodes_to_move:
                metric = 0
                for incident_edge in graph.edges(node):
                    if incident_edge in crossings:
                        if order_method == 'sum':
                            metric += len(crossings[incident_edge])
                        elif order_method == 'sq':
                            metric += len(crossings[incident_edge]) ** 2
                        else:
                            metric += math.log2(len(crossings[incident_edge]) + 1)
                node_crossing_metrics[node] = metric

            graph_backup = graph.copy()
            n_crossings_before = total_crossings
            for node, _ in sorted(node_crossing_metrics.items(), key=lambda i: i[1], reverse=descending):
                (x, y), _ = find_optimal_position(graph, node, plot_arrangement=plot_intermediate_graphs,
                                                  plot_dual_graph=plot_intermediate_graphs, **kwargs)
                graph.nodes[node]['x'] = x
                graph.nodes[node]['y'] = y
            n_crossings_after = count_crossings(graph)
            if n_crossings_after >= n_crossings_before:
                total_crossings = n_crossings_before
                graph = graph_backup
                node_i, node_j = edge
                node_i_x, node_i_y = graph.nodes[node_i]['x'], graph.nodes[node_i]['y']
                node_j_x, node_j_y = graph.nodes[node_j]['x'], graph.nodes[node_j]['y']
                mid_pt = ((node_i_x + node_j_x) / 2, (node_i_y + node_j_y) / 2)
                tmp_graph, inter_node = add_dummy_node(graph, mid_pt, [edge], label=n_inter_node_added)
                (x, y), delta = find_optimal_position(tmp_graph, inter_node, plot_arrangement=plot_intermediate_graphs,
                                                      plot_dual_graph=plot_intermediate_graphs, **kwargs)
                if delta < 0:
                    tmp_graph.nodes[inter_node]['x'] = x
                    tmp_graph.nodes[inter_node]['y'] = y
                    graph = tmp_graph
                    n_inter_node_added += 1
                    total_crossings += delta
            else:
                total_crossings = n_crossings_after

            n_cnt += 1
            updated_crossings = crossings_by_edge(graph)
            print(total_crossings)
            if plot_intermediate_graphs:
                plot_graph(graph, save_figure=False)

        del crossings[edge]
        for x_edge in x_edges:
            crossed_edges = crossings[x_edge]
            crossed_edges.remove(edge)
            if len(crossed_edges) > 0:
                crossings[x_edge] = crossed_edges
            else:
                del crossings[x_edge]
    end = time.time()

    total_time += (end - start)
    if save_model:
        save_graph_model((graph, total_time), save_path)
    return graph, total_time
