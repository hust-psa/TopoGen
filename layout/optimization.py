import os
import shutil
import math
import time
import networkx as nx
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from utils.interface import save_model as save_graph_model
from utils.crossing import find_crossings
from utils.graph import add_dummy_node
from utils.layout import incident_edges_by_order, radian_between_vectors, node_direction
from utils.plotting import plot_graph

BIGM = 10 ** 6


def preprocessing(graph):
    """
    Preprocess a graph for layout planning
    :param graph: networkx.Graph
    :return: networkx.Graph, the preprocessed graph
    """
    # add a dummy node at each crossing
    n_dummy = 0
    crossings = find_crossings(graph)
    while len(crossings) > 0:
        x_edges, x_pt = crossings[0]
        graph, _ = add_dummy_node(graph, x_pt, x_edges, label=n_dummy)
        n_dummy += 1
        crossings = find_crossings(graph)
    return graph


def layout_optimization(graph, max_retry=10, save_intermediate=True, save_dir='files', **kwargs):
    """
    Optimize the layout of the graph
    :param graph: networkx.Graph
    :param max_retry: int, max rounds of optimization allowed
    :param save_intermediate: boolean, whether to save intermediate processes
    :param save_dir: str, the path to save the graph model
    :return:
        networkx.Graph, layout optimized graph embedding
        float, time consumption in seconds
    """
    params = kwargs.copy()

    start = time.time()
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    graph = preprocessing(graph)
    depated_edge_pairs = []
    opt_graph, updated_k, updated_seed = milp_layout(graph, **kwargs)
    end = time.time()
    total_time = end - start
    plot_graph(opt_graph, save_figure=save_intermediate, save_path=os.path.join(save_dir, 'round_0.pdf'))
    if save_intermediate:
        params['k'] = updated_k
        params['seed'] = updated_seed
        save_graph_model((opt_graph, params, total_time), os.path.join(save_dir, 'round_0.mdl'))

    n_retry = 0
    while n_retry < max_retry:
        start = time.time()
        kwargs['k'] = updated_k
        kwargs['seed'] = np.random.randint(0, 1000)
        # find crossed edge pairs in this round of optimization
        crossed_edge_pairs = [ep for ep, _ in find_crossings(opt_graph)]
        if len(crossed_edge_pairs) == 0:
            end = time.time()
            total_time += (end - start)
            break
        depated_edge_pairs += crossed_edge_pairs
        hint_node_pos = dict()
        for node, attr in opt_graph.nodes(data=True):
            hint_node_pos[node] = (attr['x'], attr['y'])
        # conduct the next round of optimization
        opt_graph, updated_k, updated_seed = milp_layout(graph, depated_edge_pairs, hint_node_pos, **kwargs)
        end = time.time()
        total_time += (end - start)
        n_retry += 1
        plot_graph(opt_graph, save_figure=save_intermediate, save_path=os.path.join(save_dir, 'round_%d.pdf' % n_retry))
        if save_intermediate:
            params['k'] = updated_k
            params['seed'] = updated_seed
            save_graph_model((opt_graph, params, total_time), os.path.join(save_dir, 'round_%d.mdl' % n_retry))
    return opt_graph, params, total_time


def milp_layout(graph, departed_edge_pairs=None, hint_node_pos=None, **kwargs):
    """
    Build the MILP problem and conduct a round of layout optimization
    :param graph: networkx.Graph
    :param departed_edge_pairs: list of edge pairs not allowed to cross
    :param hint_node_pos: dict, suggested coordinates of nodes
    :return: networkx.Graph, layout optimized graph embedding
    """
    k = kwargs.get('k', None)
    min_e_len = kwargs.get('min_e_len', 1)
    min_n_dis = kwargs.get('min_n_dis', 1)
    n_flex_sec = kwargs.get('n_flex_sec', -1)
    w_relative_pos = kwargs.get('w_relative_pos', 5)
    w_compact = kwargs.get('w_compact', 2)
    w_hor_ver = kwargs.get('w_hor_ver', 3)
    w_evenness = kwargs.get('w_evenness', 2)
    seed = kwargs.get('seed', np.random.randint(0, 1000))
    if departed_edge_pairs is None:
        departed_edge_pairs = []
    if hint_node_pos is None:
        hint_node_pos = dict()
    # initialize the MIPS optimization problem
    model = gp.Model('MILP')

    # scan all edges and determine the length constraint for each edge
    min_edge_lengths = dict()
    min_raw_edge_length = np.inf
    avr_raw_edge_length = 0
    for u, v in graph.edges:
        u_pt = graph.nodes[u]['x'], graph.nodes[u]['y']
        v_pt = graph.nodes[v]['x'], graph.nodes[v]['y']
        edge_length = math.dist(u_pt, v_pt)
        avr_raw_edge_length += edge_length
        min_edge_lengths[(u, v)] = edge_length
        if edge_length < min_raw_edge_length:
            min_raw_edge_length = edge_length
    avr_raw_edge_length /= len(graph.edges)
    for edge, e_len in min_edge_lengths.items():
        if e_len > 0.5 * avr_raw_edge_length:
            min_edge_lengths[edge] = math.floor(e_len / min_raw_edge_length) * min_e_len
        else:
            min_edge_lengths[edge] = math.floor(0.5 * avr_raw_edge_length / min_raw_edge_length) * min_e_len
    # for u, v in graph.edges:
    #     min_edge_lengths[(u, v)] = min_e_len

    # hard constraint #1: coordinate system
    # find a proper k for building the k-aligned coordinate system
    if k is None:
        max_degree = max(dict(graph.degree).values())
        criterion_1 = math.ceil(max_degree / 2)
        min_avr_angle, min_avr_angle_node_degree = 2 * math.pi, 0
        for node in graph.nodes:
            incident_edges = incident_edges_by_order(graph, node)
            if len(incident_edges) > 1:
                edge_i, edge_j = incident_edges[0], incident_edges[-1]
                (u_i_x, u_i_y), (v_i_x, v_i_y) = ((graph.nodes[n]['x'], graph.nodes[n]['y']) for n in edge_i)
                (u_j_x, u_j_y), (v_j_x, v_j_y) = ((graph.nodes[n]['x'], graph.nodes[n]['y']) for n in edge_j)
                vector_i = (v_i_x - u_i_x, v_i_y - u_i_y)
                vector_j = (v_j_x - u_j_x, v_j_y - u_j_y)
                angle = radian_between_vectors(vector_i, vector_j)
                avr_angle = angle / (len(incident_edges) - 1)
                if avr_angle < min_avr_angle:
                    min_avr_angle = avr_angle
                    min_avr_angle_node_degree = math.floor(math.pi / 2 / avr_angle)
        criterion_2 = min_avr_angle_node_degree * 2
        k = criterion_2 if criterion_2 > criterion_1 else criterion_1
        if k > 2 * criterion_1:
            k = 2 * criterion_1
    thetas = [i * math.pi / k for i in range(k)]

    coordinates = dict()
    # canvas bounds are determined according to the max value of min edge lengths
    cv_bnd = max(min_edge_lengths.values()) * len(graph.nodes)
    for node in graph.nodes:
        coordinates_per_node = dict()
        x = model.addVar(lb=-cv_bnd, ub=cv_bnd, vtype=GRB.CONTINUOUS, name='n%s_x' % node)
        y = model.addVar(lb=-cv_bnd, ub=cv_bnd, vtype=GRB.CONTINUOUS, name='n%s_y' % node)
        if node in hint_node_pos:
            x.varhintval, y.varhintval = hint_node_pos[node]
        coordinates_per_node['x'] = x
        coordinates_per_node['y'] = y
        for i in range(k):
            z_i = model.addVar(lb=-2 * cv_bnd, ub=2 * cv_bnd, vtype=GRB.CONTINUOUS, name='n%s_z%d' % (node, i))
            z_o_i = model.addVar(lb=-2 * cv_bnd, ub=2 * cv_bnd, vtype=GRB.CONTINUOUS, name='n%s_zo%d' % (node, i))
            coordinates_per_node['z%d' % i] = z_i
            coordinates_per_node['zo%d' % i] = z_o_i
            if abs(math.sin(thetas[i])) >= 1 / BIGM and abs(math.cos(thetas[i])) >= 1 / BIGM:
                model.addConstr(z_i == math.cos(thetas[i]) * x + math.sin(thetas[i]) * y)
                model.addConstr(z_o_i == -1 * math.sin(thetas[i]) * x + math.cos(thetas[i]) * y)
            else:
                if abs(math.sin(thetas[i])) < 1 / BIGM:
                    model.addConstr(z_i == math.cos(thetas[i]) * x)
                    model.addConstr(z_o_i == math.cos(thetas[i]) * y)
                else:
                    model.addConstr(z_i == math.sin(thetas[i]) * y)
                    model.addConstr(z_o_i == -1 * math.sin(thetas[i]) * x)
        coordinates[node] = coordinates_per_node
    model.update()

    # hard constraint #2: edge directions and minimum length
    directions = dict()
    for u, v in graph.edges:
        min_edge_length = min_edge_lengths.get((u, v), 1)
        if n_flex_sec < 0:
            n_flex = max(math.ceil((graph.degree(u) - 1) / 2), 1)
        else:
            n_flex = n_flex_sec
        u_pos, v_pos = ((graph.nodes[n]['x'], graph.nodes[n]['y']) for n in (u, v))
        origin_sec = node_direction(u_pos, v_pos, k)
        possible_secs = [i % (2 * k) for i in range(origin_sec - n_flex, origin_sec + n_flex + 1)]
        alphas, sec_alphas, oppo_sec_alphas = [], [], []
        for sec in possible_secs:
            alpha = model.addVar(vtype=GRB.BINARY, name='alpha%d_%s-%s' % (sec, u, v))
            z_o_i_prime_u = coordinates[u]['zo%d' % (sec % k)]
            z_o_i_prime_v = coordinates[v]['zo%d' % (sec % k)]
            z_i_prime_u = coordinates[u]['z%d' % (sec % k)]
            z_i_prime_v = coordinates[v]['z%d' % (sec % k)]
            model.addConstr((alpha == 1) >> (z_o_i_prime_u == z_o_i_prime_v))
            if sec < k:
                model.addConstr((alpha == 1) >> (z_i_prime_v - z_i_prime_u >= min_edge_length))
            else:
                model.addConstr((alpha == 1) >> (z_i_prime_u - z_i_prime_v >= min_edge_length))
            alphas.append(alpha)
            sec_alphas.append(sec * alpha)
            oppo_sec_alphas.append(((sec + k) % (2 * k)) * alpha)
        model.addConstr(gp.quicksum(alphas) == 1)
        direction = model.addVar(lb=0, ub=2 * k - 1, vtype=GRB.INTEGER, name='dir_%s-%s' % (u, v))
        model.addConstr(direction == gp.quicksum(sec_alphas))
        directions[(u, v)] = direction
        oppo_direction = model.addVar(lb=0, ub=2 * k - 1, vtype=GRB.INTEGER, name='dir_%s-%s' % (v, u))
        model.addConstr(oppo_direction == gp.quicksum(oppo_sec_alphas))
        directions[(v, u)] = oppo_direction
    model.update()

    # hard constraint #3: combinatorial embedding
    for node in graph.nodes:
        betas = []
        incident_edges = incident_edges_by_order(graph, node)
        for i in range(len(incident_edges)):
            this_edge = incident_edges[i]
            next_edge = incident_edges[(i + 1) % len(incident_edges)]
            this_direction = directions[this_edge]
            next_direction = directions[next_edge]
            beta = model.addVar(vtype=GRB.BINARY, name='beta_%s-%s' % (this_edge[0], this_edge[1]))
            model.addConstr(this_direction + 1 <= next_direction + 2 * k * beta)
            betas.append(beta)
        model.addConstr(gp.quicksum(betas) == 1)
    model.update()

    # hard constraint #4: planarity
    for e, e_prime in departed_edge_pairs:
        (u, v), (u_prime, v_prime) = e, e_prime
        gammas = []
        for i in range(2 * k):
            parted = model.addVars(4, vtype=GRB.BINARY)
            z_i_prime_u = coordinates[u]['z%d' % (i % k)]
            z_i_prime_v = coordinates[v]['z%d' % (i % k)]
            z_i_prime_u_prime = coordinates[u_prime]['z%d' % (i % k)]
            z_i_prime_v_prime = coordinates[v_prime]['z%d' % (i % k)]
            if i < k:
                model.addConstr((parted[0] == 1) >> (z_i_prime_u_prime - z_i_prime_u >= min_n_dis))
                model.addConstr((parted[1] == 1) >> (z_i_prime_u_prime - z_i_prime_v >= min_n_dis))
                model.addConstr((parted[2] == 1) >> (z_i_prime_v_prime - z_i_prime_u >= min_n_dis))
                model.addConstr((parted[3] == 1) >> (z_i_prime_v_prime - z_i_prime_v >= min_n_dis))
            else:
                model.addConstr((parted[0] == 1) >> (z_i_prime_u - z_i_prime_u_prime >= min_n_dis))
                model.addConstr((parted[1] == 1) >> (z_i_prime_v - z_i_prime_u_prime >= min_n_dis))
                model.addConstr((parted[2] == 1) >> (z_i_prime_u - z_i_prime_v_prime >= min_n_dis))
                model.addConstr((parted[3] == 1) >> (z_i_prime_v - z_i_prime_v_prime >= min_n_dis))
            gamma_i = model.addVar(vtype=GRB.BINARY, name='gamma_%d_%s-%s_%s-%s' % (i, u, v, u_prime, v_prime))
            model.addConstr((gamma_i == 1) >> (gp.quicksum(parted) == 4))
            gammas.append(gamma_i)
        model.addConstr(gp.quicksum(gammas) >= 1)
    model.update()

    # soft constraint #1: relative position
    if w_relative_pos > 0:
        ksis = []
        for edge in graph.edges:
            u_pos, v_pos = ((graph.nodes[n]['x'], graph.nodes[n]['y']) for n in edge)
            original_direction = node_direction(u_pos, v_pos, k)
            direction = directions[edge]
            ksi = model.addVar(vtype=GRB.INTEGER, name='ksi_%s-%s' % (edge[0], edge[1]))
            ksis.append(ksi)
            model.addConstr(direction - original_direction <= ksi)
            model.addConstr(original_direction - direction <= ksi)
        cost_relative_pos = gp.quicksum(ksis)
        model.update()
    else:
        cost_relative_pos = 0

    # soft constraint #2: compactness
    if w_compact > 0:
        lamdas = []
        for edge in graph.edges:
            u, v = edge
            lamda = model.addVar(vtype=GRB.INTEGER, name='lamda_%s-%s' % (u, v))
            lamdas.append(lamda)
            for i in range(k):
                z_i_u = coordinates[u]['z%d' % i]
                z_i_v = coordinates[v]['z%d' % i]
                model.addConstr(z_i_u - z_i_v <= lamda)
                model.addConstr(z_i_v - z_i_u <= lamda)
        cost_compact = gp.quicksum(lamdas)
        model.update()
    else:
        cost_compact = 0

    # soft constraint #3: horizontal and vertical edges
    if w_hor_ver > 0:
        sigmas = []
        for edge in graph.edges:
            hor = model.addVar(vtype=GRB.BINARY, name='hor_%s-%s' % (edge[0], edge[1]))
            model.addConstr((hor == 1) >> (directions[edge] == 0))
            ver = model.addVar(vtype=GRB.BINARY, name='ver_%s-%s' % (edge[0], edge[1]))
            model.addConstr((ver == 1) >> (directions[edge] == k // 2))
            sigmas.append(1 - hor - ver)
        cost_hor_ver = gp.quicksum(sigmas)
        model.update()
    else:
        cost_hor_ver = 0

    # soft constraint #4: edge length evenness
    if w_evenness > 0:
        lamda_diffs = []
        n_edges = len(graph.edges)
        if isinstance(cost_compact, int):
            lamdas = []
            for edge in graph.edges:
                u, v = edge
                lamda = model.addVar(vtype=GRB.INTEGER, name='lamda_%s-%s' % (u, v))
                lamdas.append(lamda)
                for i in range(k):
                    z_i_u = coordinates[u]['z%d' % i]
                    z_i_v = coordinates[v]['z%d' % i]
                    model.addConstr(z_i_u - z_i_v <= lamda)
                    model.addConstr(z_i_v - z_i_u <= lamda)
            _cost_compact = gp.quicksum(lamdas)
            model.update()
        else:
            _cost_compact = cost_compact
        for edge in graph.edges:
            u, v = edge
            lamda = model.getVarByName('lamda_%s-%s' % (u, v))
            l_diff = model.addVar(vtype=GRB.CONTINUOUS, name='lamda_diff_%s-%s' % (u, v))
            lamda_diffs.append(l_diff)
            model.addConstr(l_diff >= lamda - _cost_compact / n_edges)
            model.addConstr(l_diff >= -lamda + _cost_compact / n_edges)
        cost_evenness = gp.quicksum(lamda_diffs) / n_edges
        model.update()
    else:
        cost_evenness = 0

    # set objective function and run optimization
    weights = np.array([w_relative_pos, w_compact, w_hor_ver, w_evenness])
    relative_weights = weights / weights.sum()
    w_relative_pos, w_compact, w_hor_ver, w_evenness = relative_weights
    objective = sum([w_relative_pos * cost_relative_pos,
                     w_compact * cost_compact,
                     w_hor_ver * cost_hor_ver,
                     w_evenness * cost_evenness])
    params = [('Seed', seed), ('Minimal edge length', min_e_len), ('K', k), ('Minimal node distance', min_n_dis),
              ('Flexibility of relative position', n_flex_sec), ('Weight for relative position', w_relative_pos),
              ('Weight for compactness', w_compact), ('Weight for orthogonality', w_hor_ver),
              ('Weight for edge length evenness', w_evenness), ('Edge pairs not allowed to cross', departed_edge_pairs)]
    for name, value in params:
        print('%s: %g' % (name, value) if not isinstance(value, list) else '%s: %s' % (name, str(value)))

    new_graph = graph.copy()
    model.setObjective(objective, GRB.MINIMIZE)
    model.Params.Seed = seed
    model.Params.TimeLimit = 100
    model.Params.MIPFocus = 1
    model.Params.MIPGap = 1.0
    model.optimize()
    model.Params.TimeLimit = 300
    model.Params.MIPFocus = 0
    model.Params.MIPGap = 0.5
    model.optimize()
    model.Params.TimeLimit = 300
    model.Params.MIPFocus = 2
    model.Params.MIPGap = 0.3
    model.optimize()

    updated_k, updated_seed = k, seed
    if model.Status == GRB.INFEASIBLE:
        print('Model is infeasible')
        model.computeIIS()
        model.write('iis_%d.ilp' % k)
        model.dispose()
        kwargs['k'] = k + 2
        kwargs['seed'] = np.random.randint(0, 1000)
        new_graph, updated_k, updated_seed = milp_layout(graph, departed_edge_pairs, hint_node_pos, **kwargs)
    else:
        if model.SolCount > 0:
            # retrieve optimized vertex positions
            pos_x, pos_y = dict(), dict()
            for node in new_graph.nodes:
                x = model.getVarByName('n%s_x' % node).x
                y = model.getVarByName('n%s_y' % node).x
                pos_x[node] = x
                pos_y[node] = y
            nx.set_node_attributes(new_graph, pos_x, 'x')
            nx.set_node_attributes(new_graph, pos_y, 'y')
            model.dispose()
        else:
            print('Optimization reaches the time limit without finding a feasible solution')
            model.dispose()
            kwargs['k'] = k + 2
            kwargs['seed'] = np.random.randint(0, 1000)
            new_graph, updated_k, updated_seed = milp_layout(graph, departed_edge_pairs, hint_node_pos, **kwargs)
    return new_graph, updated_k, updated_seed
