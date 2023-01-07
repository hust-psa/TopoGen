import os
from crossing import move_inter_node
from layout import layout_optimization
from utils.interface import load_model, save_model
from utils.plotting import plot_graph


def simple_diagram(graph, graph_name, max_layout_retry=10, save_diagram=True, save_dir='files', **kwargs):
    print('%s: %s' % (graph_name, str(graph)))
    plot_graph(graph, save_figure=save_diagram, save_path=os.path.join(save_dir, 'raw.pdf'))

    # crossing reduction
    cross_model_path = os.path.join(save_dir, 'cross.mdl')
    cross_figure_path = os.path.join(save_dir, 'cross.pdf')
    saved_model = load_model(cross_model_path)
    if saved_model is not None:
        cr_graph, cr_time = saved_model
    else:
        cr_graph, cr_time = move_inter_node(graph, save_model=True, save_path=cross_model_path, **kwargs)
    plot_graph(cr_graph, save_figure=save_diagram, save_path=cross_figure_path)
    print('Crossing reduction time: %g s' % cr_time)

    # layout optimization
    layout_model_path = os.path.join(save_dir, 'layout.mdl')
    layout_figure_path = os.path.join(save_dir, 'layout.pdf')
    layout_save_dir = os.path.join(save_dir, 'layout')
    saved_model = load_model(layout_model_path)
    if saved_model is not None:
        opt_graph, params, opt_time = saved_model
    else:
        opt_graph, params, opt_time = layout_optimization(cr_graph, max_layout_retry, save_intermediate=True,
                                                          save_dir=layout_save_dir, **kwargs)
        save_model((opt_graph, params, opt_time), layout_model_path)
    plot_graph(opt_graph, save_figure=save_diagram, save_path=layout_figure_path)
    print('Layout optimization time: %g s' % opt_time)
    print('Total time: %g s' % (cr_time + opt_time))
    pass
