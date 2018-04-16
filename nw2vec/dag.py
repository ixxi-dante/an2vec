import networkx as nx


def _collect_filtered_children_dict(g, node, func, collected):
    if node in collected:
        raise ValueError("Was already called on node '{}'".format(node))

    children = set()
    for child in g.successors(node):
        if child not in collected:
            _collect_filtered_children_dict(g, child, func, collected)
        if func(child):
            children.add(child)
        else:
            children.update(collected[child])

    collected[node] = children


def subdag(g, func):
    assert nx.is_directed_acyclic_graph(g)

    roots = [n for n in g.nodes if len(list(g.predecessors(n))) == 0]
    filtered_children = {}
    for root in roots:
        _collect_filtered_children_dict(g, root, func, filtered_children)

    subdag_dict = dict((n, children) for n, children in filtered_children.items() if func(n))
    return nx.from_dict_of_lists(subdag_dict, create_using=nx.DiGraph())


def _collect_layer_dag_dict(layer, collected):
    if layer in collected:
        raise ValueError("Was already called on layer '{}'".format(layer))

    children = [outnode.outbound_layer for outnode in layer._outbound_nodes]
    for child in children:
        if child not in collected:
            _collect_layer_dag_dict(child, collected)

    collected[layer] = children


def model_dag(model):
    dag_dict = {}
    for input_layer in model.input_layers:
        _collect_layer_dag_dict(input_layer, dag_dict)

    # Check that each of the model's output layers is in the dag collected,
    # and that each dag leaf is in the model's output layers (but there can be
    # output layers that are not leaves).
    model_outputs = set(model.output_layers)
    for output in model_outputs:
        assert output in dag_dict
    leaves = set([n for n, children in dag_dict.items() if len(children) == 0])
    for leaf in leaves:
        assert leaf in model_outputs

    return nx.from_dict_of_lists(dag_dict, create_using=nx.DiGraph())
