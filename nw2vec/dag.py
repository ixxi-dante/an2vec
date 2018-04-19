import networkx as nx


def _collect_node_filtered_children_dict(dag, node, func, collected):
    assert isinstance(collected, dict)

    if node in collected:
        raise ValueError("Was already called on node '{}'".format(node))

    children = set()
    for child in dag.successors(node):
        if child not in collected:
            _collect_node_filtered_children_dict(dag, child, func, collected)
        if func(child):
            children.add(child)
        else:
            children.update(collected[child])

    collected[node] = children


def _collect_node_descendants_dict(dag, node, collected):
    assert isinstance(collected, dict)

    if node in collected:
        raise ValueError("Was already called on node '{}'".format(node))

    descendants = set()
    for child in dag.successors(node):
        if child not in collected:
            _collect_node_descendants_dict(dag, child, collected)
        descendants.add(child)
        descendants.update(collected[child])

    collected[node] = descendants


def subdag(dag, func):
    assert nx.is_directed_acyclic_graph(dag)

    roots = [n for n in dag.nodes if len(list(dag.predecessors(n))) == 0]
    filtered_children = {}
    for root in roots:
        _collect_node_filtered_children_dict(dag, root, func, filtered_children)

    subdag_dict = dict((n, children) for n, children in filtered_children.items() if func(n))
    return nx.from_dict_of_lists(subdag_dict, create_using=nx.DiGraph())


def _collect_layer_dag_dict(layer, collected):
    assert isinstance(collected, dict)

    if layer in collected:
        raise ValueError("Was already called on layer '{}'".format(layer))

    children = [outnode.outbound_layer for outnode in layer._outbound_nodes]
    for child in children:
        if child not in collected:
            _collect_layer_dag_dict(child, collected)

    collected[layer] = children


def restrict(dag, roots):
    assert nx.is_directed_acyclic_graph(dag)

    descendants = {}
    for root in roots:
        if root not in descendants:
            _collect_node_descendants_dict(dag, root, descendants)

    roots_descendants = set().union(roots, *[descendants[root] for root in roots])
    return nx.from_dict_of_lists(nx.to_dict_of_lists(dag, nodelist=roots_descendants),
                                 create_using=nx.DiGraph())


def model_dag(model):
    dag_dict = {}
    for input_layer in model.input_layers:
        _collect_layer_dag_dict(input_layer, dag_dict)

    # Restrict the dag to what is reachable from the model inputs and outputs
    dag = nx.from_dict_of_lists(dag_dict, create_using=nx.DiGraph())
    dag = restrict(dag, model.input_layers)
    dag = nx.reverse(dag)
    dag = restrict(dag, model.output_layers)
    dag = nx.reverse(dag)

    # Check that each of the model's output layers is in the dag collected,
    # and that each dag leaf is in the model's output layers (but there can be
    # output layers that are not leaves).
    model_outputs = set(model.output_layers)
    for output in model_outputs:
        assert output in dag.nodes
    leaves = set([n for n in dag.nodes if len(list(dag.successors(n))) == 0])
    for leaf in leaves:
        assert leaf in model_outputs

    return dag
