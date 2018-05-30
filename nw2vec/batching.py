import random

import numpy as np
from scipy import sparse
import joblib
import networkx as nx

from nw2vec import dag


def _layer_csr_adj(out_nodes, adj, neighbour_samples):
    assert isinstance(adj, sparse.csr_matrix)

    # out_nodes should be provided as a set
    assert isinstance(out_nodes, set)
    out_nodes = np.array(sorted(out_nodes))
    assert out_nodes[0] >= 0

    if neighbour_samples is None:
        # Get all the neighbours of the out nodes
        row_ind, col_ind = adj[out_nodes, :].nonzero()
        row_ind = out_nodes[row_ind]  # both `row_ind` and `col_ind` must index into `adj`
        return (row_ind, col_ind)

    # Sample each node's row
    sampled_row_ind = []
    sampled_col_ind = []
    for out_node in out_nodes:
        neighbours = adj[out_node, :].nonzero()[1]
        if len(neighbours) == 0:
            # If there are no neighours, nothing to sample from, so we're done for this node
            continue

        sample_size = np.min([len(neighbours), neighbour_samples])
        sampled_row_ind.extend([out_node] * sample_size)
        sampled_col_ind.extend(np.random.choice(neighbours, size=sample_size, replace=False))

    return (sampled_row_ind, sampled_col_ind)


def mask_indices(indices, size):
    assert isinstance(indices, set)
    indices = list(indices)
    mask = np.zeros(size)
    mask[indices] = 1
    return mask


def _collect_layers_crops(model, adj, final_nodes, neighbour_samples):
    assert isinstance(adj, sparse.csr_matrix)

    layers_crops = {}
    model_dag = dag.model_dag(model)
    gc_dag = dag.subdag_GC(model_dag)

    for layer in reversed(list(nx.topological_sort(gc_dag))):
        # Get the nodes required by our children layers, which are the ones we will output for
        children = list(gc_dag.successors(layer))
        if len(children) == 0:
            out_nodes = final_nodes
        else:
            # `reversed(list(nx.topological_sort()))` ensures that all children
            # have been seen before this point, and are in `layers_crops`.
            out_nodes = set().union(*[layers_crops[child.name]['in_nodes']
                                      for child in children])

        # Get the corresponding csr_adj for this layer, and record the necessary nodes for
        # the layer (i.e. neighbours of `out_nodes`, and `out_nodes` themselves)
        csr_adj = _layer_csr_adj(out_nodes, adj, neighbour_samples)
        layers_crops[layer.name] = {'csr_adj': csr_adj,
                                    'out_nodes': out_nodes,
                                    'in_nodes': set().union(out_nodes, csr_adj[1])}

    return layers_crops


def _compute_batch(model, adj, final_nodes, neighbour_samples):
    assert isinstance(adj, sparse.csr_matrix)

    # Collect the crop dicts for each layer
    layers_crops = _collect_layers_crops(model, adj, final_nodes, neighbour_samples)

    # Get the sorted list of nodes required by the whole network for this batch
    required_nodes = set().union(*[crop['in_nodes'] for crop in layers_crops.values()])
    required_nodes = np.array(sorted(required_nodes))
    # Reduce the global adjacency matrix to only those nodes
    reqadj = adj[required_nodes, :][:, required_nodes]
    # Pre-compute conversion of node ids in `adj` to ids in `reqadj`
    global_to_req = {node: i for i, node in enumerate(required_nodes)}

    # Create the reqadjs to be fed to each layer
    feeds = {}
    for name, crop in layers_crops.items():
        row_ind_reqadj = [global_to_req[row] for row in crop['csr_adj'][0]]
        col_ind_reqadj = [global_to_req[col] for col in crop['csr_adj'][1]]
        layer_reqadj_mask = sparse.csr_matrix((np.ones(len(row_ind_reqadj)),
                                               (row_ind_reqadj, col_ind_reqadj)),
                                              shape=reqadj.shape)
        feeds[name + '_adj'] = reqadj.multiply(layer_reqadj_mask)

        out_nodes_reqadj = set([global_to_req[out_node] for out_node in crop['out_nodes']])
        feeds[name + '_output_mask'] = mask_indices(out_nodes_reqadj, reqadj.shape[0])

    return required_nodes, feeds


def _collect_maxed_connected_component(g, node, max_size, exclude_nodes, collected):
    assert node not in collected

    collected.add(node)
    if len(collected.difference(exclude_nodes)) > max_size:
        # The nodes collected in the component minus `exclude_nodes` are more than the limit, tell
        # caller we maxed out so that it does not continue collecting.
        return True

    for neighbour in g.neighbors(node):
        if neighbour in collected:
            continue

        if _collect_maxed_connected_component(g, neighbour, max_size, exclude_nodes, collected):
            return True

    return False


def filtered_connected_component_or_none(g, node, max_size, exclude_nodes):
    # Return the set of nodes making up the connected component containing `node` and excluding
    # `exclude_nodes`, up to `max_size` (included), or `None` if the component minus the excluded
    # nodes is bigger than `max_size`.
    collected = set()
    maxed = _collect_maxed_connected_component(g, node, max_size, exclude_nodes, collected)
    return None if maxed else collected.difference(exclude_nodes)


def distinct_random_walk(g, max_walk_length, exclude_nodes):
    seed = random.sample(set(g.nodes).difference(exclude_nodes), 1)[0]
    seed_filtered_component = filtered_connected_component_or_none(g, seed, max_walk_length,
                                                                   exclude_nodes)
    if seed_filtered_component is not None:
        # This connected component minus the already-collected nodes has `<= max_walk_length`
        # unique nodes, so the walk would span it all.
        return seed_filtered_component

    current_node = seed
    walk_nodes = set([current_node])
    # We know this walk will stop eventually because this component minus `exclude_nodes` is
    # larger than `max_walk_length`.
    while len(walk_nodes) < max_walk_length:
        current_node = random.sample(list(g.neighbors(current_node)), 1)[0]
        if current_node not in exclude_nodes.union(walk_nodes):
            walk_nodes.add(current_node)

    assert len(walk_nodes) == max_walk_length
    return walk_nodes


def jumpy_distinct_random_walk(g, max_size, max_walk_length):
    if len(g.nodes) <= max_size:
        # Our jumpy random walk would span all of `g`. Take a shortcut.
        return set(g.nodes)

    collected = set()
    while len(collected) < max_size:
        current_max_walk_length = min(max_walk_length, max_size - len(collected))
        walk_nodes = distinct_random_walk(g, current_max_walk_length, collected)
        collected.update(walk_nodes)

    assert len(collected) == max_size
    return collected


def _get_nx_from_ndarray_or_spmatrix():
    cache = {}

    def _nx_from_ndarray_or_spmatrix(adj):
        key = joblib.hash(adj)
        if key not in cache:
            # Check the adjacency matrix is:
            # ... an ndarray...
            if not isinstance(adj, (np.ndarray, sparse.csr_matrix)):
                raise ValueError('Unsupported adjacency matrix type: {}'.format(type(adj)))
            # ... undirected...
            assert (adj.T != adj).sum() == 0
            if isinstance(adj, np.ndarray):
                # ... unweighted...
                assert ((adj == 0) | (adj == 1)).all()
                # ... with no diagonal elements.
                assert np.trace(adj) == 0
                # ... and create the graph.
                cache[key] = nx.from_numpy_array(adj)
            else:
                # ... unweighted...
                assert ((adj.data == 0) | (adj.data == 1)).all()
                # ... with no diagonal elements.
                csr_eye = sparse.csr_matrix((np.ones(adj.shape[0]),
                                             (np.arange(adj.shape[0]), np.arange(adj.shape[0]))),
                                            shape=adj.shape)
                assert adj.multiply(csr_eye).sum() == 0
                # ... and create the graph.
                cache[key] = nx.from_scipy_sparse_matrix(adj)
        return cache[key]

    return _nx_from_ndarray_or_spmatrix


nx_from_ndarray_or_spmatrix = _get_nx_from_ndarray_or_spmatrix()


def jumpy_walks(adj, batch_size, max_walk_length):
    # FIXME: see if we can't use scipy.sparse.csgraph methods and forego using networkx altogether
    g = nx_from_ndarray_or_spmatrix(adj)
    nodes = set(g.nodes)
    while len(nodes) > 0:
        g = g.subgraph(nodes)
        sample = jumpy_distinct_random_walk(g, batch_size, max_walk_length)
        yield sample
        nodes.difference_update(sample)


def epoch_batches(model, adj, batch_size, max_walk_length, neighbour_samples=None):
    for final_nodes in jumpy_walks(adj, batch_size, max_walk_length):
        required_nodes, feeds = _compute_batch(model, adj, final_nodes,
                                               neighbour_samples=neighbour_samples)
        yield required_nodes, np.array(sorted(final_nodes)), feeds
