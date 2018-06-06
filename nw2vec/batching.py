import random

import numpy as np
from scipy import sparse
import networkx as nx

from nw2vec import dag
from nw2vec import graph
from nw2vec import utils


# TODO: numbise what can be


def _layer_csr_adj(out_nodes, adj, neighbour_samples):
    assert isinstance(adj, sparse.csr_matrix)

    # out_nodes should be provided as a set
    assert isinstance(out_nodes, set)
    out_nodes = np.array(sorted(out_nodes))
    assert out_nodes[0] >= 0

    if neighbour_samples is None:
        # Explore and add each node's row
        row_ind = []
        col_ind = []
        for out_node in out_nodes:
            neighbours = adj.indices[adj.indptr[out_node]:adj.indptr[out_node + 1]]
            n_neighbours = len(neighbours)
            if n_neighbours == 0:
                # If there are no neighours, nothing to sample from, so we're done for this node
                continue
            row_ind.extend([out_node] * n_neighbours)
            col_ind.extend(neighbours)

    else:
        # `neighbour_samples` is not None, we can preallocate then cut back what we didn't fill
        max_samples = len(out_nodes) * neighbour_samples
        row_ind = np.zeros(max_samples, dtype=int)
        col_ind = np.zeros(max_samples, dtype=int)
        n_collected = 0

        for out_node in out_nodes:
            neighbours = adj.indices[adj.indptr[out_node]:adj.indptr[out_node + 1]]
            n_neighbours = len(neighbours)
            if n_neighbours == 0:
                # If there are no neighours, nothing to sample from, so we're done for this node
                continue

            sample_size = min(n_neighbours, neighbour_samples)
            row_ind[n_collected:n_collected + sample_size] = out_node
            # This is faster than using `np.random.choice(replace=False)`
            np.random.shuffle(neighbours)
            col_ind[n_collected:n_collected + sample_size] = neighbours[:sample_size]
            n_collected += sample_size

        # Chop off what we didn't use
        row_ind = row_ind[:n_collected]
        col_ind = col_ind[:n_collected]

    return (row_ind, col_ind)


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
    global_to_req = np.zeros(adj.shape[0], dtype=int)
    global_to_req[required_nodes] = np.arange(len(required_nodes))

    # Create the reqadjs to be fed to each layer
    feeds = {}
    for name, crop in layers_crops.items():
        row_ind_reqadj = global_to_req[crop['csr_adj'][0]]
        col_ind_reqadj = global_to_req[crop['csr_adj'][1]]
        layer_reqadj_mask = sparse.csr_matrix((np.ones(len(row_ind_reqadj)),
                                               (row_ind_reqadj, col_ind_reqadj)),
                                              shape=reqadj.shape)
        tensor_indices, tensor_values, tensor_dense_shape = \
            utils.csr_to_sparse_tensor_parts(reqadj.multiply(layer_reqadj_mask))
        feeds[name + '_adj/indices'] = tensor_indices
        feeds[name + '_adj/values'] = tensor_values
        feeds[name + '_adj/dense_shape'] = tensor_dense_shape

        out_nodes_reqadj = set([global_to_req[out_node] for out_node in crop['out_nodes']])
        feeds[name + '_output_mask'] = mask_indices(out_nodes_reqadj, reqadj.shape[0])

    return required_nodes, feeds


def _collect_maxed_connected_component(g, node, max_size, collected):
    assert node not in collected

    collected.add(node)
    if len(collected) > max_size:
        # The nodes collected in the component are more than the limit, tell
        # caller we maxed out so that it does not continue collecting.
        return True

    for neighbour in g.neighbors(node):
        if neighbour in collected:
            continue

        if _collect_maxed_connected_component(g, neighbour, max_size, collected):
            return True

    return False


def connected_component_or_none(g, node, max_size):
    # Return the set of nodes making up the connected component containing `node`,
    # up to `max_size` (included), or `None` if the component is bigger than `max_size`.
    collected = set()
    maxed = _collect_maxed_connected_component(g, node, max_size, collected)
    return None if maxed else collected


def distinct_random_walk(g, seed, max_walk_length):
    seed_component = connected_component_or_none(g, seed, max_walk_length)
    if seed_component is not None:
        # This connected component has `<= max_walk_length` unique nodes,
        # so the walk would span it all.
        return seed_component

    current_node = seed
    last_node = None
    walk_nodes = set([current_node])
    # We know this walk will stop eventually because this component is
    # larger than `max_walk_length`.
    while len(walk_nodes) < max_walk_length:
        prev_last_node = last_node
        last_node = current_node
        if prev_last_node is None:
            current_node = random.sample(list(g.neighbors(last_node)), 1)[0]
        else:
            current_node = g.draw_after_edge(prev_last_node, last_node)

        # No-op if current_node was already seen. We just move on to it and sample
        # from its neighbours in the next iteration.
        walk_nodes.add(current_node)

    return walk_nodes


def batch_walks(g, seeds_per_batch, max_walk_length):
    seeds = g.nodes.copy()
    np.random.shuffle(seeds)

    # TODO: parallelise
    for batch_seeds in utils.grouper(seeds, seeds_per_batch):
        yield set().union(*[distinct_random_walk(g, seed, max_walk_length)
                            for seed in batch_seeds])


def epoch_batches(model, g,
                  seeds_per_batch, max_walk_length,
                  neighbour_samples):
    # TODO: parallelise
    for final_nodes in batch_walks(g, seeds_per_batch, max_walk_length):
        required_nodes, feeds = _compute_batch(model, g.adj, final_nodes,
                                               neighbour_samples=neighbour_samples)
        yield required_nodes, np.array(sorted(final_nodes)), feeds


def batches(model, adj, features, target_func,
            seeds_per_batch, max_walk_length,
            p=1.0, q=1.0, neighbour_samples=None):
    # Prepare graph structure
    if not isinstance(adj, sparse.csr_matrix):
        assert isinstance(adj, np.ndarray)
        adj = sparse.csr_matrix(adj)
    g = graph.get_csgraph(adj, p, q)

    # Prepare features
    assert adj.shape[0] == adj.shape[1]
    assert features.shape[0] == adj.shape[0]
    features = utils.scale_center(features)

    while True:
        for required_nodes, final_nodes, feeds in epoch_batches(
             model, g, seeds_per_batch, max_walk_length, neighbour_samples):
            batch_adj = adj[final_nodes, :][:, final_nodes]
            yield (features[required_nodes],
                   target_func(batch_adj, required_nodes, final_nodes),
                   feeds)
