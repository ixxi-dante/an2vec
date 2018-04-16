import numpy as np
import scipy.sparse
import networkx as nx

from nw2vec import layers
from nw2vec import dag
from nw2vec import utils


def _layer_csr_adj(out_nodes, adj, neighbour_samples):
    # TODO:
    # - take adj as a csr_matrix

    # out_nodes should be provided as a set
    assert isinstance(out_nodes, set)
    out_nodes = np.array(sorted(out_nodes))

    # Get the neighbours of the out nodes
    row_ind, col_ind = np.where(adj[out_nodes, :] > 0)  # TODO: memoize
    row_ind = out_nodes[row_ind]  # both `row_ind` and `col_ind` must index into `adj`
    if neighbour_samples is not None:
        sampled_row_ind = []
        sampled_col_ind = []
        for out_node in out_nodes:
            neighbours = col_ind[row_ind == out_node]  # TODO: memoize
            if len(neighbours) > 0:
                # If there are no neighours, nothing to sample from, so we're done for this node
                sample_size = np.min([len(neighbours), neighbour_samples])
                sampled_row_ind.extend([out_node] * sample_size)
                sampled_col_ind.extend(np.random.choice(neighbours, size=sample_size,
                                                        replace=False))

        row_ind, col_ind = sampled_row_ind, sampled_col_ind

    return (row_ind, col_ind)


def mask_indices(indices, size):
    assert isinstance(indices, set)
    indices = list(indices)
    mask = np.zeros(size)
    mask[indices] = 1
    return mask


def _compute_batch(model, adj, final_nodes, neighbour_samples):
    layers_crops = {}
    model_dag = dag.model_dag(model)
    gc_dag = dag.subdag(model_dag, lambda n: isinstance(n, layers.GC))

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
        # the layer (i.e. neigbours of `out_nodes`, and `out_nodes` themselves)
        csr_adj = _layer_csr_adj(out_nodes, adj, neighbour_samples=neighbour_samples)
        layers_crops[layer.name] = {'csr_adj': csr_adj,
                                    'in_nodes': set().union(out_nodes, csr_adj[1])}

    # Get the sorted list of nodes required by the whole network for this batch
    initial_nodes = set().union(*[crop['in_nodes'] for crop in layers_crops.values()])
    initial_nodes = np.array(sorted(initial_nodes))
    # Reduce the global adjacency matrix to only those nodes
    subadj = adj[initial_nodes, :][:, initial_nodes]
    # Pre-compute conversion of node ids in `adj` to ids in `subadj`
    global_to_sub = {node: i for i, node in enumerate(initial_nodes)}

    # Create the subadjs to be fed to each layer
    feeds = {}
    for name, crop in layers_crops.items():
        out_nodes_subadj = [global_to_sub[out_node] for out_node in crop['csr_adj'][0]]
        in_nodes_subadj = [global_to_sub[in_node] for in_node in crop['csr_adj'][1]]
        layer_subadj_mask = np.array(scipy.sparse.csr_matrix((np.ones(len(out_nodes_subadj)),
                                                              (out_nodes_subadj, in_nodes_subadj)),
                                                             shape=subadj.shape).todense())
        feeds[name + '_adj'] = subadj * layer_subadj_mask
        feeds[name + '_output_mask'] = mask_indices(set(out_nodes_subadj), subadj.shape[0])

    return initial_nodes, feeds


def batches(model, features, adj, final_batch_size, neighbour_samples=None):
    # Check the adjacency matrix is:
    # - an ndarray
    # - undirected
    # - unweighted
    # - with no diagonal elements
    assert isinstance(adj, np.ndarray)
    n_nodes = adj.shape[0]
    assert len(features) == n_nodes
    assert (adj.T == adj).all()
    assert ((adj == 0) | (adj == 1)).all()
    assert np.trace(adj) == 0

    # Shuffle the node indices, to group-iterate through them
    order = np.arange(n_nodes)
    np.random.shuffle(order)

    for final_nodes in utils.grouper(order, final_batch_size):
        # Prepare `final_nodes`
        final_nodes = set(final_nodes)
        try:
            # Remove `None`s coming from `utils.grouper()` not having
            # enough elements to finish the last group
            final_nodes.remove(None)
        except KeyError:
            # No `None`s in `final_nodes`
            pass

        initial_nodes, feeds = _compute_batch(model, adj, final_nodes,
                                              neighbour_samples=neighbour_samples)
        yield features[initial_nodes], feeds
