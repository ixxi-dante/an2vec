import pytest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
import networkx as nx
import keras

from nw2vec import ae
from nw2vec import batching


def test__layer_csr_adj():
    # Our test data
    adj = sparse.csr_matrix(np.array([[0, 1, 0, 0, 1],
                                      [1, 0, 1, 1, 0],
                                      [0, 0, 1, 0, 0],
                                      [1, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 1]]))

    # Works with good arguments and no sampling
    row_ind, col_ind = batching._layer_csr_adj(set([1, 2]), adj, None)
    assert_array_equal(row_ind, [1, 1, 1, 2])
    assert_array_equal(col_ind, [3, 2, 0, 2])

    # Works with good arguments and sampling
    row_ind, col_ind = batching._layer_csr_adj(set([1, 2]), adj, 2)
    # node `1` has 3 potential neighbours, always two sampled,
    # while node `2` has only one neighbour, always sampled
    assert_array_equal(row_ind, [1, 1, 2])
    assert len(col_ind) == 2 + 1
    assert 2 in col_ind
    assert set(col_ind).issubset([0, 2, 3])
    # Covers all possibilities when sampling
    col_inds = set()
    for _ in range(5):
        _, col_ind = batching._layer_csr_adj(set([1, 2]), adj, 2)
        col_inds.update(col_ind)
    assert col_inds == set([0, 2, 3])

    # Rejects out-of-bounds indices in `out_nodes`
    with pytest.raises(IndexError):
        batching._layer_csr_adj(set([5]), adj, None)

    # Rejects negative indices in `out_nodes`
    with pytest.raises(AssertionError):
        batching._layer_csr_adj(set([-1]), adj, None)

    # Takes only sets for `out_nodes`
    with pytest.raises(AssertionError):
        batching._layer_csr_adj([1], adj, None)


def test_mask_indices():
    # Works with good arguments
    assert_array_equal(batching.mask_indices(set([2, 4]), 6), [0, 0, 1, 0, 1, 0])

    # Works with negative indices
    assert_array_equal(batching.mask_indices(set([-1]), 5), [0, 0, 0, 0, 1])

    # Takes only sets for `indices`
    with pytest.raises(AssertionError):
        batching.mask_indices([1], 5)

    # Rejects out-of-bounds indices
    with pytest.raises(IndexError):
        batching.mask_indices(set([5]), 5)
    with pytest.raises(IndexError):
        batching.mask_indices(set([-6]), 5)


@pytest.fixture
def model_depth2():
    layer_input = keras.layers.Input(shape=(20,), name='layer_input')
    layer1a_placeholders, layer1a = ae.gc_layer_with_placeholders(
        10, 'layer1a', {'activation': 'relu'}, layer_input)
    layer1b_placeholders, layer1b = ae.gc_layer_with_placeholders(
        10, 'layer1b', {'activation': 'relu'}, layer_input)
    layer1ab = keras.layers.Concatenate(name='layer1ab')([layer1a, layer1b])
    layer2_placeholders, layer2 = ae.gc_layer_with_placeholders(
        5, 'layer2', {'gather_mask': True}, layer1ab)
    return ae.Model(inputs=([layer_input]
                            + layer1a_placeholders
                            + layer1b_placeholders
                            + layer2_placeholders),
                    outputs=layer2)


@pytest.fixture
def model_depth3():
    layer_input = keras.layers.Input(shape=(40,), name='layer_input')
    layer0_placeholders, layer0 = ae.gc_layer_with_placeholders(
        20, 'layer0', {'activation': 'relu'}, layer_input)
    layer1a_placeholders, layer1a = ae.gc_layer_with_placeholders(
        10, 'layer1a', {'activation': 'relu'}, layer0)
    layer1b_placeholders, layer1b = ae.gc_layer_with_placeholders(
        10, 'layer1b', {'activation': 'relu'}, layer0)
    layer1ab = keras.layers.Concatenate(name='layer1ab')([layer1a, layer1b])
    layer2_placeholders, layer2 = ae.gc_layer_with_placeholders(
        5, 'layer2', {'gather_mask': True}, layer1ab)
    return ae.Model(inputs=([layer_input]
                            + layer0_placeholders
                            + layer1a_placeholders
                            + layer1b_placeholders
                            + layer2_placeholders),
                    outputs=layer2)


def test__collect_layers_crops(model_depth3):
    # Our test data
    adj = sparse.csr_matrix(np.array([[0, 1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0],
                                      [1, 0, 1, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 0, 0]]))

    # Works well with good arguments and no sampling
    crops = batching._collect_layers_crops(model_depth3, adj, set([1, 2]), None)
    crop0 = crops['layer0']
    crop1a = crops['layer1a']
    crop1b = crops['layer1b']
    crop2 = crops['layer2']
    # Layer 0
    assert_array_equal(crop0['csr_adj'], ([0, 1, 2, 3, 3, 3, 4], [1, 2, 3, 4, 2, 0, 5]))
    assert crop0['in_nodes'] == set([0, 1, 2, 3, 4, 5])
    assert crop0['out_nodes'] == set([0, 1, 2, 3, 4])
    # Layer 1a
    assert_array_equal(crop1a['csr_adj'], ([1, 2, 3, 3, 3], [2, 3, 4, 2, 0]))
    assert crop1a['in_nodes'] == set([0, 1, 2, 3, 4])
    assert crop1a['out_nodes'] == set([1, 2, 3])
    # Layer 1b
    assert_array_equal(crop1b['csr_adj'], ([1, 2, 3, 3, 3], [2, 3, 4, 2, 0]))
    assert crop1b['in_nodes'] == set([0, 1, 2, 3, 4])
    assert crop1b['out_nodes'] == set([1, 2, 3])
    # Layer 2
    assert_array_equal(crop2['csr_adj'], ([1, 2], [2, 3]))
    assert crop2['in_nodes'] == set([1, 2, 3])
    assert crop2['out_nodes'] == set([1, 2])

    # Works well with good arguments and sampling
    crops = batching._collect_layers_crops(model_depth3, adj, set([1, 2]), 2)
    assert set(crops.keys()) == set(['layer0', 'layer1a', 'layer1b', 'layer2'])
    crop0 = crops['layer0']
    crop1a = crops['layer1a']
    crop1b = crops['layer1b']
    crop2 = crops['layer2']
    # Layer 0
    # Not testing crop0['csr_adj'] as the randomness makes it hellish.
    # If all the rest works, then it should work.
    assert 4 <= len(crop0['in_nodes']) <= 6
    assert set([1, 2, 3]).issubset(crop0['in_nodes'])
    assert crop0['in_nodes'].issubset([0, 1, 2, 3, 4, 5])
    assert crop0['out_nodes'] == crop1a['in_nodes'].union(crop1b['in_nodes'])
    # Layer 1a
    assert_array_equal(crop1a['csr_adj'][0], [1, 2, 3, 3])
    assert_array_equal(crop1a['csr_adj'][1][:2], [2, 3])
    assert set(crop1a['csr_adj'][1][2:]).issubset([0, 2, 4])
    assert 4 <= len(crop1a['in_nodes']) <= 5
    assert set([1, 2, 3]).issubset(crop1a['in_nodes'])
    assert 0 in crop1a['in_nodes'] or 4 in crop1a['in_nodes']
    assert crop1a['in_nodes'].issubset(set([0, 1, 2, 3, 4]))
    assert crop1a['out_nodes'] == set([1, 2, 3])
    # Layer 1b
    assert_array_equal(crop1b['csr_adj'][0], [1, 2, 3, 3])
    assert_array_equal(crop1b['csr_adj'][1][:2], [2, 3])
    assert set(crop1b['csr_adj'][1][2:]).issubset([0, 2, 4])
    assert 4 <= len(crop1b['in_nodes']) <= 5
    assert set([1, 2, 3]).issubset(crop1b['in_nodes'])
    assert 0 in crop1b['in_nodes'] or 4 in crop1b['in_nodes']
    assert crop1b['in_nodes'].issubset(set([0, 1, 2, 3, 4]))
    assert crop1b['out_nodes'] == set([1, 2, 3])
    # Layer 2
    assert_array_equal(crop2['csr_adj'], ([1, 2], [2, 3]))
    assert crop2['in_nodes'] == set([1, 2, 3])
    assert crop2['out_nodes'] == set([1, 2])

    # Rejects out-of-bounds indices in `final_nodes`
    with pytest.raises(IndexError):
        batching._collect_layers_crops(model_depth3, adj, set([6]), None)

    # Rejects negative indices in `final_nodes`
    with pytest.raises(AssertionError):
        batching._collect_layers_crops(model_depth3, adj, set([-1]), None)

    # Takes only sets for `final_nodes`
    with pytest.raises(AssertionError):
        batching._collect_layers_crops(model_depth3, adj, set([-6]), None)


def test__compute_batch(model_depth2):
    # Our test data
    adj = sparse.csr_matrix(np.array([[0, 1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0],
                                      [1, 0, 1, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 0, 0]]))

    # Works well with good arguments and no sampling
    required_nodes, feeds = batching._compute_batch(model_depth2, adj, set([1, 2]), None)
    assert_array_equal(required_nodes, [0, 1, 2, 3, 4])
    assert set(feeds.keys()) == set(['layer1a_adj', 'layer1a_output_mask',
                                     'layer1b_adj', 'layer1b_output_mask',
                                     'layer2_adj', 'layer2_output_mask'])
    # Layer 1a
    assert_array_equal(feeds['layer1a_adj'].toarray(), [[0, 0, 0, 0, 0],
                                                        [0, 0, 1, 0, 0],
                                                        [0, 0, 0, 1, 0],
                                                        [1, 0, 1, 0, 1],
                                                        [0, 0, 0, 0, 0]])
    assert_array_equal(feeds['layer1a_output_mask'], [0, 1, 1, 1, 0])
    # Layer 1b
    assert_array_equal(feeds['layer1b_adj'], feeds['layer1a_adj'])
    assert_array_equal(feeds['layer1b_output_mask'], feeds['layer1a_output_mask'])
    # Layer 2
    assert_array_equal(feeds['layer2_adj'].toarray(), [[0, 0, 0, 0, 0],
                                                       [0, 0, 1, 0, 0],
                                                       [0, 0, 0, 1, 0],
                                                       [0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0]])
    assert_array_equal(feeds['layer2_output_mask'], [0, 1, 1, 0, 0])

    # No test with sampling as it will become hellish and non-maintainable
    # (but the constituent parts are tested with sampling)

    # Rejects out-of-bounds indices in `final_nodes`
    with pytest.raises(IndexError):
        batching._compute_batch(model_depth2, adj, set([6]), None)

    # Rejects negative indices in `final_nodes`
    with pytest.raises(AssertionError):
        batching._compute_batch(model_depth2, adj, set([-1]), None)

    # Takes only sets for `final_nodes`
    with pytest.raises(AssertionError):
        batching._compute_batch(model_depth2, adj, set([-6]), None)


def test__collect_maxed_connected_component():
    # Our test data
    adj = sparse.csr_matrix(np.array([[1, 1, 1, 0, 0, 0],
                                      [1, 1, 0, 0, 0, 0],
                                      [1, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1, 0]]))
    g = nx.from_numpy_array(adj)

    # Works well with good arguments, not maxing out, no exclusions
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 0, 3, set(), collected)
    assert collected == set([0, 1, 2])
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 1, 4, set(), collected)
    assert collected == set([0, 1, 2])
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 2, 5, set(), collected)
    assert collected == set([0, 1, 2])
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 3, 5, set(), collected)
    assert collected == set([3])
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 4, 2, set(), collected)
    assert collected == set([4, 5])
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 5, 3, set(), collected)
    assert collected == set([4, 5])

    # Works well with good arguments, maxing out, no exclusions
    assert batching._collect_maxed_connected_component(g, 0, 2, set(), set())
    assert batching._collect_maxed_connected_component(g, 1, 2, set(), set())
    assert batching._collect_maxed_connected_component(g, 2, 2, set(), set())
    assert batching._collect_maxed_connected_component(g, 3, 0, set(), set())
    assert batching._collect_maxed_connected_component(g, 4, 1, set(), set())
    assert batching._collect_maxed_connected_component(g, 5, 1, set(), set())

    # Works well with good arguments, not maxing out, with exclusions
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 0, 2, set([1]), collected)
    assert collected == set([0, 1, 2])
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 1, 3, set([0]), collected)
    assert collected == set([0, 1, 2])
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 2, 4, set([0]), collected)
    assert collected == set([0, 1, 2])
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 3, 0, set([3]), collected)
    assert collected == set([3])
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 4, 1, set([4]), collected)
    assert collected == set([4, 5])
    collected = set()
    assert not batching._collect_maxed_connected_component(g, 5, 2, set([0, 1, 2]), collected)
    assert collected == set([4, 5])

    # Works well with good arguments, maxing out, with exclusions
    assert batching._collect_maxed_connected_component(g, 0, 1, set([1]), set())
    assert batching._collect_maxed_connected_component(g, 1, 2, set([4, 5]), set())
    assert batching._collect_maxed_connected_component(g, 2, 1, set([0]), set())
    assert batching._collect_maxed_connected_component(g, 3, 0, set([4]), set())
    assert batching._collect_maxed_connected_component(g, 4, 1, set([3]), set())
    assert batching._collect_maxed_connected_component(g, 5, 0, set([4]), set())

    # Raises an error if the source node has already been collected
    with pytest.raises(AssertionError):
        batching._collect_maxed_connected_component(g, 2, None, set(), set([2]))


def test_filtered_connected_component_or_none():
    # Our test data
    adj = sparse.csr_matrix(np.array([[1, 1, 1, 0, 0, 0],
                                      [1, 1, 0, 0, 0, 0],
                                      [1, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1, 0]]))
    g = nx.from_numpy_array(adj)

    # Works well with good arguments, not maxing out, no exclusions
    assert batching.filtered_connected_component_or_none(g, 0, 3, set()) == set([0, 1, 2])
    assert batching.filtered_connected_component_or_none(g, 1, 4, set()) == set([0, 1, 2])
    assert batching.filtered_connected_component_or_none(g, 2, 5, set()) == set([0, 1, 2])
    assert batching.filtered_connected_component_or_none(g, 3, 5, set()) == set([3])
    assert batching.filtered_connected_component_or_none(g, 4, 2, set()) == set([4, 5])
    assert batching.filtered_connected_component_or_none(g, 5, 3, set()) == set([4, 5])

    # Works well with good arguments, maxing out, no exclusions
    assert batching.filtered_connected_component_or_none(g, 0, 2, set()) is None
    assert batching.filtered_connected_component_or_none(g, 1, 2, set()) is None
    assert batching.filtered_connected_component_or_none(g, 2, 2, set()) is None
    assert batching.filtered_connected_component_or_none(g, 3, 0, set()) is None
    assert batching.filtered_connected_component_or_none(g, 4, 1, set()) is None
    assert batching.filtered_connected_component_or_none(g, 5, 1, set()) is None

    # Works well with good arguments, not maxing out, with exclusions
    assert batching.filtered_connected_component_or_none(g, 0, 2, set([1])) == set([0, 2])
    assert batching.filtered_connected_component_or_none(g, 1, 3, set([0])) == set([1, 2])
    assert batching.filtered_connected_component_or_none(g, 2, 4, set([0])) == set([1, 2])
    assert batching.filtered_connected_component_or_none(g, 3, 0, set([3])) == set()
    assert batching.filtered_connected_component_or_none(g, 4, 1, set([4])) == set([5])
    assert batching.filtered_connected_component_or_none(g, 5, 2, set([0, 1, 2])) == set([4, 5])

    # Works well with good arguments, maxing out, with exclusions
    assert batching.filtered_connected_component_or_none(g, 0, 1, set([1])) is None
    assert batching.filtered_connected_component_or_none(g, 1, 2, set([4, 5])) is None
    assert batching.filtered_connected_component_or_none(g, 2, 1, set([0])) is None
    assert batching.filtered_connected_component_or_none(g, 3, 0, set([4])) is None
    assert batching.filtered_connected_component_or_none(g, 4, 1, set([3])) is None
    assert batching.filtered_connected_component_or_none(g, 5, 0, set([4])) is None


def test_distinct_random_walk():
    # Our test data
    adj = sparse.csr_matrix(np.array([[1, 1, 0, 0],
                                      [1, 1, 1, 0],
                                      [0, 1, 1, 1],
                                      [0, 0, 1, 1]]))
    g = nx.from_numpy_array(adj)

    # Gets the entire connected component if the randow walk is long enough, no exclusions
    assert batching.distinct_random_walk(g, 4, set()) == set([0, 1, 2, 3])

    # Gets the entire connected component if the randow walk is long enough, with exclusions,
    # walking across a separating set of excluded nodes if necessary
    assert batching.distinct_random_walk(g, 4, set([2])) == set([0, 1, 3])

    # Gets a subset of the connected component, no exclusions
    walk = batching.distinct_random_walk(g, 3, set())
    assert len(walk) == 3
    assert walk.issubset([0, 1, 2, 3])

    # Gets a subset of the connected component, with exclusions
    walk = batching.distinct_random_walk(g, 2, set([2]))
    assert len(walk) == 2
    assert walk.issubset([0, 1, 3])


def test_jumpy_distinct_random_walk():
    # Our test data
    adj = sparse.csr_matrix(np.array([[1, 1, 0, 0, 0, 0],
                                      [1, 1, 1, 0, 0, 0],
                                      [0, 1, 1, 1, 0, 0],
                                      [0, 0, 1, 1, 0, 0],
                                      [0, 0, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1, 1]]))
    g = nx.from_numpy_array(adj)

    # A `max_size` larger than the network returns the whole network
    assert batching.jumpy_distinct_random_walk(g, 10, 3) == set(range(adj.shape[0]))

    # A `max_size` smaller than the network returns a set of size `max_size`
    assert len(batching.jumpy_distinct_random_walk(g, 4, 3)) == 4

    # If all connected components are multiples of `max_walk_length`, the left-overs
    # are also multiples of `max_walk_length`
    walk = batching.jumpy_distinct_random_walk(g, 4, 2)
    left_overs = set(range(adj.shape[0])).difference(walk)
    assert (left_overs == set([0, 1]) or
            left_overs == set([0, 2]) or
            left_overs == set([0, 3]) or
            left_overs == set([1, 2]) or
            left_overs == set([1, 3]) or
            left_overs == set([2, 3]) or
            left_overs == set([4, 5]))


def test_jumpy_walks():
    # Rejects a non-ndarray numpy matrix
    with pytest.raises(AssertionError):
        list(batching.jumpy_walks(np.ones((5, 5)) - np.eye(5), 3, 2))
    # Rejects a directed graph
    with pytest.raises(AssertionError):
        list(batching.jumpy_walks(sparse.csr_matrix(np.array([[0, 1],
                                                              [0, 0]])),
                                  3, 2))
    # Rejects a weighted graph
    with pytest.raises(AssertionError):
        list(batching.jumpy_walks(sparse.csr_matrix(np.array([[0, 2],
                                                              [2, 0]])),
                                  3, 2))
    # Rejects a graph with self-connections
    with pytest.raises(AssertionError):
        list(batching.jumpy_walks(sparse.csr_matrix(np.array([[1, 1],
                                                              [1, 0]])),
                                  3, 2))

    # Yields the right number of walks to span the whole graph
    adj = sparse.csr_matrix(np.array([[0, 1, 0, 0, 0],
                                      [1, 0, 1, 0, 0],
                                      [0, 1, 0, 1, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0]]))
    walks = list(batching.jumpy_walks(adj, 2, 2))
    assert len(walks) == 3


def test_epoch_batches(model_depth2):
    # Our test data
    adj = sparse.csr_matrix(np.array([[0, 1, 0, 0, 0],
                                      [1, 0, 1, 0, 0],
                                      [0, 1, 0, 1, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0]]))

    # Works well with good arguments, no sampling
    batches = list(batching.epoch_batches(model_depth2, adj, 2, 2))
    assert len(batches) == 3
    required_nodes, final_nodes, feeds = batches[0]
    assert_array_equal(final_nodes, sorted(final_nodes))
    assert len(final_nodes) == 2
    assert set(feeds.keys()) == set(['layer1a_adj', 'layer1a_output_mask',
                                     'layer1b_adj', 'layer1b_output_mask',
                                     'layer2_adj', 'layer2_output_mask'])

    # Works well with good arguments, with sampling
    batches = list(batching.epoch_batches(model_depth2, adj, 2, 2, neighbour_samples=1))
    assert len(batches) == 3
    required_nodes, final_nodes, feeds = batches[0]
    assert_array_equal(final_nodes, sorted(final_nodes))
    assert len(final_nodes) == 2
    assert set(feeds.keys()) == set(['layer1a_adj', 'layer1a_output_mask',
                                     'layer1b_adj', 'layer1b_output_mask',
                                     'layer2_adj', 'layer2_output_mask'])
