import pytest
import numpy as np
from numpy.testing import assert_array_equal
import keras

from nw2vec import ae
from nw2vec import batching


def test__layer_csr_adj():
    # Our test data
    adj = np.array([[0, 1, 0, 0, 1],
                    [1, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1]])

    # Works with good arguments and no sampling
    row_ind, col_ind = batching._layer_csr_adj(set([1, 2]), adj, None)
    assert_array_equal(row_ind, [1, 1, 1, 2])
    assert_array_equal(col_ind, [0, 2, 3, 2])

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
def model():
    layer_input = keras.layers.Input(shape=(20,), name='layer_input')
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


def test__collect_layers_crops(model):
    # Our test data
    adj = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0]])

    # Works well with good arguments and no sampling
    crops = batching._collect_layers_crops(model, adj, set([1, 2]), None)
    crop0 = crops['layer0']
    crop1a = crops['layer1a']
    crop1b = crops['layer1b']
    crop2 = crops['layer2']
    # Layer 0
    assert_array_equal(crop0['csr_adj'], ([0, 1, 2, 3, 3, 3, 4], [1, 2, 3, 0, 2, 4, 5]))
    assert crop0['in_nodes'] == set([0, 1, 2, 3, 4, 5])
    assert crop0['out_nodes'] == set([0, 1, 2, 3, 4])
    # Layer 1a
    assert_array_equal(crop1a['csr_adj'], ([1, 2, 3, 3, 3], [2, 3, 0, 2, 4]))
    assert crop1a['in_nodes'] == set([0, 1, 2, 3, 4])
    assert crop1a['out_nodes'] == set([1, 2, 3])
    # Layer 1b
    assert_array_equal(crop1b['csr_adj'], ([1, 2, 3, 3, 3], [2, 3, 0, 2, 4]))
    assert crop1b['in_nodes'] == set([0, 1, 2, 3, 4])
    assert crop1b['out_nodes'] == set([1, 2, 3])
    # Layer 2
    assert_array_equal(crop2['csr_adj'], ([1, 2], [2, 3]))
    assert crop2['in_nodes'] == set([1, 2, 3])
    assert crop2['out_nodes'] == set([1, 2])

    # Works well with good arguments and sampling
    crops = batching._collect_layers_crops(model, adj, set([1, 2]), 2)
    crop0 = crops['layer0']
    crop1a = crops['layer1a']
    crop1b = crops['layer1b']
    crop2 = crops['layer2']
    # Layer 0
    # Not testing crop0['csr_adj'] as the randomness make it hellish.
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
        batching._collect_layers_crops(model, adj, set([6]), None)

    # Rejects negative indices in `final_nodes`
    with pytest.raises(AssertionError):
        batching._collect_layers_crops(model, adj, set([-1]), None)

    # Takes only sets for `final_nodes`
    with pytest.raises(AssertionError):
        batching._collect_layers_crops(model, adj, set([-6]), None)


@pytest.mark.skip(reason='TODO')
def test__compute_batch():
    pass


@pytest.mark.skip(reason='TODO')
def test__collect_maxed_connected_component():
    pass


@pytest.mark.skip(reason='TODO')
def test_filtered_connected_component_or_none():
    pass


@pytest.mark.skip(reason='TODO')
def test_distinct_random_walk():
    pass


@pytest.mark.skip(reason='TODO')
def test_jumpy_distinct_random_walk():
    pass


@pytest.mark.skip(reason='TODO')
def test_jumpy_walks():
    pass


@pytest.mark.skip(reason='TODO')
def test_epoch_batches():
    pass
