import pytest
import numpy as np
from numpy.testing import assert_array_equal

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

    # Takes only sets
    with pytest.raises(AssertionError):
        batching.mask_indices([1], 5)

    # Fails when indices are out of bounds
    with pytest.raises(IndexError):
        batching.mask_indices(set([5]), 5)
    with pytest.raises(IndexError):
        batching.mask_indices(set([-6]), 5)


@pytest.mark.skip(reason='TODO')
def test__collect_layers_crops():
    pass


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
