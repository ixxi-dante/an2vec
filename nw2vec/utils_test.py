import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import numba

from nw2vec import utils


def test_alias_setup():
    # Fails if `probs` is not an ndarray or is empty
    with pytest.raises(numba.errors.TypingError):
        utils.alias_setup([.5, .5])
    with pytest.raises(AssertionError):
        utils.alias_setup(np.array([]))

    # Works with good arguments
    probs = np.ones(4)
    probs = probs / probs.sum()
    choices, weights = utils.alias_setup(probs)
    assert_array_equal(choices, [0, 0, 0, 0])
    assert_array_equal(weights, [1, 1, 1, 1])

    probs = np.array([1, 1, 2, 2])
    probs = probs / probs.sum()
    choices, weights = utils.alias_setup(probs)
    assert_array_equal(choices, [3, 3, 0, 2])
    assert_allclose(weights, [2 / 3, 2 / 3, 1, 2 / 3], rtol=1e-6)

    probs = np.arange(5)
    probs = probs / probs.sum()
    choices, weights = utils.alias_setup(probs)
    assert_array_equal(choices, [4, 4, 0, 0, 3])
    assert_array_equal(weights, [0, .5, 1, 1, .5])


def test_alias_draw():
    # Works with good arguments
    probs = np.arange(5)
    probs = probs / probs.sum()
    choices, weights = utils.alias_setup(probs)
    assert utils.alias_draw(choices, weights) in range(1, 5)

    # Gives the right statistics
    n_draws = 1000
    draws = np.array([utils.alias_draw(choices, weights) for _ in range(n_draws)])
    counts = [(draws == i).sum() for i in range(5)]
    assert counts[0] == 0
    assert np.sum(counts[1:]) == n_draws
    assert_allclose(counts[1:], n_draws * probs[1:], atol=50)

    # Fails with empty or None inputs
    with pytest.raises(numba.errors.TypingError):
        utils.alias_draw(None, weights)
    with pytest.raises(numba.errors.TypingError):
        utils.alias_draw(choices, None)
    with pytest.raises(numba.errors.TypingError):
        utils.alias_draw(None, None)
    with pytest.raises(AssertionError):
        utils.alias_draw(choices[:0], weights)
    with pytest.raises(AssertionError):
        utils.alias_draw(choices, weights[:0])
    with pytest.raises(AssertionError):
        utils.alias_draw(choices[:0], weights[:0])


@pytest.mark.skip(reason='Implement test')
def test_csr_to_sparse_tensor_parts():
    pass


@pytest.mark.skip(reason='Implement test')
def test_sparse_tensor_parts_to_csr():
    pass


@pytest.mark.skip(reason='Implement test')
def test__csr_to_sparse_tensor_parts():
    pass


@pytest.mark.skip(reason='Implement test')
def test_inner_repeat():
    pass


def test_grouper():
    assert list(map(list, utils.grouper(range(5), 2))) == [[0, 1], [2, 3], [4]]


@pytest.mark.skip(reason='Implement test')
def test_scale_center():
    pass


@pytest.mark.skip(reason='Implement test')
def test_softmax():
    pass


@pytest.mark.skip(reason='Implement test')
def test_Softmax():
    pass


@pytest.mark.skip(reason='Implement test')
def test_broadcast_left():
    pass


@pytest.mark.skip(reason='Implement test')
def test_get_backend():
    pass


@pytest.mark.skip(reason='Implement test')
def test_right_squeeze2():
    pass


@pytest.mark.skip(reason='Implement test')
def test_expand_dims_tile():
    pass
