import pytest

from nw2vec import utils


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
