import pytest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse

from nw2vec import graph


def test_csr_eye():
    assert_array_equal(graph.csr_eye(5).toarray(), np.eye(5))


@pytest.fixture
def g():
    adj = [[0, 1, 1, 1, 0, 0],
           [1, 0, 0, 1, 0, 1],
           [1, 0, 0, 1, 0, 1],
           [1, 1, 1, 0, 1, 0],
           [0, 0, 0, 1, 0, 1],
           [0, 1, 1, 0, 1, 0]]
    return graph.CSGraph(sparse.csr_matrix(np.array(adj)))


def test_CSGraph_init():
    # Works with good arguments
    graph.CSGraph(sparse.csr_matrix(np.array([[0, 1],
                                              [1, 0]])))

    # Rejects non-sparse matrices
    with pytest.raises(ValueError):
        graph.CSGraph(np.array([[0, 1],
                                [0, 0]]))

    # Rejects directed graphs
    with pytest.raises(AssertionError):
        graph.CSGraph(sparse.csr_matrix(np.array([[0, 1],
                                                  [0, 0]])))

    # Rejects weighted graphs
    with pytest.raises(AssertionError):
        graph.CSGraph(sparse.csr_matrix(np.array([[0, 1],
                                                  [2, 0]])))

    # Rejects diagonal elements
    with pytest.raises(AssertionError):
        graph.CSGraph(sparse.csr_matrix(np.array([[1, 1],
                                                  [1, 0]])))


def test_CSGraph_labels_exist(g):
    # All initial labels exist
    assert g.labels_exist(range(6)).all()
    # This also works with smaller test lists, sets, or ndarrays, works
    assert g.labels_exist([2, 4]).all()
    assert g.labels_exist(set([2, 4])).all()
    assert g.labels_exist(np.array([2, 4])).all()
    # Removing labels make those labels not exist
    g.remove_nodes_from([2, 3])
    assert_array_equal(g.labels_exist([1, 2, 3, 4]), [True, False, False, True])
    # Negative labels raise an exception
    with pytest.raises(IndexError):
        g.labels_exist([-1, 0])
    # Out-of-range labels raise an exception
    with pytest.raises(IndexError):
        g.labels_exist([0, 6])
    # `labels` not being a list, set, range, or ndarray raises an exception
    with pytest.raises(ValueError):
        g.labels_exist({})


def test_CSGraph_remove_nodes_from(g):
    # Removing nodes works
    g.remove_nodes_from([2, 3, 4])
    assert_array_equal(g.nodes, [0, 1, 5])
    assert_array_equal(g.adj.toarray(), [[0, 1, 0],
                                         [1, 0, 1],
                                         [0, 1, 0]])
    # Removing negative nodes fails
    with pytest.raises(IndexError):
        g.remove_nodes_from(set([-1]))
    # Removing out-of-range nodes fails
    with pytest.raises(IndexError):
        g.remove_nodes_from(set([6]))
    # Removing nothing works
    g.remove_nodes_from([])
    assert_array_equal(g.nodes, [0, 1, 5])
    # Removing more nodes works
    g.remove_nodes_from(set([0, 1]))
    assert_array_equal(g.nodes, [5])
    assert_array_equal(g.adj.toarray(), [[0]])
    # Removing already-removed nodes fails
    with pytest.raises(AssertionError):
        g.remove_nodes_from(set([1, 5]))


def test_CSGraph_neighbors(g):
    assert_array_equal(g.neighbors(1), [0, 3, 5])
    g.remove_nodes_from([3, 5])
    assert_array_equal(g.neighbors(1), [0])


def test_CSGraph_copy(g):
    g2 = g.copy()
    # Original and copy are equal
    assert g == g2
    # Removing from the copy does not affect the original
    g2.remove_nodes_from([1, 5])
    assert not g2.labels_exist([1, 5]).any()
    assert g.labels_exist([1, 5]).all()
    assert_array_equal(g.neighbors(1), [0, 3, 5])


def test_CSGraph_eq():
    # Two graphs initialised with the same value are equal
    g1 = graph.CSGraph(sparse.csr_matrix(np.array([[0, 1, 0],
                                                   [1, 0, 1],
                                                   [0, 1, 0]])))
    g2 = graph.CSGraph(sparse.csr_matrix(np.array([[0, 1, 0],
                                                   [1, 0, 1],
                                                   [0, 1, 0]])))
    assert g1 == g2
    # And still equal after removing the same nodes
    g1.remove_nodes_from([1])
    g2.remove_nodes_from([1])
    assert g1 == g2
