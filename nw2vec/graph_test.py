import pytest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse

from nw2vec import graph
from nw2vec import utils


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
    return graph.CSGraph(sparse.csr_matrix(np.array(adj)), .5, 4)


def assert_alias_equal(alias1, alias2):
    assert_array_equal(alias1[0], alias2[0])
    assert_array_equal(alias1[1], alias2[1])


def test_CSGraph_init():
    # Works with good arguments
    adj = sparse.csr_matrix(np.array([[1, 1], [1, 0]]))
    adj.data[0] = 0
    g = graph.CSGraph(adj, .5, 4)
    # Got the parameters right
    assert g._p == .5
    assert g._q == 4
    # Eliminates zeros in adjacency matrix
    assert_array_equal(g.adj.data, [1, 1])
    assert_array_equal(g.adj.toarray(), [[0, 1], [1, 0]])
    # Computed the edge aliases
    assert set(g._edge_aliases.keys()) == {(0, 1), (1, 0)}
    assert_alias_equal(g._edge_aliases[(0, 1)], ([0], [1]))

    # Rejects non-sparse matrices
    with pytest.raises(ValueError):
        graph.CSGraph(np.array([[0, 1],
                                [0, 0]]),
                      .5, 4)

    # Rejects directed graphs
    with pytest.raises(AssertionError):
        graph.CSGraph(sparse.csr_matrix(np.array([[0, 1],
                                                  [0, 0]])),
                      .5, 4)

    # Rejects weighted graphs
    with pytest.raises(AssertionError):
        graph.CSGraph(sparse.csr_matrix(np.array([[0, 1],
                                                  [2, 0]])),
                      .5, 4)

    # Rejects diagonal elements
    with pytest.raises(AssertionError):
        graph.CSGraph(sparse.csr_matrix(np.array([[1, 1],
                                                  [1, 0]])),
                      .5, 4)


def test_CSGraph__get_alias_edge(g):
    # Works well with good parameters
    probs01 = np.array([2, 1, 1/4])
    assert_alias_equal(g._get_alias_edge(0, 1), utils.alias_setup(probs01 / probs01.sum()))

    # Fails if `dst` has no neighbours (which should never happen,
    # since we always arrive through a link)
    g.adj = sparse.csr_matrix(np.array([[0, 1, 1, 1, 0, 0],
                                        [1, 0, 0, 1, 0, 0],
                                        [1, 0, 0, 1, 0, 0],
                                        [1, 1, 1, 0, 1, 0],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0]]))
    with pytest.raises(AssertionError):
        g._get_alias_edge(1, 5)


def test_CSGraph__init_transition_probs(g):
    # Gives the right values upon class instanciation
    set(g._edge_aliases.keys()) == {(0, 1), (0, 2), (0, 3),
                                    (1, 0), (1, 3), (1, 5),
                                    (2, 0), (2, 3), (2, 5),
                                    (3, 0), (3, 1), (3, 2), (3, 4),
                                    (4, 3), (4, 5),
                                    (5, 1), (5, 2), (5, 4)}
    probs20 = np.array([1/4, 2, 1])
    assert_array_equal(g._edge_aliases[(2, 0)], utils.alias_setup(probs20 / probs20.sum()))
    probs31 = np.array([1, 2, 1/4])
    assert_array_equal(g._edge_aliases[(3, 1)], utils.alias_setup(probs31 / probs31.sum()))


def test_CSGraph_draw_after_edge(g):
    # Works well with good parameters
    assert g.draw_after_edge(0, 1) in [0, 3, 5]
    assert g.draw_after_edge(1, 3) in [0, 1, 2, 4]
    draws01 = set()
    draws13 = set()
    for _ in range(100):
        draws01.add(g.draw_after_edge(0, 1))
        draws13.add(g.draw_after_edge(1, 3))
    assert draws01 == {0, 3, 5}
    assert draws13 == {0, 1, 2, 4}

    # Fails if the edge is not found
    with pytest.raises(KeyError):
        g.draw_after_edge(1, 2)


def test_CSGraph_neighbors(g):
    # Gives the right value
    assert_array_equal(g.neighbors(1), [0, 3, 5])

    # Fails on out of bounds
    with pytest.raises(AssertionError):
        g.neighbors(-1)
    with pytest.raises(AssertionError):
        g.neighbors(6)


def test_CSGraph_eq():
    # Two graphs initialised with the same value are equal
    g1 = graph.CSGraph(sparse.csr_matrix(np.array([[0, 1, 0],
                                                   [1, 0, 1],
                                                   [0, 1, 0]])),
                       2, 4)
    g2 = graph.CSGraph(sparse.csr_matrix(np.array([[0, 1, 0],
                                                   [1, 0, 1],
                                                   [0, 1, 0]])),
                       2, 4)
    assert g1 == g2

    # Two graphs initialised with different matrices are different
    g1 = graph.CSGraph(sparse.csr_matrix(np.array([[0, 1, 1],
                                                   [1, 0, 1],
                                                   [1, 1, 0]])),
                       2, 4)
    g2 = graph.CSGraph(sparse.csr_matrix(np.array([[0, 1, 0],
                                                   [1, 0, 1],
                                                   [0, 1, 0]])),
                       2, 4)
    assert g1 != g2

    # Two graphs initialised with different p or q are different
    g1 = graph.CSGraph(sparse.csr_matrix(np.array([[0, 1, 0],
                                                   [1, 0, 1],
                                                   [0, 1, 0]])),
                       2, 4)
    g2 = graph.CSGraph(sparse.csr_matrix(np.array([[0, 1, 0],
                                                   [1, 0, 1],
                                                   [0, 1, 0]])),
                       3, 4)
    assert g1 != g2

    # Two graphs initialised with different matrices are different
    g1 = graph.CSGraph(sparse.csr_matrix(np.array([[0, 1, 0],
                                                   [1, 0, 1],
                                                   [0, 1, 0]])),
                       2, 4)
    g2 = graph.CSGraph(sparse.csr_matrix(np.array([[0, 1, 0],
                                                   [1, 0, 1],
                                                   [0, 1, 0]])),
                       2, 3)
    assert g1 != g2


def test_get_csgraph():
    # `adj_a` and `adj_b` are two different objects
    adj_a = sparse.csr_matrix(np.ones((5, 5)) - np.eye(5))
    adj_b = sparse.csr_matrix(np.ones((5, 5)) - np.eye(5))
    assert id(adj_a) != id(adj_b)

    # `g11_a` and `g11_b` are the same object
    g11_a = graph.get_csgraph(adj_a, 1, 1)
    g11_b = graph.get_csgraph(adj_b, 1, 1)
    assert id(g11_a) == id(g11_b)

    # `g11_a` and `g12_a` are different objects
    g12_a = graph.get_csgraph(adj_a, 1, 2)
    assert id(g11_a) != id(g12_a)
