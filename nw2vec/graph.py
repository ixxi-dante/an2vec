import numba
import numpy as np
from scipy import sparse
import joblib
from cached_property import cached_property

from nw2vec import utils


def csr_eye(n):
    return sparse.csr_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))


def _caching_get_csgraph():
    cache = {}

    def _get_csgraph(adj, p, q):
        key = (joblib.hash(adj), p, q)
        if key not in cache:
            cache[key] = CSGraph(adj, p, q)
        return cache[key]

    return _get_csgraph


get_csgraph = _caching_get_csgraph()


class CSGraph:

    def __init__(self, adj, p, q):
        # Check the adjacency matrix is:
        # ... a sparse matrix...
        if not isinstance(adj, sparse.csr_matrix):
            raise ValueError
        # ... undirected...
        assert (adj.T != adj).sum() == 0
        # ... unweighted...
        assert ((adj.data == 0) | (adj.data == 1)).all()
        # ... with no diagonal elements.
        assert adj.multiply(csr_eye(adj.shape[0])).sum() == 0

        self.adj = adj.copy()
        self.adj.eliminate_zeros()
        self._p = p
        self._q = q

        self.nodes = np.arange(self.adj.shape[0])
        self._init_transition_probs()

    def _get_alias_edge(self, src, dst):
        return _get_alias_edge(src, dst,
                               self._p, self._q,
                               self.adj.data, self.adj.indices, self.adj.indptr, self.adj.shape)

    def _init_transition_probs(self):
        # Note: this first part will only be relevant once (if) we add edge weights
        # (for now it's the same as uniform sampling)
        # Create alias arrays for nodes without looking at incoming link
        # data_length = len(self.adj.data)
        # data_dtype = self.adj.data.dtype
        # node_alias_choices = np.zeros(data_length, dtype=np.int)
        # node_alias_weights = np.zeros(data_length, dtype=data_dtype)
        # for node in self.nodes:
        #     ptrs = slice(self.adj.indptr[node], self.adj.indptr[node + 1])
        #     uprobs = self.adj.data[ptrs]
        #     node_alias_choices[ptrs], node_alias_weights[ptrs] = \
        #         utils.alias_setup(uprobs / uprobs.sum())
        #
        # self.node_alias_choices = sparse.csr_matrix(
        #     (node_alias_choices, self.adj.indices, self.adj.indptr), shape=self.adj.shape)
        # self.node_alias_weights = sparse.csr_matrix(
        #     (node_alias_weights, self.adj.indices, self.adj.indptr), shape=self.adj.shape)

        # Create alias arrays for edges by looking at incoming link
        edge_aliases = {}
        for src_node in self.nodes:
            for dst_node in self.neighbors(src_node):
                edge_aliases[(src_node, dst_node)] = self._get_alias_edge(src_node, dst_node)
        self._edge_aliases = edge_aliases

    def draw_after_edge(self, src, dst):
        drawn = utils.alias_draw(*self._edge_aliases[(src, dst)])
        return self.neighbors(dst)[drawn]

    def neighbors(self, idx):
        return _neighbours(idx, self.adj.data, self.adj.indices, self.adj.indptr, self.adj.shape)

    @cached_property
    def __key(self):
        key = (self.adj, self._p, self._q)
        return tuple(map(joblib.hash, key))

    def __hash__(self):
        return hash(self.__key)

    def __eq__(self, other):
        return self.__key == other.__key


@numba.jit(nopython=True)
def _get_alias_edge(src, dst,
                    p, q,
                    adj_data, adj_indices, adj_indptr, adj_shape):
    src_neigbours = _neighbours(src, adj_data, adj_indices, adj_indptr, adj_shape)
    dst_neigbours = _neighbours(dst, adj_data, adj_indices, adj_indptr, adj_shape)

    n_dst_neigbours = len(dst_neigbours)
    assert n_dst_neigbours > 0

    uprobs = np.ones(n_dst_neigbours) / q
    uprobs[dst_neigbours == src] = 1.0 / p
    dst_src_neigbours = (np.expand_dims(dst_neigbours, 1)
                         == np.expand_dims(src_neigbours, 0)).sum(1).astype(np.bool8)
    uprobs[dst_src_neigbours] = 1

    return utils.alias_setup(uprobs / uprobs.sum())


@numba.jit(nopython=True)
def _neighbours(idx, adj_data, adj_indices, adj_indptr, adj_shape):
    assert idx >= 0 and idx < adj_shape[0]
    return adj_indices[adj_indptr[idx]:adj_indptr[idx + 1]]
