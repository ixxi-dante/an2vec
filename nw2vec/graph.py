import numpy as np
from scipy import sparse
import joblib
from cached_property import cached_property

from nw2vec import utils


def csr_eye(n):
    return sparse.csr_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))


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
        self._init_transition_probs()

    def _get_alias_edge(self, src, dst):
        src_neigbours = self.neighbors(src)
        dst_neigbours = self.neighbors(dst)

        n_dst_neigbours = len(dst_neigbours)
        assert n_dst_neigbours > 0

        uprobs = np.zeros(n_dst_neigbours)
        for i, next_dst in enumerate(dst_neigbours):
            if next_dst == src:
                uprobs[i] = 1.0 / self._p
            elif next_dst in src_neigbours:
                uprobs[i] = 1.0
            else:
                uprobs[i] = 1.0 / self._q

        return utils.alias_setup(uprobs / uprobs.sum())

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
        assert idx >= 0 and idx < self.adj.shape[0]
        return self.adj.indices[self.adj.indptr[idx]:self.adj.indptr[idx + 1]]

    @cached_property
    def nodes(self):
        return np.arange(self.adj.shape[0])

    @cached_property
    def __key(self):
        key = (self.adj, self._p, self._q)
        return tuple(map(joblib.hash, key))

    def __hash__(self):
        return hash(self.__key)

    def __eq__(self, other):
        return self.__key == other.__key
