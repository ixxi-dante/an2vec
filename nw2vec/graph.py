import numpy as np
from scipy import sparse
import joblib


def csr_eye(n):
    return sparse.csr_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))


class CSGraph:

    _REMOVED_LABEL = np.iinfo(np.int).max

    def __init__(self, adj):
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

        self._adj = adj.copy()
        self._adj.eliminate_zeros()
        self._initial_shape = self._adj.shape
        self._ids2labels = np.arange(adj.shape[0])
        self._labels2ids = np.arange(adj.shape[0])
        self._kept_labels_bool = np.ones(adj.shape[0], dtype=bool)

    def labels_exist(self, labels):
        if isinstance(labels, set):
            labels = np.array(list(labels))
        elif isinstance(labels, (list, range)):
            labels = np.array(labels)
        elif not isinstance(labels, np.ndarray):
            raise ValueError('Unsupported label type')

        if len(labels) == 0:
            return np.array([])

        if ((labels < 0) + (labels >= self._initial_shape[0])).any():
            raise IndexError

        return self._labels2ids[labels] != self._REMOVED_LABEL

    def remove_nodes_from(self, labels):
        # Process labels
        labels = list(labels)
        assert self.labels_exist(labels).all()

        # Pre-compute the shift of ids due to node removals
        shifts = np.zeros_like(self._ids2labels, dtype=int)
        shifts[self._labels2ids[labels]] = 1
        shifts = shifts.cumsum()

        # Update our tracking of which labels are kept, get the corresponding ids,
        # and reduce `shifts` to only tho kept ids
        self._kept_labels_bool[labels] = False
        kept_ids = self._labels2ids[self._kept_labels_bool]
        shifts = shifts[kept_ids]

        # `labels2ids` receives the shifts for labels that are kept,
        # and removed labels are marked as such
        self._labels2ids[self._kept_labels_bool] -= shifts
        self._labels2ids[labels] = self._REMOVED_LABEL

        # Reduce ids2labels
        self._ids2labels = self._ids2labels[kept_ids]

        # Reduce the adjacency matrix
        self._adj = self._adj[kept_ids, :][:, kept_ids]

    def neighbors(self, label):
        assert label >= 0 and label < self._initial_shape[0]
        idx = self._labels2ids[label]
        neighbor_ids = self._adj.indices[self._adj.indptr[idx]:self._adj.indptr[idx + 1]]
        return self._ids2labels[neighbor_ids]

    @property
    def nodes(self):
        return self._ids2labels

    @property
    def adj(self):
        return self._adj

    def copy(self):
        other = CSGraph(self._adj)
        other._ids2labels = self._ids2labels.copy()
        other._labels2ids = self._labels2ids.copy()
        other._kept_labels_bool = self._kept_labels_bool.copy()
        return other

    def __key(self):
        key = (self._adj, self._ids2labels, self._labels2ids)
        return tuple(map(joblib.hash, key))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()
