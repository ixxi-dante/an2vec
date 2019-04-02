import pickle

import numpy as np
import scipy.sparse as sp
import networkx as nx


# (Dataset name, whether or not to clean it)
DATASETS = [
    ('cora', False),
    ('citeseer', False),
    ('citeseer', True),
    ('pubmed', False)
]
DATASET_PARTS = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph']
INPATH = "gcn/data"
OUTPATH = 'datasets/gae-benchmarks'


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_files(name, clean):
    # Load features
    parts = {}
    for part_name in DATASET_PARTS:
        with open(INPATH + '/ind.{}.{}'.format(name, part_name), 'rb') as f:
            # Load a scipy sparse matrix, convert it to a dense array
            parts[part_name] = pickle.load(f, encoding='latin1')

    # Load adjacency matrix
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(parts['graph'])).tocoo()
    del parts['graph']

    # Densify feature and label arrays
    for part_name, part in parts.items():
        if not isinstance(part, np.ndarray):
            parts[part_name] = part.toarray()

    # Load test indices
    parts['test-indices'] = \
        np.array(
            parse_index_file(INPATH + '/ind.{}.test.index'.format(name))
        )

    # Fix citeseer dataset:
    # some test indices are out-of-bounds for tx, meaning that some nodes
    # are not included in tx. We create a tx where indices referenced in
    # the test indices are taken from the old tx, and the missing indices
    # are assigned zeros.
    if name == 'citeseer':
        test_indices_span = (parts['test-indices'].max()
                             - parts['test-indices'].min() + 1)
        ordered_test_indices = np.sort(parts['test-indices'])
        assert ordered_test_indices.min() == parts['allx'].shape[0]
        assert ordered_test_indices.min() == parts['ally'].shape[0]

        tx_extended = np.zeros((test_indices_span, parts['tx'].shape[1]))
        tx_extended[ordered_test_indices - ordered_test_indices.min(), :] = \
            parts['tx']
        parts['tx'] = tx_extended

        ty_extended = np.zeros((test_indices_span, parts['ty'].shape[1]))
        ty_extended[ordered_test_indices - ordered_test_indices.min(), :] = \
            parts['ty']
        parts['ty'] = ty_extended

    # Assemble parts of the features and labels
    features, labels = assemble_features_labels(parts)

    if clean:
        # Get indices of isolated nodes
        adj = adj.toarray()
        selfconnections = np.where(np.diag(adj))[0]
        adj[selfconnections, selfconnections] = 0
        isolated = np.where(adj.sum(1) == 0)[0]

        # Get indices of nodes with 0 features or 0 labels
        features0 = np.where(features.sum(1) == 0)[0]
        labels0 = np.where(labels.sum(1) == 0)[0]

        # Join indices and remove from the dataset
        keep = sorted(set(range(adj.shape[0]))
                      .difference(isolated, features0, labels0))
        adj = adj[keep, :][:, keep]
        features = features[keep, :]
        labels = labels[keep, :]
        adj = sp.coo_matrix(adj)

    return (adj.data, adj.row, adj.col), features, labels


def assemble_features_labels(parts):
    sorted_test_indices = np.sort(parts['test-indices'])

    features = np.vstack((parts['allx'], parts['tx']))
    features[parts['test-indices'], :] = features[sorted_test_indices, :]

    labels = np.vstack((parts['ally'], parts['ty']))
    labels[parts['test-indices'], :] = labels[sorted_test_indices, :]

    return features, labels


if __name__ == '__main__':
    print("Loading dataset parts and saving them as npz")
    for name, clean in DATASETS:
        print('Loading {} dataset (cleaning = {})...'.format(name, clean))
        (adjdata, adjrow, adjcol), features, labels = load_files(name, clean)
        npzpath = OUTPATH + "/{}{}.npz".format(name, '-clean' if clean else '')
        print('Saving to "{}"...'.format(npzpath))
        np.savez_compressed(npzpath,
                            features=features, labels=labels,
                            adjdata=adjdata, adjrow=adjrow, adjcol=adjcol)
    print('All done.')
