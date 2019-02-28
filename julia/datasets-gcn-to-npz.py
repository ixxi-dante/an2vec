import pickle

import numpy as np
import networkx as nx


DATASET_NAMES = ['cora', 'citeseer', 'pubmed']
DATASET_PARTS = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph']
INPATH = "gcn/data"
OUTPATH = 'datasets/gae-benchmarks'


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_files(name):
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

    return (adj.data, adj.row, adj.col), parts


def assemble_features_labels(parts):
    sorted_test_indices = np.sort(parts['test-indices'])

    features = np.vstack((parts['allx'], parts['tx']))
    features[parts['test-indices'], :] = features[sorted_test_indices, :]

    labels = np.vstack((parts['ally'], parts['ty']))
    labels[parts['test-indices'], :] = labels[sorted_test_indices, :]

    return features, labels


if __name__ == '__main__':
    print("Loading dataset parts and saving them as npz")
    for name in DATASET_NAMES:
        print('Loading {} dataset...'.format(name))
        (adjdata, adjrow, adjcol), feature_parts = load_files(name)
        features, labels = assemble_features_labels(feature_parts)
        npzpath = OUTPATH + "/{}.npz".format(name)
        print('Saving to "{}"...'.format(npzpath))
        np.savez_compressed(npzpath,
                            features=features, labels=labels,
                            adjdata=adjdata, adjrow=adjrow, adjcol=adjcol)
    print('All done.')
