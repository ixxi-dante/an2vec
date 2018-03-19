import tempfile
import logging

from keras import backend as K
import jwalk
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


# TOTEST
def right_squeeze2(array):
    """TODOC"""
    return K.squeeze(K.squeeze(array, -1), -1)


# TOTEST
def expand_dims_tile(array, dim_position, multiple):
    """TODOC"""

    # Since expand_dims counts `-1` as "extend to the right",
    # `dim_position` is in fact the position of the new dimension *in
    # the new shape list*, not in the old. So it's important we expand
    # the dimensions before indexing into the list of multiples with
    # `dim_position`.
    if isinstance(array, tf.Tensor):
        backend = K
    else:
        backend = np
    array = backend.expand_dims(array, dim_position)
    multiples = np.ones(len(array.shape), dtype=np.int32)
    multiples[dim_position] = multiple
    return backend.tile(array, multiples)


def node2vec(infile, outfile,
             num_walks=10, embedding_size=128, window_size=10,
             walk_length=80, delimiter=None, model_path=None, stats=False,
             has_header=False, workers=8, undirected=False):

    if infile.lower().endswith('.npz'):  # load graph file instead of edges
        logger.debug("Detected npz extension. Assuming input is CSR matrix.")
        logger.info("Loading graph from %s", infile)
        graph, labels = jwalk.load_graph(infile)
    else:
        logger.info("Loading edges from %s", infile)
        edges = jwalk.load_edges(infile, delimiter, has_header)
        logger.debug("Loaded edges of shape %s", edges.shape)

        logger.info("Building adjacency matrix")
        graph, labels = jwalk.build_adjacency_matrix(edges, undirected)
        logger.debug("Number of unique nodes: %d", len(labels))

        graph_path = infile + '.npz'
        logger.info("Saving graph to %s", graph_path)
        jwalk.save_graph(graph_path, graph, labels)

    logger.info("Doing %d random walks of length %d", num_walks, walk_length)
    random_walks, word_freq = jwalk.walk_graph(graph, labels, walk_length,
                                               num_walks, workers)
    logger.debug("Walks shape: %s", random_walks.shape)

    if stats:
        import pandas as pd
        df = pd.DataFrame(random_walks)
        unique_nodes_in_path = df.apply(lambda x: x.nunique(), axis=1)
        logger.info("Unique nodes per walk description: \n" +
                    unique_nodes_in_path.describe().__repr__())

    logger.info("Building corpus from walks")
    with tempfile.NamedTemporaryFile(delete=False) as f_corpus:
        jwalk.build_corpus(random_walks, outpath=f_corpus.name)

        logger.info("Running Word2Vec on corpus")
        corpus_count = len(labels) * num_walks
        model = jwalk.train_model(f_corpus.name, embedding_size, window_size,
                                  workers=workers, model_path=model_path,
                                  word_freq=word_freq,
                                  corpus_count=corpus_count)
        model.save(outfile)
        logger.info("Model saved: %s", outfile)

    return outfile
