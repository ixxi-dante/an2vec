import tempfile
import logging
import itertools

from keras import backend as K
import jwalk
import numpy as np
from scipy import sparse
import tensorflow as tf
import numba


logger = logging.getLogger(__name__)


def slice_size(s, n):
    start, stop, step = s.indices(n)

    if step == 1:
        return stop - start
    else:
        return 1 + (stop - start) // step


@numba.jit(nopython=True)
def alias_setup(probs):
    """Compute utility lists for non-uniform sampling from discrete distributions.

    Requires a real, i.e. normalised, distribution in `probs`.

    Refer to
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details.
    """
    n_choices = len(probs)
    assert n_choices > 0

    mixture_weights = probs.copy() * n_choices
    mixture_choices = np.zeros(n_choices, np.int32)

    small_choices = []
    large_choices = []
    for choice in range(n_choices):
        if mixture_weights[choice] < 1.0:
            small_choices.append(choice)
        else:
            large_choices.append(choice)

    while len(small_choices) > 0 and len(large_choices) > 0:
        small = small_choices.pop()
        large = large_choices.pop()

        mixture_choices[small] = large
        mixture_weights[large] = mixture_weights[large] + mixture_weights[small] - 1.0
        if mixture_weights[large] < 1.0:
            small_choices.append(large)
        else:
            large_choices.append(large)

    return mixture_choices, mixture_weights


@numba.jit(nopython=True)
def alias_draw(mixture_choices, mixture_weights):
    """Draw sample from a non-uniform discrete distribution using alias sampling."""
    assert len(mixture_choices) > 0
    assert len(mixture_choices) == len(mixture_weights)

    pre_choice = np.random.randint(len(mixture_choices))
    if np.random.rand() < mixture_weights[pre_choice]:
        return pre_choice
    else:
        return mixture_choices[pre_choice]


# TOTEST
def csr_to_sparse_tensor_parts(m):
    return _csr_to_sparse_tensor_parts(m.indices, m.indptr, m.data, m.shape)


# TOTEST
def sparse_tensor_parts_to_csr(parts):
    ind, data, shape = parts
    return sparse.csr_matrix((data, (ind[:, 0], ind[:, 1])), shape=shape)


# TOTEST
@numba.jit(nopython=True)
def _csr_to_sparse_tensor_parts(indices, indptr, data, shape):
    n_values = len(data)
    ind = np.zeros((n_values, 2), dtype=np.int32)
    collected = 0
    for i in range(shape[0]):
        cols = indices[indptr[i]:indptr[i + 1]]
        n_cols = len(cols)
        ind[collected:collected + n_cols, 0] = i
        ind[collected:collected + n_cols, 1] = cols
        collected += n_cols
    return (ind, data, shape)


# TOTEST
def inner_repeat(it, n):
    return itertools.chain(*zip(*itertools.tee(it, n)))


# TOTEST
def grouper(iterable, n):
    it = iter(iterable)
    while True:
        chunk = itertools.islice(it, n)
        try:
            first = next(chunk)
        except StopIteration:
            return
        yield itertools.chain((first,), chunk)


# TOTEST
def scale_center(x, norm='l2'):
    assert norm in ['l1', 'l2']
    x = x - x.mean(1, keepdims=True)
    if norm == 'l1':
        x_norm = np.abs(x).sum(1, keepdims=True)
    if norm == 'l2':
        x_norm = np.sqrt((x ** 2).sum(1, keepdims=True))
    return x / x_norm


# TOTEST
def softmax(x, axes=[-1]):
    if len(x.shape) == 0:
        raise ValueError("Cannot perform softmax on 0D tensor")
    e = tf.exp(x - tf.reduce_max(x, axis=axes, keep_dims=True))
    s = tf.reduce_sum(e, axis=axes, keep_dims=True)
    return e / s


# TOTEST
def Softmax(axes=[-1]):
    def axes_softmax(x):
        return softmax(x, axes=axes)
    return axes_softmax


# TOTEST
def broadcast_left(array, target_array):
    """TODOC"""

    with tf.name_scope('broadcast') as scope:
        array = tf.convert_to_tensor(array)
        array_shape = tf.shape(array)
        array_rank = tf.rank(array)
        target_array = tf.convert_to_tensor(target_array)
        target_shape = tf.shape(target_array)
        target_rank = tf.rank(target_array)

        with tf.control_dependencies(
                [tf.assert_rank_at_least(target_array, array_rank),
                 tf.assert_equal(array_shape, target_shape[- array_rank:])]):
            # Actually reshape and broadcast
            degen_shape = tf.concat([tf.ones(target_rank - array_rank, dtype=tf.int32),
                                     array_shape],
                                    0)
            multiples = tf.concat([target_shape[:- array_rank],
                                   tf.ones(array_rank, dtype=tf.int32)],
                                  0)
            out = tf.tile(tf.reshape(array, degen_shape), multiples, name=scope)

            # Inform the static shape
            out.set_shape(target_array.shape)

            return out


# TOTEST
def get_backend(array):
    return tf if isinstance(array, (tf.Tensor, tf.Variable)) else np


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
    backend = get_backend(array)
    array = backend.expand_dims(array, dim_position)
    multiples = backend.ones(len(array.shape), dtype=backend.int32)
    concat = backend.concatenate if backend == np else backend.concat
    multiples = concat([multiples[:dim_position], [multiple],
                        multiples[dim_position + 1:]],
                       0)
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
