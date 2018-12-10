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


def select(selector, items, unique_item=True, unique_output=True):
    """Select one or several values in `items` that match `selector`.

    (This is just a way of avoiding nested dicts.)

    `items` is a list of tuples, where each tuple contains parameters and
    a value resulting from those parameters. For instance, items is a list of
    values like `(scenario_name, overlap_size, sampling_id, training_history)`.
    In this case we want to be able to get the `training_history` corresponding
    to a combination of `(scenario_name, overalp_size, sampling_id)`, which
    will be the selector. We do that by calling `select(selector, items)`.

    Parameters
    ----------
    selector : tuple
        Tuple according to which we select.
    items : list of tuples
        List of items in which we select. This function returns the item(s) for which
        `item[:len(selector)] == selector`, truncated to what's left after removing the
        `selector` part of the item.
    unique_item : bool, optional
        If False:
        * If `selector` selects a unique item, return a list containing only that item.
        * If `selector` selects several items, return the list of those items.
        If True:
        * If `selector` selects a unique item, return that item (not in a list).
        * Raise an error if `selector` does not select a unique item.
        Defaults to True.
    unique_output : bool, optional
        If False, return `item[len(selector):]` (or a list thereof), which is a sequence;
        in particular if `selector` leaves more than one value to its right in the `item` tuple,
        return all those values. If True, return the last value in the item tuple (and not a
        sequence), and raise an error if `selector` doesn't exhaust all but one value in the item
        tuple. Defaults to True.

    Returns
    -------
    With default options, `select((1, 2), [(1, 1, 'a'), (1, 2, 'b')]) == 'b'`, which is what you
    want. `unique_item` and `unique_output` let you return things like `('b',)`, `[`b`]`, and
    `[('b',)]`.

    """

    lim = len(selector)
    filtered = list(filter(lambda item: item[:lim] == selector, items))
    filtered = [item[lim:] for item in filtered]
    if unique_output:
        for item in filtered:
            assert len(item) == 1, "Not a single output"
        filtered = [item[0] for item in filtered]
    if unique_item:
        assert len(filtered) == 1, "Not a single item"
        return filtered[0]
    else:
        return filtered


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
    # Don't turn x into nans if there are rows full of zeros
    zeros = np.where(x_norm == 0)
    x_norm[zeros] = 1
    return x / x_norm


# TOTEST
def softmax(x, axes=(-1,)):
    assert isinstance(axes, (int, tuple))
    if not isinstance(x, (np.ndarray, tf.Tensor)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        backend = np
    else:
        # x is a tf.Tensor
        backend = K
    if len(x.shape) == 0:
        raise ValueError("Cannot perform softmax on 0D tensor/array")
    e = backend.exp(x - backend.max(x, axis=axes, keepdims=True))
    s = backend.sum(e, axis=axes, keepdims=True)
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
