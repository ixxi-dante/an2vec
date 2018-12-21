# Explore the effect of overlap size in slicing embeddings for scenarios having different network-feature correlation

##
## Imports and setup
##

import time
import random
import os
from collections import defaultdict, Counter
import warnings
import functools
import pickle
import gc
import datetime

import numpy as np
import scipy as sp
import scipy.stats
import sklearn.preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sb
import pandas as pd
import networkx as nx
import dask
import distributed

import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.utils.vis_utils import model_to_dot
from keras_tqdm import TQDMCallback as TQDMCallback
from tqdm import tqdm as tqdm

from progressbar import ProgressBar
from IPython.display import SVG, HTML, display

from nw2vec import ae
from nw2vec import utils
from nw2vec import codecs
from nw2vec import layers
from nw2vec import viz
from nw2vec import batching
from nw2vec import generative
import settings

client = distributed.Client('localhost:8786')
client

N_ALPHAS = 11
N_MODEL_SAMPLES = 20
N_CLUSTERINGS = 2
N_NODES = 1000
N_CLUSTERS = 100
SPILL_V2ADJ = int(os.environ['SPILL_V2ADJ'])
OVERWRITE_RESULTS = True
COLORS_PATH = os.path.join(settings.BEHAVIOUR_PATH, 'colors')
RESULTS_PATH = (COLORS_PATH
                + '/S2_S3-ov_noov_asym-n_nodes={n_nodes}-n_clusters={n_clusters}'
                + '-n_alphas={n_alphas}-n_models={n_models}-n_clusterings={n_clusterings}'
                + '-spill_v2adj={spill_v2adj}'
                + '-{data_name}.pkl')

##
## Creating the scenarios
##

α_range = np.linspace(start=0, stop=1, num=N_ALPHAS, endpoint=True)
π_gap_range = np.linspace(start=.075, stop=.12, num=N_CLUSTERINGS, endpoint=True)

def scenario_from_aid_cid(aid, cid, n_nodes=N_NODES, n_clusters=N_CLUSTERS):
    α = α_range[aid]
    ρ = np.ones(n_clusters) / n_clusters
    π_gap = π_gap_range[cid]
    π = ((.13 + π_gap) * np.diag(np.ones(n_clusters))
         + (.13 - π_gap) * (np.ones((n_clusters, n_clusters)) - np.diag(np.ones(n_clusters))))
    Y, A, labels = generative.colors(n_nodes, ρ, π, α)
    features = labels + np.random.normal(scale=.1, size=labels.shape)
    return Y, scipy.sparse.csr_matrix(A), labels, features

##
## Build the VAEs
##

# The rest of the notebook generates VAE models for the different scenarios and different overlaps. This section defines the model-generating function.

# Define the last VAE Parameters
n_ξ_samples = 5  # Number of embeddings sampled at the embedding layer.

def get_loss_weights(n_nodes, dims, q_overlap):
    dim_data, _, dim_ξ_adj, dim_ξ_v = dims
    return {
        # embedding-gaussian divergence scales with number of embedding dimensions,
        # but we also don't want it to overpower the other losses, hence the 1e-2
        'q_mulogS_flat': 1000 * 1e-3 * 1.0 / (dim_ξ_adj - q_overlap + dim_ξ_v),
        # Adj loss scales with the number of nodes
        'p_adj': 1000 * 1.0 / (n_nodes * np.log(2)),
        # Feature loss scales with the average number of sampled words
        'p_v': 1000 * 1.0 / np.log(dim_data),
    }

def make_vae(n_nodes, dims, q_overlap, p_ξ_slices):
    """Build a VAE with features of dimension `dims`,
    and an overlap size of `q_overlap` between adj and feature embeddings,
    and p_ξ_slices distributing embedding outputs to adj/feature tasks.

    Returns
    -------
    q_model : nw2vec.ae.Model
        The encoder model.
    q_codecs : list of strings
        The list of codec names (corresponding to subclasses of nw2vec.codecs.Codec)
        used to interpret the output of `q_model`. Currently this list has always a single item.
    vae : nw2vec.ae.Model
        The full VAE, which includes `q_model`.
    vae_codecs : list of strings
        The list of codec names used to interpret the output of `vae`. The output of `vae` is made of
        the output of `q_model` (so that the first codec name her is always the single codec in `q_codecs`),
        the adjacency reconstruction, and the feature reconstruction.

    In the original model, the adj and feature outputs of `vae` would be parameters to distributions
    from which you sample to create a prediction. For the adjacency, they would be numbers between 0 and 1
    (parameters to a Bernoulli distribution for each value in the matrix). The same goes for binary features.
    For non-binary features, we use a Gaussian distribution for feature prediction, so the output for features
    would be the μ values and values of the diagonal of log(Σ) (just like the parametrisation of the embeddings).
    But! The last operation on these layers would be a sigmoid, and the loss we compute downstream
    is a cross-entropy loss, and the combination of these two is not numerically stable (it easily vanishes to 0).

    So instead, we push the sigmoid operation into the loss computation (implemented in the codec object,
    where sigmoid and cross-entropy are combined into a numerically stable operation), such that what
    the `vae` outputs for the adj and feature predictions is the values that should go into a sigmoid.
    (This is why some codecs are named 'Sigmoid*'.)

    Long story short: if you want to plot predictions, you must:
    1) put the `vae` output values for adj and features into a sigmoid
    2) sample from those values according to what the parameters represent
       (i.e. Bernoulli or parametrised Gaussian).

    The plot_predictions() function above does that for you.

    """

    dim_data, dim_l1, dim_ξ_adj, dim_ξ_v = dims

    # Build the encoder model
    q_model, q_codecs = ae.build_q(
        dims,
        # Size of the overlap between the separated adj and feature layers
        # which are combined (with overlap) to construct the embedding
        overlap=q_overlap,
        # Function used to generate fullbatches,
        # used behind the scenes by Model.predict_fullbatch() and Model.fit_fullbatches()
        fullbatcher=batching.fullbatches,
        # Function used to generate minibatches,
        # used behind the scenes by Model.predict_minibatches() and Model.fit_minibatches()
        minibatcher=batching.pq_batches)

    # Build a function which, given an input (which would be the embeddings, i.e. the output of the
    # encoder model), will produce the output of the decoder.
    p_builder = ae.build_p_builder(
        dims,
        # The input features are not binary, but between 0 and 1 summing to 1,
        # so we model them as Multinomial variables.
        # The output of the model for the features is therefore passed to the nw2vec.codecs.SoftmaxMultinomial
        # codec, which knows how to compute the loss values.
        feature_codec='SoftmaxMultinomial',
        # Slices used to separate what part of the embeddings is used for adjacency prediction from what part
        # is used for feature prediction. After sampling of embedding values (using the parameters output by
        # the encoder), the first item in this list defines the input to adjacency decoding, and the second
        # item defines the input to feature decoding. In this case they overlap with `overlap` dimensions.
        embedding_slices=p_ξ_slices,
        # Whether or not to use an intermediate layer in decoding.
        with_l1=True,
        # If `with_l1=True`, this defines whether the adjacency flow and the feature flow share the same weights
        # for their intermediate decoding layer.
        share_l1=False)

    # Build the actual VAE. This takes the encoder, adds a sampling layer, puts that into the decoder,
    # and compiles the resulting model.
    vae, vae_codecs = ae.build_vae(
        # Encoder and decoder parameters used to create the VAE
        (q_model, q_codecs), p_builder,
        # Number of samples to generate at the embedding layer.
        n_ξ_samples,
        # Weights used in computing the total final loss of the model. The three values are applied to
        # 1) the divergence between embeddings and a standard centred Gaussian
        # 2) the ajd reconstruction loss
        # 3) the feature reconstruction loss
        loss_weights=get_loss_weights(n_nodes, dims, q_overlap)
    )

    return q_model, q_codecs, vae, vae_codecs

# A few notes about the above:
# * Decoding seems to be much better with an intermediate dense layer (i.e. `with_l1=True`) before the final dense layer for feature reconstruction and the bilinear layer for adjacency reconstruction
# * The weights for this layer should NOT be shared between adj and feature tracks (i.e. `shared_l1=False`), since this would re-create a dependency between the separated parts of the embeddings.

##
## Train
##

# This section generates VAE models with different overlap values for each of the STBM scenarios, and trains them on the distributed Dask cluster. It recovers the history of the losses during training, and the weights of the trained models.

# We start by defining the training parameters, and some helpers for the loops further down.

# Maximum number of epochs to train for
n_epochs = 1000

# `target_func` is given to the training function (in fact it's given as a parameter to the fullbatcher,
# through an argument to the training function) and generates the values against which training loss is computed
# for each batch. The complexity of this comes from the fact that for minibatches, the targets change at each
# minibatch (which is why we need a function to create the targets for each minibatch). In the fullbatch case
# the targets are always the same, but the way it's coded is unchanged so we're left with a bit of useless
# complexity here.
# The target values also depend on the STBM scenario used, so here we make a function which
# creates a `target_func` given the STBM node labels.
def make_target_func(labels):

    def target_func(batch_adj, required_nodes, final_nodes):
        """Create training targets given the batch's adjacency matrix, the subset of nodes it used as input,
        and the set of nodes for which output values are computed.

        In the minibatch setting, `required_nodes` is usually the 2-hop neighbourhood of `final_nodes`.
        In the fullbatch setting all this is constant and equal to the values of the full dataset.

        """

        # Keras requires a target value for each output of the model trained,
        # so we have one entry per output/loss in the model. Each entry must also have
        # a shape which is compatible with the corresponding output, which is why we have
        # the `utils.expand_dims_tile()` calls: the target value is the same for each of the
        # values generated from the `n_ξ_samples` embeddings.
        return [
            # Keras feeds this to the embedding-gaussian divergence loss function,
            # but we ignore this input in the actual loss function.
            np.zeros(1), # ignored
            # Target value for adjacency reconstruction
            utils.expand_dims_tile(utils.expand_dims_tile(batch_adj + np.eye(batch_adj.shape[0]),
                                                          0, n_ξ_samples),
                                   0, 1),
            # Target value for feature reconstruction
            utils.expand_dims_tile(labels[final_nodes], 1, n_ξ_samples),
        ]

    return target_func

def train(model, A, labels, features, with_progress=True):
    """Train `model` using adjacency matrix `A` and `labels` as input.

    `with_progress` controls whether or not to show a TQDM progress bar.

    Note that the fullbatcher takes care of normalising and centring the input labels,
    so `labels` can be the verbatim values from the STBM scenario.

    """

    # Show a progressbar if asked to.
    callbacks = []
    if with_progress:
        callbacks.append(TQDMCallback(show_inner=False, leave_outer=False))

    # Ignore warnings here: TensorFlow always complains that we convert a sparse matrix to a dense matrix,
    # which might use memory. I know, it's an open issue on GitHub, but it's not a problem for now.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit_fullbatches(
            # kwargs sent to the fullbatch generator. See `nw2vec.batching.fullbatches()` to see
            # what happens to these.
            batcher_kws={'adj': A, 'features': features, 'target_func': make_target_func(labels)},
            # Self-explanatory.
            epochs=n_epochs, verbose=0, callbacks=callbacks
        )

def make_train_vae(dims, q_overlap, p_ξ_slices, skey, with_progress=True):
    """Build a VAE, train it, and return the training history and trained weights.

    Returns
    -------
    history : dict of (string, ndarray)
        Dict associating loss name to the history of that loss value during model training
        Includes total loss and components that make up the total.

    """

    scenario = scenario_from_aid_cid(*skey)
    _, A, labels, features = scenario
    # This function runs in Dask workers, i.e. in another process on another machine. We need to make sure
    # there is a different session for each worker (which is where tensor values are stored),
    # and setting the default TF session does just that.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config).as_default():
        # Create the model given our parameters.
        q_model, _, vae, _ = make_vae(A.shape[0], dims, q_overlap, p_ξ_slices)
        # Actually train the model.
        history = train(vae, A, labels, features, with_progress=with_progress)
        # Massage the output history and return both history and model weights. Both these values
        # are picklable, so travel without problem across processes and across the network
        # (i.e. across Dask workers).
        # Note that keras models can't be sent easily across the network, as pickling them
        # requires many hacks. Instead we just transport the weights, and recreate any model
        # we want to inspect on the receiver end, i.e. in this notebook process.
        history = dict((title, np.array(values)) for title, values in history.history.items())
        return history

##
## Actual training
##

# We're now in a position to generate a range of models (different overlaps for each scenario) and train them on the Dask cluster.

# Where our results will go.
submitted = {}

max_dim_ξ = 20
assert max_dim_ξ % 2 == 0
dim_l1 = max(max_dim_ξ, int(np.round(np.exp(np.mean(np.log([max_dim_ξ, N_CLUSTERS]))))))

MODEL_OVERLAP = 'overlap'
MODEL_NOOVERLAP = 'no-overlap'

def submit_training(**kws):
    dims = (N_CLUSTERS, kws['dim_l1'], kws['dim_ξ_adj'], kws['dim_ξ_v'])

    scenario_key = (kws['aid'], kws['cid'])
    model_key = (kws['model_type'], kws['spill_v2adj'], kws['ov'], kws['sampling_id'])

    history = client.submit(
        make_train_vae, dims, kws['q_overlap'], kws['p_ξ_slices'], scenario_key,
        # Don't show progress bars for these tasks, there are hundreds of them.
        with_progress=False,
        # They key uniquely identifies a task in Dask, so it's important we include the sampling id here,
        # or Dask would just assume the different samples are the same task.
        key=('make_train_vae', id(make_train_vae),  # function definition
             scenario_key,  # scenario definition
             model_key)  # model definition
    )
    return (scenario_key, model_key, history)

# Sample several points for each (scenario, overlap).
for sampling_id in tqdm(range(N_MODEL_SAMPLES)):
    # Loop over scenarios
    for cid in [1]:#tqdm(range(N_CLUSTERINGS)):
        for aid in tqdm(range(N_ALPHAS)):
            dim_ξ_ov_adj = max_dim_ξ // 2 + SPILL_V2ADJ
            dim_ξ_ov_v = max_dim_ξ // 2 - SPILL_V2ADJ
            # Loop over all overlap values: 0 up to maximum overlap by steps of 2
            for ov in tqdm(range(0, min(dim_ξ_ov_v, dim_ξ_ov_adj) + 1, 2)):
            #for dim_ξ in tqdm(range(max_dim_ξ, max_dim_ξ // 2 - 1, -2)):
                for model_type in tqdm([MODEL_OVERLAP, MODEL_NOOVERLAP]):
                    if model_type == MODEL_OVERLAP:
                        dim_ξ_adj = dim_ξ_ov_adj
                        dim_ξ_v = dim_ξ_ov_v
                        q_overlap = ov
                        p_ξ_slices = [slice(0, dim_ξ_adj),
                                      slice(dim_ξ_adj - q_overlap, dim_ξ_adj - q_overlap + dim_ξ_v)]
                    else:
                        assert model_type == MODEL_NOOVERLAP
                        dim_ξ_adj = dim_ξ_ov_adj - ov // 2
                        dim_ξ_v = dim_ξ_ov_v - ov // 2
                        q_overlap = 0
                        p_ξ_slices = [slice(0, dim_ξ_adj),
                                      slice(dim_ξ_adj, dim_ξ_adj + dim_ξ_v)]

                    skey, mkey, history = submit_training(**{
                        'aid': aid, 'cid': cid,
                        'dim_l1': dim_l1, 'dim_ξ_adj': dim_ξ_adj, 'dim_ξ_v': dim_ξ_v,
                        'model_type': model_type, 'spill_v2adj': SPILL_V2ADJ,
                        'ov': ov, 'sampling_id': sampling_id,
                        'q_overlap': q_overlap, 'p_ξ_slices': p_ξ_slices,
                    })
                    submitted[(skey, mkey)] = history

##
## Gather successful tasks and save to disk
##

gathered = {}

def get_statuses():
    return Counter(map(lambda t: t.status, submitted.values()))

statuses = get_statuses()

while statuses['pending'] + statuses['lost'] + statuses['finished'] > 0:
    newly_gathered = 0
    for (skey, mkey), history in submitted.items():#tqdm(submitted.items()):
        if (skey, mkey) not in gathered and history.status == 'finished':
            newly_gathered += 1
            gathered[(skey, mkey)] = history.result()
    print("{}: gathered newly finished tasks (+{}, total {})".format(datetime.datetime.utcnow(), newly_gathered, len(gathered)))

    for (skey, mkey) in gathered.keys():
        submitted.pop((skey, mkey), None)
    gc.collect();

    if OVERWRITE_RESULTS:
        with open(RESULTS_PATH.format(n_nodes=N_NODES, n_clusters=N_CLUSTERS,
                                      n_alphas=N_ALPHAS, n_models=N_MODEL_SAMPLES, n_clusterings=N_CLUSTERINGS,
                                      spill_v2adj=SPILL_V2ADJ,
                                      data_name='histories'),
                  'wb') as f:
            pickle.dump(gathered, f)
        print("{}: saved gathered tasks".format(datetime.datetime.utcnow()))

    time.sleep(300)
    statuses = get_statuses()
    print("{}: statuses: {}".format(datetime.datetime.utcnow(), statuses))

print("All done")
