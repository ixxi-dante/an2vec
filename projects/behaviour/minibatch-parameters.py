import os
import pickle
import warnings

import numpy as np
import networkx as nx
from tqdm import tqdm

import keras
from keras_tqdm import TQDMCallback

from nw2vec import ae
from nw2vec import utils
from nw2vec import batching
import settings


# ### PARAMETERS ###

# Data
n_communities = 20
community_size = 100
p_in = .4
p_out = .01
features_noise_scale = 0

# Model
n_ξ_samples = 5
dim_l1, dim_ξ = 10, 2
use_bias = False

# Training
n_runs = 4
n_epochs = 1000
seeds_per_batch = 10
grid_max_walk_length = [int(community_size * .2), int(community_size * .5),
                        int(community_size * .8), community_size,
                        int(community_size * 1.5), int(community_size * 2),
                        int(community_size * 3)]
grid_pq = [(1,   1),    # Non-biased
           (100, 100),  # Walk triangles
           (1,   .01),  # Walk out
           (.01, 1)]    # Walk back
neighbour_samples = 30


# ### MISC. SETUP VARIABLES ###

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    raise ValueError('CUDA_VISIBLE_DEVICES not set')
MODEL_NAME = os.path.split(__file__)[-1][:-3]
MODEL_PATH = os.path.join(settings.BEHAVIOUR_PATH, MODEL_NAME)
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)


def train(max_walk_length, p, q, run):
    # ### DEFINE TRAINING DATA ###

    g = nx.planted_partition_graph(n_communities, community_size, p_in, p_out)
    labels = np.zeros((n_communities * community_size, n_communities), dtype=np.float32)
    for c in range(n_communities):
        labels[range(c * community_size, (c + 1) * community_size), c] = 1
    # features = (np.random.random((n_communities * community_size, n_communities))
    #             .astype(np.float32))
    features = (labels + np.abs(np.random.normal(loc=0.0,
                                                 scale=features_noise_scale,
                                                 size=(n_communities * community_size,
                                                       n_communities))
                                .astype(np.float32)))

    # ## Update model parameters ##
    dim_data = n_communities
    dims = (dim_data, dim_l1, dim_ξ)
    DATA_PARAMETERS = (
        'n_communities={n_communities}'
        '-community_size={community_size}'
        '-p_in={p_in}'
        '-p_out={p_out}'
        '-features_noise_scale={features_noise_scale}').format(n_communities=n_communities,
                                                               community_size=community_size,
                                                               p_in=p_in, p_out=p_out,
                                                               features_noise_scale=features_noise_scale)
    VAE_PARAMETERS = (
        'n_ξ_samples={n_ξ_samples}'
        '-dims={dims}'
        '-bias={use_bias}').format(n_ξ_samples=n_ξ_samples,
                                   dims=dims, use_bias=use_bias)
    TRAINING_PARAMETERS = (
        'seeds_per_batch={seeds_per_batch}'
        '-WL={max_walk_length}'
        '-p={p}'
        '-q={q}'
        '-neighbour_samples={neighbour_samples}'
        '-n_epochs={n_epochs}'
        '-run={run}').format(seeds_per_batch=seeds_per_batch,
                             max_walk_length=max_walk_length,
                             p=p, q=q,
                             neighbour_samples=neighbour_samples,
                             n_epochs=n_epochs,
                             run=run)
    MODEL_DATA = os.path.join(MODEL_PATH,
                              DATA_PARAMETERS + '---' +
                              VAE_PARAMETERS + '---' +
                              TRAINING_PARAMETERS)
    MODEL_RESULTS = MODEL_DATA + '.results.pkl'
    if os.path.exists(MODEL_RESULTS):
        warnings.warn('"{}" already exist, skipping.'.format(MODEL_RESULTS))
        return

    # ### BUILD THE VAE ###

    adj = nx.adjacency_matrix(g).astype(np.float32)
    q_model, q_codecs = ae.build_q(dims, use_bias=use_bias)
    p_builder = ae.build_p_builder(dims, use_bias=use_bias)
    vae, vae_codecs = ae.build_vae(
        (q_model, q_codecs),
        p_builder,
        n_ξ_samples,
        [
            1.0,  # q loss
            1.0,  # p adj loss
            1.0,  # p v loss
        ],
    )

    # ### DEFINE TRAINING OBJECTIVES ###

    def target_func(batch_adj, required_nodes, final_nodes):
        return [
            np.zeros(1),  # ignored
            utils.expand_dims_tile(
                utils.expand_dims_tile(batch_adj + np.eye(batch_adj.shape[0]),
                                       0, n_ξ_samples),
                0, 1
            ),
            utils.expand_dims_tile(labels[final_nodes], 1, n_ξ_samples),
        ]

    # ### TRAIN ###

    steps_per_epoch = int(np.ceil(len(features) / seeds_per_batch))

    history = vae.fit_generator_feed(
        batching.batches(vae, adj, labels, target_func,
                         seeds_per_batch, max_walk_length,
                         p=p, q=q, neighbour_samples=neighbour_samples),
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        check_array_lengths=False,
        shuffle=False,
        verbose=0,
        callbacks=[
            # keras.callbacks.TensorBoard(),
            ae.ModelBatchCheckpoint(MODEL_DATA + '.batch-checkpoint.h5',
                                    monitor='loss', period=10,
                                    save_best_only=True),
            keras.callbacks.ModelCheckpoint(MODEL_DATA + '.epoch-checkpoint.h5',
                                            monitor='loss', period=10,
                                            save_best_only=True),
            TQDMCallback(),
        ]
    )

    # ### SAVE HISTORY ###
    x, _, feeds = next(batching.batches(vae, adj, features, target_func,
                                        adj.shape[0], 1, p=1, q=1, neighbour_samples=None))
    embeddings, adj_pred, features_pred = vae.predict_on_fed_batch(x, feeds=feeds)
    with open(MODEL_RESULTS, 'wb') as outfile:
        pickle.dump({'history': history.history,
                     'labels': labels,  'features': features, 'adj': adj,
                     'embeddings': embeddings,
                     'adj_pred': adj_pred,
                     'features_pred': features_pred},
                    outfile)


for max_walk_length in tqdm(grid_max_walk_length):
    for (p, q) in tqdm(grid_pq):
        for run in tqdm(range(n_runs)):
            train(max_walk_length, p, q, run)
