import time
import random
import itertools
import pickle
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import scipy as sp
import scipy.stats
import sklearn.preprocessing
import matplotlib.pyplot as plt
import seaborn as sb
import networkx as nx

import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.utils.vis_utils import model_to_dot

from keras_tqdm import TQDMCallback as TQDMCallback
from tqdm import tqdm
from progressbar import ProgressBar
from IPython.display import SVG, HTML, display

from nw2vec import ae
from nw2vec import utils
from nw2vec import codecs
from nw2vec import layers
from nw2vec import viz
from nw2vec import batching

MODEL_NAME = ('vae_'
              'l={l}_k={k}_p_out={p_out}_p_in={p_in}_'
              'use_bias={use_bias}_'
              'mini_batch_size={mini_batch_size}_mini_batch_walk_length={mini_batch_walk_length}')
MODEL_PATH = 'data/issue-19-mini-batch-validation/{}.pkl'

user_def_bias = input("Please enter boolean for bias (0: no bias, 1: bias)")
assert user_def_bias in ['0', '1']
use_bias = bool(int(user_def_bias))
os.environ["CUDA_VISIBLE_DEVICES"] = user_def_bias

# Parameter grid for the networks
l_grid = [20, 50]  # number of groups
k_grid = [20, 50]  # number of vertices in each group
p_in = .4  # probability of connecting vertices within a group
p_out_grid = np.logspace(start=-5, stop=np.log10(p_in), num=4, endpoint=True)  # probability of connecting vertices across groups

# How many runs per random network
runs = 1

# Parameter grid for mini-batches
mini_batch_size_grid = [10, 50, 100, 200]  # max it out at current network size
mini_batch_walk_length_grid = [10, 50, 100]  # max it out at current mini_batch_size

# Fixed model and training parameters
dim_l1, dim_ξ = 10, 2
n_ξ_samples = 5
n_epochs = 20000

def build_vae(g, l, k, n_ξ_samples, dims, use_bias):
    n_nodes = l * k
    adj = np.array(nx.adjacency_matrix(g).todense().astype(np.float32))

    dim_data, dim_l1, dim_ξ = dims

    # Encoder
    q_model, q_codecs = ae.build_q(dims, use_bias=use_bias)

    # Decoder
    p_input = keras.layers.Input(shape=(dim_ξ,), name='p_input')
    p_layer1 = keras.layers.Dense(dim_l1,
                                  use_bias=use_bias,
                                  activation='relu',
                                  kernel_regularizer='l2',
                                  bias_regularizer='l2',
                                  name='p_layer1')(p_input)
    #p_adj = layers.Bilinear(0,
    #                        use_bias=use_bias,
    #                        #fixed_kernel=np.eye(dim_ξ),
    #                        kernel_regularizer='l2',
    #                        bias_regularizer='l2',
    #                        name='p_adj')([p_layer1, p_layer1])
    p_v_μ_flat = keras.layers.Dense(dim_data,
                                    use_bias=use_bias,
                                    kernel_regularizer='l2',
                                    bias_regularizer='l2',
                                    name='p_v_mu_flat')(p_layer1)
    p_v_logD_flat = keras.layers.Dense(dim_data,
                                       use_bias=use_bias,
                                       kernel_regularizer='l2',
                                       bias_regularizer='l2',
                                       name='p_v_logD_flat')(p_layer1)
    p_v_u_flat = keras.layers.Dense(dim_data,
                                    use_bias=use_bias,
                                    kernel_regularizer='l2',
                                    bias_regularizer='l2',
                                    name='p_v_u_flat')(p_layer1)
    p_v_μlogDu_flat = keras.layers.Concatenate(name='p_v_mulogDu_flat')(
        [p_v_μ_flat, p_v_logD_flat, p_v_u_flat])
    p_model = ae.Model(inputs=p_input,
                       outputs=[
                           #p_adj,
                           p_v_μlogDu_flat
                       ])

    # Actual VAE
    vae, vae_codecs = ae.build_vae(
        (q_model, q_codecs),
        (p_model, (
            #'SigmoidBernoulliAdjacency',
            'Gaussian',
        )),
        n_ξ_samples,
        [
            1.0,  # q loss
            #1.0,  # p adj loss
            1.0  # p v loss
        ],
    )
    
    return q_model, vae, adj

def batches(model, features, adj, batch_size, max_walk_length, neighbour_samples=None):
    features = utils.scale_center(features)
    
    while True:
        for needed_nodes, output_nodes, feeds in batching.epoch_batches(model, adj, batch_size, max_walk_length, neighbour_samples):
            batch_adj = adj[output_nodes, :][:, output_nodes]

            x = features[needed_nodes]
            y = [
                np.zeros(1), # ignored
                #utils.expand_dims_tile(utils.expand_dims_tile(batch_adj + np.eye(batch_adj.shape[0]), 0, n_ξ_samples), 0, 1),
                utils.expand_dims_tile(features[output_nodes], 1, n_ξ_samples)
            ]
            
            yield x, y, feeds

for p_out, l, k in tqdm(itertools.product(p_out_grid, l_grid, k_grid)):
    # Generate labels for the network
    n_nodes = l * k
    labels = np.zeros((n_nodes, l), dtype=np.float32)
    for c in range(l):
        labels[range(c * k, (c + 1) * k), c] = 1

    # Generate the model parameters
    dim_data = l
    dims = (dim_data, dim_l1, dim_ξ)

    for mini_batch_size in mini_batch_size_grid:
        if mini_batch_size > n_nodes:
            # Mini-batch is at most the size of the network
            continue

        for mini_batch_walk_length in mini_batch_walk_length_grid:
            if mini_batch_walk_length > mini_batch_size:
                # Mini-batch walk length is at most the size of the mini-batch itself
                continue
            
            model_name = MODEL_NAME.format(l=l, k=k, p_out=p_out, p_in=p_in,
                                           use_bias=use_bias,
                                           mini_batch_size=mini_batch_size,
                                           mini_batch_walk_length=mini_batch_walk_length)
            # Prepare the outputs
            trainings = []
            embeddings = []
            
            for run in range(runs):
                print(model_name + ' run #{}'.format(run))
                
                # Generate the network, its features, and the model
                g = nx.planted_partition_graph(l, k, p_in, p_out)
                features = labels + np.abs(np.random.normal(loc=0.0, scale=1.5, size=(n_nodes, l))).astype(np.float32)
                q_model, model, adj = build_vae(g, l, k, n_ξ_samples, dims, use_bias)
                
                # Train
                training = model.fit_generator_feed(batches(model, features, adj, mini_batch_size, mini_batch_walk_length, neighbour_samples=None),
                                                    steps_per_epoch=int(np.ceil(len(features) / mini_batch_size)),
                                                    epochs=n_epochs,
                                                    check_array_lengths=False,
                                                    shuffle=False,
                                                    verbose=0,
                                                    callbacks=[TQDMCallback()])
                del training.model
                trainings.append(training)
                
                # Predict
                x, _, feeds = next(batches(q_model, features, adj, n_nodes, n_nodes, None))
                embeddings.append(q_model.predict_on_fed_batch(x, feeds=feeds))
            
            # Save the results for this model
            with open(MODEL_PATH.format(model_name), 'wb') as f:
                pickle.dump({'trainings': trainings, 'embeddings': embeddings}, f)
