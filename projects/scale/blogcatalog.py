import csv
from collections import defaultdict
import os
import pickle

import numpy as np
import networkx as nx

import keras
from keras_tqdm import TQDMCallback

from nw2vec import ae
from nw2vec import utils
from nw2vec import batching
import settings


# ### PARAMETERS ###

# Data
crop = None

# Model
n_ξ_samples = 1
dim_l1, dim_ξ = 10, 10
use_bias = False

# Training
loss_weights = [1.0, 1.0, 1.0]  # q, p_adj, p_v
n_epochs = 10000
# seeds_per_batch = len(nodes) -> defined below
max_walk_length = 1
p = 1
q = 1
neighbour_samples = 30


# ### MISC. SETUP VARIABLES ###

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    raise ValueError('CUDA_VISIBLE_DEVICES not set')
MODEL_NAME = os.path.split(__file__)[-1][:-3]
MODEL_PATH = os.path.join(settings.SCALE_PATH, MODEL_NAME)
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)


# ### LOAD DATA ###

# ## Get the full list of nodes and groups ##

# nodes
nodes = []
with open('datasets/BlogCatalog-dataset/data/nodes.csv') as csvfile:
    reader = csv.reader(csvfile)
    nodes = [int(row[0]) for row in reader]
if crop is not None:
    nodes = nodes[:crop]
assert len(nodes) == len(set(nodes))
nodes = set(nodes)

# groups
groups = []
with open('datasets/BlogCatalog-dataset/data/groups.csv') as csvfile:
    reader = csv.reader(csvfile)
    groups = [int(row[0]) for row in reader]
assert len(groups) == len(set(groups))
groups = set(groups)


# ## Generate graph from edges and node data ##

# Read edges.csv and make a network out of it
edges = defaultdict(list)
with open('datasets/BlogCatalog-dataset/data/edges.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if crop is not None:
            if int(row[0]) in nodes and int(row[1]) in nodes:
                edges[int(row[0])].append(int(row[1]))
        else:
            edges[int(row[0])].append(int(row[1]))

g = nx.from_dict_of_lists(edges, create_using=nx.Graph())
if crop is not None:
    g.add_nodes_from(nodes)

# Read group-edges.csv and add that info to each node
group_edges = defaultdict(list)
with open('datasets/BlogCatalog-dataset/data/group-edges.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if crop is not None:
            if int(row[0]) in nodes:
                group_edges[int(row[0])].append(int(row[1]))
        else:
            group_edges[int(row[0])].append(int(row[1]))

for node, data in g.nodes.items():
    data['groups'] = group_edges[node]


# ## Sanity checks ##

assert set(g.nodes) == nodes
if crop is None:
    assert set().union(*[groups for _, groups in g.nodes(data='groups')]) == \
        groups

# ## Update model parameters ##
seeds_per_batch = len(nodes)
dim_data = len(groups)
dims = (dim_data, dim_l1, dim_ξ)
DATA_PARAMETERS = 'crop={crop}'.format(crop=crop)
VAE_PARAMETERS = (
    'n_ξ_samples={n_ξ_samples}'
    '-dims={dims}'
    '-bias={use_bias}').format(n_ξ_samples=n_ξ_samples,
                               dims=dims, use_bias=use_bias)
TRAINING_PARAMETERS = (
    'loss_weights={loss_weights}'
    '-seeds_per_batch={seeds_per_batch}'
    '-WL={max_walk_length}'
    '-p={p}'
    '-q={q}'
    '-neighbour_samples={neighbour_samples}'
    '-n_epochs={n_epochs}').format(loss_weights=loss_weights,
                                   seeds_per_batch=seeds_per_batch,
                                   max_walk_length=max_walk_length,
                                   p=p, q=q,
                                   neighbour_samples=neighbour_samples,
                                   n_epochs=n_epochs)
MODEL_DATA = os.path.join(MODEL_PATH,
                          DATA_PARAMETERS + '---' +
                          VAE_PARAMETERS + '---' +
                          TRAINING_PARAMETERS)


# ### DEFINE TRAINING LABELS ###

labels = np.zeros((len(nodes), len(groups)))
nodes_offset = min(nodes)
groups_offset = min(groups)
for n, data in g.nodes.items():
    labels[n - nodes_offset, np.array(data['groups']) - groups_offset] = 1
# labels += np.random.normal(scale=.2, size=labels.shape)

features = utils.scale_center(labels)


# ### BUILD THE VAE ###

n_nodes = len(nodes)
adj = nx.adjacency_matrix(g).astype(np.float32)
q_model, q_codecs = ae.build_q(dims, use_bias=use_bias)
p_builder = ae.build_p_builder(dims, use_bias=use_bias)
vae, vae_codecs = ae.build_vae(
    (q_model, q_codecs),
    p_builder,
    n_ξ_samples,
    loss_weights
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
        utils.expand_dims_tile(features[final_nodes], 1, n_ξ_samples),
    ]


# ### TRAIN ###

steps_per_epoch = int(np.ceil(len(labels) / seeds_per_batch))
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
        #ae.ModelBatchCheckpoint(MODEL_DATA + '.batch-checkpoint.epoch={epoch:05d}-batch={batch:05d}-loss={loss:.2f}.h5',
        #                        monitor='loss', period=50,
        #                        #save_best_only=True
        #                        ),
        keras.callbacks.ModelCheckpoint(MODEL_DATA + '.epoch-checkpoint.h5',
                                        monitor='loss',# period=1,
                                        save_best_only=True
                                        ),
        TQDMCallback(),
    ]
)


# ### SAVE HISTORY ###

with open(MODEL_DATA + '.results.pkl', 'wb') as outfile:
    pickle.dump({'history': history.history}, outfile)
