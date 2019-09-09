import os
import argparse

import networkx as nx
import pandas as pd
import numpy as np
from scipy import sparse

import stellargraph as sg
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import GraphSAGENodeGenerator

import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


## Parse args
parser = argparse.ArgumentParser(description='Run GraphSAGE node classification benchmark.')
parser.add_argument('--dataset', type=str, required=True,
                    help="path to a npz file containing adjacency and features of a dataset")
parser.add_argument('--testtype', type=str, required=True,
                    help=("either \"nodes\" for classification, or \"edges\" for link prediction"))
parser.add_argument('--testprop', type=float, required=True,
                    help=("percentage of edges/nodes to add (or add and remove, for edges) to "
                          "training dataset for reconstruction/classification testing"))
parser.add_argument('--diml1enc', type=int, required=True,
                    help="dimension of intermediary encoder layer")
parser.add_argument('--dimxiadj', type=int, required=True,
                    help="embedding dimensions for adjacency")
parser.add_argument('--nepochs', type=int, required=True,
                    help="number of epochs to train for")
parser.add_argument('--bias', type=bool, required=True,
                    help="activate/deactivate bias in GraphSAGE")
parser.add_argument('--nworkers', type=int, required=True,
                    help="number of parallel workers to use")

args = parser.parse_args()

## Set up Julia

# This can be simplified once https://github.com/JuliaPy/pyjulia/issues/310 is fixed
with open("/etc/hostname") as f:
    hostname = f.readline().strip()
from julia.api import LibJulia
api = LibJulia.load()
api.sysimage = "julia/sys-{}.so".format(hostname)
api.init_julia()
from julia import Main
Main.eval('include("julia/dataset.jl")')


## Settings
testtype = args.testtype
assert testtype in ['edges', 'nodes']
testprop = args.testprop
bias = args.bias
layer_sizes = [args.diml1enc, args.dimxiadj]
nworkers = args.nworkers

number_of_walks = 1
length_of_walks = 5

batch_size = 50
nepochs = args.nepochs
num_samples = [10, 5]

verbose = 0

## Load the dataset

# Load npz file
dataset = np.load(args.dataset)
# Extract adjacency and graph
adj = sparse.coo_matrix((dataset['adjdata'], (dataset['adjrow'], dataset['adjcol'])))
Gnx = nx.from_scipy_sparse_matrix(adj, edge_attribute='label')
nx.set_edge_attributes(Gnx, 'cites', 'label')
nx.set_node_attributes(Gnx, "paper", "label")
# Extract features
node_features = pd.DataFrame(dataset['features'])


## Create a test set
def to_julia_edgelist(g):
    return 1 + nx.to_pandas_edgelist(Gnx)[['source', 'target']].values

def from_julia_edgelist(edgelist):
    pd_edgelist = pd.DataFrame(np.array(edgelist) - 1, columns=['source', 'target'])
    pd_edgelist['label'] = 'cites'
    g = nx.from_pandas_edgelist(pd_edgelist, edge_attr="label")
    nx.set_node_attributes(g, "paper", "label")
    return g

if testtype == 'nodes':
    gtrain_edgelist, nodes_test, nodes_train = Main.Dataset.make_nodes_test_set(to_julia_edgelist(Gnx), testprop)
    nodes_test = nodes_test - 1
    nodes_train = nodes_train - 1
    Gtrain_nx = from_julia_edgelist(gtrain_edgelist)
else:
    assert testtype == 'edges'
    gtrain_edgelist, edges_test_true, edges_test_false = Main.Dataset.make_edges_test_set(to_julia_edgelist(Gnx), testprop)
    edges_test_true = edges_test_true - 1
    edges_test_false = edges_test_false - 1
    Gtrain_nx = from_julia_edgelist(gtrain_edgelist)
    # Recover nodes that are now isolated in Gtrain_nx, not seen through the edgelist
    for n in Gnx.nodes():
        if n not in Gtrain_nx.nodes():
            Gtrain_nx.add_node(n)
    nx.set_node_attributes(Gtrain_nx, "paper", "label")


## Train the embedding
#                    mo"number of epochs to train for"l
G = sg.StellarGraph(Gnx, node_features=node_features)
Gtrain = sg.StellarGraph(Gtrain_nx, node_features=node_features)

# The graph G
#                    together wi"number of parallel workers to use" the unsupervised sampler will be used to generate samples.
actual_nodes_train = list(Gtrain.nodes())
if testtype == 'nodes':
    assert set(nodes_train).issuperset(actual_nodes_train)
unsupervised_samples = UnsupervisedSampler(Gtrain, nodes=actual_nodes_train, length=length_of_walks, number_of_walks=number_of_walks)
train_gen = GraphSAGELinkGenerator(Gtrain, batch_size, num_samples).flow(unsupervised_samples)

# Build the model
assert len(layer_sizes) == len(num_samples)
graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=train_gen, bias=bias, dropout=0.0, normalize="l2"
    )
x_inp, x_out = graphsage.build(flatten_output=False)
prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method='ip'
    )(x_out)
model = keras.Model(inputs=x_inp, outputs=prediction)
model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

# Train the model
history = model.fit_generator(
        train_gen,
        epochs=nepochs,
        verbose=verbose,
        use_multiprocessing=False,
        workers=nworkers,
        shuffle=True,
    )


## Get embeddings for all nodes

# Build a new node-based model
x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

# The node generator feeds graph nodes to `embedding_model`. We want to evaluate node embeddings for all nodes in the graph:
node_ids = sorted(G.nodes)
node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)
emb = embedding_model.predict_generator(node_gen, workers=nworkers, verbose=verbose)
node_embeddings = emb[:, 0, :]


if testtype == 'nodes':
    ## Node classification
    X = node_embeddings
    y = np.where(dataset['labels'])[1]

    # Train a Logistic Regression classifier on the training data.
    X_train, X_test, y_train, y_test = X[nodes_train, :], X[nodes_test, :], y[nodes_train], y[nodes_test]
    clf = LogisticRegression(verbose=verbose, solver='liblinear', multi_class="ovr")
    clf.fit(X_train, y_train)

    # Predict the hold out test set.
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier on the test set.
    print('f1macro={}'.format(f1_score(y_test, y_pred, average='macro')))
    print('f1micro={}'.format(f1_score(y_test, y_pred, average='micro')))
else:
    ## Link prediction
    assert testtype == 'edges'

    # Prepare test and real arrays
    edges_test_all = np.concatenate([edges_test_true, edges_test_false])
    edges_real_all = np.concatenate([np.ones(edges_test_true.shape[0]), np.zeros(edges_test_false.shape[0])])

    # Get predictions
    test_gen = GraphSAGELinkGenerator(Gtrain, batch_size, num_samples).flow(edges_test_all)
    edges_pred_all = model.predict_generator(test_gen)[:,0]

    # Print out metrics
    print('roc_auc={}'.format(roc_auc_score(edges_real_all, edges_pred_all)))
    print('ap={}'.format(average_precision_score(edges_real_all, edges_pred_all)))
