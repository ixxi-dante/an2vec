print("Starting")
import itertools
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import scipy as sp
import scipy.stats
import sklearn.preprocessing
import seaborn as sb
import networkx as nx

import keras
from keras import backend as K
from keras_tqdm import TQDMCallback
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.utils.vis_utils import model_to_dot

from nw2vec import ae
from nw2vec import utils
from nw2vec import codecs
from nw2vec import layers
from nw2vec import batching
from decimal import Decimal

print ("Imports Finished")

usr_def_bias = input("Please enter  boolean for bias (0: No bias,1 : Bias)")
use_bias=bool(int(usr_def_bias))
os.environ["CUDA_VISIBLE_DEVICES"]=usr_def_bias

# Parameters Network
l = 10 #nb of groups
k = 10 #nb of vertices in each group
p_in = .4 #probability of connecting vertices within a group
n_nodes = l * k

# Parameters Grid Search
p_out_grid=np.logspace(start=-6,num=8,stop=np.log10(p_in),endpoint=True)
dim_ξ_grid=np.linspace(start=1,stop=l,num=3,dtype=np.int64)
n_ξ_samples_grid=np.array([1,20])
labels = np.zeros((l * k, l), dtype=np.float32)
features = labels + np.abs(np.random.normal(loc=0.0, scale=1.5, size=(l * k, l))).astype(np.float32)
dim_data = l

print ("Parameters Defined")

def batches(fts,gcn_adj,gcn_ξ_samples):
    x = utils.scale_center(fts)
    y = [
        np.zeros(n_nodes),
        utils.expand_dims_tile(gcn_adj + np.eye(gcn_adj.shape[0]), 0, gcn_ξ_samples),
        #utils.scale_center(fts),
    ]
    while True:
        yield (x, y)


def gcn_model(gcn_bias,gcn_p_out,gcn_dim_ξ,gcn_dim_l1,gcn_n_ξ_samples,fts=features,n_epochs = 10000):
    # Model name
    use_bias_name=str(int(gcn_bias))
    p_out_name='%.2E' % Decimal(str(gcn_p_out))
    dim_ξ_name=str(gcn_dim_ξ)
    dim_l1_name=str(gcn_dim_l1)
    n_ξ_samples_name=str(gcn_n_ξ_samples)
    model_name="bias_%s_p_out_%s_dim_ξ_%s_dim_l1_%s_n_ξ_samples_%s"%(use_bias_name,p_out_name,dim_ξ_name,dim_l1_name,n_ξ_samples_name)
    # Network
    g = nx.planted_partition_graph(l, k, p_in, gcn_p_out)
    gcn_adj = nx.adjacency_matrix(g).todense().astype(np.float32)
    dims = (dim_data, gcn_dim_l1, gcn_dim_ξ)
    # Encoder
    q_model, q_codecs = ae.build_q(dims, use_bias=gcn_bias)
    # Decoder
    p_input = keras.layers.Input(shape=(gcn_dim_ξ,), name='p_input')
    p_layer1 = keras.layers.Dense(gcn_dim_l1,use_bias=gcn_bias,activation='relu',
                                  kernel_regularizer='l2',bias_regularizer='l2',name='p_layer1')(p_input)
    p_adj = layers.Bilinear(0,use_bias=gcn_bias,kernel_regularizer='l2',
                            bias_regularizer='l2',name='p_adj')([p_layer1, p_layer1])
    p_v_μ_flat = keras.layers.Dense(dim_data,use_bias=gcn_bias,
                                    kernel_regularizer='l2',bias_regularizer='l2',name='p_v_mu_flat')(p_layer1)
    p_v_logD_flat = keras.layers.Dense(dim_data,use_bias=gcn_bias,kernel_regularizer='l2',
                                       bias_regularizer='l2',name='p_v_logD_flat')(p_layer1)
    p_v_u_flat = keras.layers.Dense(dim_data,use_bias=gcn_bias,kernel_regularizer='l2',
                                    bias_regularizer='l2',name='p_v_u_flat')(p_layer1)
    p_v_μlogDu_flat = keras.layers.Concatenate(name='p_v_mulogDu_flat')([p_v_μ_flat, p_v_logD_flat, p_v_u_flat])
    p_model = ae.Model(inputs=p_input,outputs=[ p_adj,p_v_μlogDu_flat])
    #
    # Actual VAE
    vae, vae_codecs = ae.build_vae(
        (q_model, q_codecs),
        (p_model, ('SigmoidBernoulliAdjacency','Gaussian')),
        n_ξ_samples,
        [1.0,  # q loss
         1.0,  # p adj loss
         1.0  # p v loss
        ],)
    #
    batch_size = len(features)
    vae_hist = vae.fit_generator_feed(batches(vae, features, gcn_adj, batch_size, neighbour_samples=None),
                                      steps_per_epoch=int(np.ceil(len(features) / batch_size)),
                                      epochs=n_epochs,check_array_lengths=False,
                                      shuffle=False,verbose=0,
                                      callbacks=[ TQDMCallback(leave_inner=False,leave_outer=False)])
    x, _, feeds = next(batches(q_model,fts, gcn_adj, len(fts),len(fts), None))
    q_pred = q_model.predict_on_fed_batch(x, feeds=feeds)
    return model_name,vae_hist,q_pred


def batches(model, features, adj, batch_size, neighbour_samples=None):
    features = utils.scale_center(features)
    #
    while True:
        for needed_nodes, output_nodes, feeds in batching.batches(model, adj, batch_size, neighbour_samples):
            batch_adj = adj[output_nodes, :][:, output_nodes]

            x = features[needed_nodes]
            y = [
                np.zeros(1), # ignored
                utils.expand_dims_tile(utils.expand_dims_tile(batch_adj + np.eye(batch_adj.shape[0]), 0, n_ξ_samples), 0, 1),
                utils.expand_dims_tile(features[output_nodes], 1, n_ξ_samples)
            ]
            #
            yield x, y, feeds



print ("Launching")

nb_reps=6
dic_ans={}

for p_out in tqdm(p_out_grid):
    for dim_ξ in dim_ξ_grid:
        for dim_l1 in np.linspace(start=dim_ξ+5,stop=10*dim_ξ,num=2,dtype=np.int64):# If not+ 5, nan in loss
            for n_ξ_samples in n_ξ_samples_grid:
                model_instance,q_pred_instance=[],[]
                for rep in range(nb_reps):
                    model_name,model_hist,q_model_pred=gcn_model(gcn_bias=use_bias,
                                                                 gcn_p_out=p_out,
                                                                 gcn_dim_ξ=dim_ξ,
                                                                 gcn_dim_l1=dim_l1,
                                                                 gcn_n_ξ_samples=n_ξ_samples)
                    del model_hist.model
                    model_instance.append(model_hist)
                    q_pred_instance.append(q_model_pred)
                dic_ans[model_name]={"history":model_instance,"q_pred":q_pred_instance}

import pickle
if use_bias:
    fname="/warehouse/COMPLEXNET/jlevyabi/nw2vec/yes_bias%d_comm_%d_npc_vae_sens_analysis.p"%(l,k)
    pickle.dump(dic_ans,open(fname,"wb"))
else:
    fname="/warehouse/COMPLEXNET/jlevyabi/nw2vec/no_bias%d_comm_%d_npc_vae_sens_analysis.p"%(l,k)
    pickle.dump(dic_ans,open(fname,"wb"))
