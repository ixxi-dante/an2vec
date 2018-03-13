import numpy as np

import tensorflow as tf
from tensorflow.contrib.bayesflow import stochastic_tensor as st
from keras import backend as K
import keras


# TODO: test
# TODO[now]: broadcast in the middle of the shape (keeping the batch as outer, the matrices as inner)
def outcast(array, shape):
    """TODO: docs"""

    ashape = tuple(array.shape)
    shape = tuple(shape)

    diff = len(shape) - len(ashape)
    assert diff >= 0
    assert ashape == shape[diff:]

    for _ in range(diff):
        array = K.expand_dims(array, 0)

    multiples = tuple(shape[:diff]) + tuple(np.ones(len(ashape),
                                                    dtype=np.int32))
    return K.tile(array, multiples)


# TODO[now]: take a codec (parametrisation) and output samples by adding a middle dimension after batches and before inner matrices
class ParametrisedStochasticLayer(keras.layers.Layer):
    pass


class GaussianCodec:

    """TODO: docs"""

    # encoders for mu, logD, and u are all Gaussian, and will
    # typically share part of their weights (e.g. a first layer).

    def __init__(self, μ_flat, logD_flat, u_flat):
        """TODO: docs"""

        D = tf.matrix_diag(K.exp(logD_flat))
        D_inv = tf.matrix_diag(K.exp(- logD_flat))
        D_inv_sqrt = tf.matrix_diag(K.exp(- .5 * logD_flat))

        u = K.expand_dims(u_flat, -1)
        uT = tf.matrix_transpose(u)
        uT_D_inv_u = uT @ D_inv @ u
        η = 1.0 / (1.0 + uT_D_inv_u)

        self.dim = μ_flat.shape[-1]
        self.μ = K.expand_dims(μ_flat, -1)
        self.logD_flat = logD_flat
        self.u = u
        self.η = η
        self.R = (
            D_inv_sqrt
            - (((1 - K.sqrt(η)) / uT_D_inv_u) * (D_inv @ u @ uT @ D_inv_sqrt))
        )
        self.C_inv = D + u @ uT
        self.C = D_inv - (η * (D_inv @ u @ uT @ D_inv))

    def sample(self, n_samples):
        """TODO: docs"""

        with st.value_type(st.SampleValue(n_samples)):
            ε = st.StochasticTensor(tf.distributions.Normal(
                loc=np.zeros(self.μ.shape, dtype=np.float32),
                scale=np.ones(self.μ.shape, dtype=np.float32)))

        return (outcast(self.μ, [n_samples] + self.μ.shape.as_list())
                + outcast(self.R, [n_samples] + self.R.shape.as_list()) @ ε)

    # TODO: test
    def logprobability(self, v):
        """TODO: docs"""
        v = outcast(K.expand_dims(v, -1), self.μ.shape)
        return - .5 * (self.dim * np.log(2 * np.pi)
                       + (K.log(self.η) - K.sum(self.logD_flat, axis=-1))
                       + (tf.matrix_transpose(v - self.μ)
                          @ self.C_inv @ (v - self.μ)))

    # TODO: test
    def kl_to_normal(self):
        """TODO: docs"""
        return .5 * (tf.trace(self.C)
                     - (K.log(self.η) - K.sum(self.logD_flat, axis=-1))
                     + tf.matrix_transpose(self.μ) @ self.μ
                     - self.dim)


# Set up the VAE

dim_data = 40    # Dimension base features for each node
dim_mid = 20   # Dimension of intermediate layers
dim_z = 5    # Dimension of the embedding space
n_ξ_samples = 10
adj = []  # Adjacency matrix for convolutions

# Encoder tensor
q_input = Input(shape=(dim_data,))
# CANDO: change act
q_layer1 = GC(dim_mid, adj, use_bias=True, act='relu')(q_input)
q_μ_flat = GC(dim_z, adj, use_bias=True)(q_layer1)
q_logD_flat = GC(dim_z, adj, use_bias=True)(q_layer1)
q_u_flat = GC(dim_z, adj, use_bias=True)(q_layer1)
q_model = Model(inputs=q_input, outputs=[q_μ_flat, q_logD_flat, q_u_flat])

# Decoder model
p_input = Input(shape=(dim_z,))
# CANDO: change act
# TODO: add regularisers to all p layers ( = ||θ||**2 / (2 * κ) )
p_layer1 = Dense(dim_mid, use_bias=True, act='relu')(decoder_input)
p_adj = Bilinear(adj.shape[0])(p_layer1)
p_v_μ_flat = Dense(dim_data, use_bias=True)(p_layer1)
p_v_logD_flat = Dense(dim_data, use_bias=True)(p_layer1)
p_v_u_flat = Dense(dim_data, use_bias=True)(p_layer1)
p_model = Model(inputs=p_input,
                outputs=[p_adj, [p_v_μ_flat, p_v_logD_flat, p_v_u_flat]])


def vae(q_input, q_model, q_codec, n_ξ_samples, p_model, p_codecs):
    # TODO: add the ξ KL as an activity regulariser on one of the layers
    ξ = ParametrisedStochasticLayer(q_codec, n_ξ_samples)(q_model(q_input))
    p_outputs_estimated = [K.mean(samples, axis=-3)
                           for samples in p_outputs_samples]
    model = Model(inputs=q_input, outputs=p_model(ξ))  # <-- OUTPUTS?
    # TODO: ξ.codec
    # TODO: codecs can compute on tensors or on ndarrays: logprobability takes (y_true, y_pred), but we don't want to re-create a new instance for each batch when codec is used inside a ParametrisedStochasticLayer
    # TODO: compose with average over ξ samples to estimate the prob
    losses = [codec.logprobability for codec in p_codecs]
    model.compile('adam',  # CANDO: tune parameters
                  losses,
                  loss_weights, # TODO
                  metrics)      # TODO


# decoders could be made of decoder_adj and decoder_features.
# decoder_adj should be Bernoulli, decoder_features should be
# Gaussian, and they could also share a first layer. They can be
# bilinear or MLP

# encoders are further regularised (Rezende et al. 2014, p.10):
#
# > We regularise the recognition model by introducing
# > additional noise, specifically, bit-flip or drop-out noise
# > at the input layer and small additional Gaussian noise
# > to samples from the recognition model. We use rectified
# > linear activation functions as non-linearities for any
# > deterministic layers of the neural network. We found
# > that such regularisation is essential and without it the
# > recognition model is unable to provide accurate inferences
# > for unseen data points.

# QUESTIONS:
#
# - what obective to use, and how does it depend on
# encoder/decoder
#
# - we should weigh each contribution of decoders to the overall
# loss according to its importance / number of contributing
# items
#
# - how much help does TF need for gradient descent
#
# - how do we minibatch (knowing that it will be model-specific
# because of convolution and the like)
#
# - can the encoders incorporate side-features if the core
# features are not distinctive enough?
