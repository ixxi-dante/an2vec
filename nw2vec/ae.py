import numpy as np

import tensorflow as tf
from tensorflow.contrib.bayesflow import stochastic_tensor as st
from keras import backend as K


def broadcast(array, shape):
    ashape = tuple(array.shape)
    shape = tuple(shape)

    diff = len(shape) - len(ashape)
    assert diff >= 0
    assert ashape == shape[diff:]

    for _ in range(diff):
        array = tf.expand_dims(array, 0)

    multiples = tuple(shape[:diff]) + tuple(np.ones(len(ashape),
                                                    dtype=np.int32))
    return tf.tile(array, multiples)


class GaussianEncoder:

    def __init__(self, μ_flat, D_flat, u_flat):
        D_flat_inv = 1.0 / D_flat
        D_inv = tf.matrix_diag(D_flat_inv)
        D_inv_sqrt = K.sqrt(D_inv)

        u = K.expand_dims(u_flat, -1)
        uT = tf.matrix_transpose(u)
        uT_D_inv_u = uT @ D_inv @ u
        η = 1.0 / (1.0 + uT_D_inv_u)

        self.μ = K.expand_dims(μ_flat, -1)
        self.D_flat = D_flat
        self.u = u
        self.η = η
        self.R = (
            D_inv_sqrt
            - (((1 - K.sqrt(η)) / uT_D_inv_u) * (D_inv @ u @ uT @ D_inv_sqrt))
        )
        self.C = D_inv - (η * (D_inv @ u @ uT @ D_inv))

    def stochastic_tensor(self, n_samples):
        shape = self.μ.shape
        with st.value_type(st.SampleValue(n_samples)):
            ε = st.StochasticTensor(tf.distributions.Normal(
                loc=np.zeros(shape, dtype=np.float32),
                scale=np.ones(shape, dtype=np.float32)))

        return (broadcast(self.μ, [n_samples] + self.μ.shape.as_list())
                + broadcast(self.R, [n_samples] + self.R.shape.as_list()) @ ε)

    def loss(self):
        # TODO: check this gives the right result on test values
        losses = .5 * (tf.trace(self.C)
                       - (K.log(self.η)
                          - K.log(tf.reduce_prod(self.D_flat, -1)))
                       + tf.matrix_transpose(self.μ) @ self.μ)
        return tf.reduce_sum(losses)


class MLPDecoder:
    pass


class BilinearDecoder:
    pass


class VAE:

    def __init__(self, encoder, p_knowing_ξ):
        pass

    def _loss(self):
        pass

    # encoders can be GCN, GraphSAGE, GAT

    # encoders for mu, logd, and u are all Gaussian, and will
    # typically share part of their weights (e.g. a first layer).
    # They can be grouped into an 'encoders' variable.

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
