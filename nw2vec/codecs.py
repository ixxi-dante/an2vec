import functools

import numpy as np

import tensorflow as tf
from tensorflow.contrib.bayesflow import stochastic_tensor as st
from keras import backend as K

from nw2vec.utils import right_squeeze2, expand_dims_tile


class Codec:

    def stochastic_value(self, n_samples):
        raise NotImplementedError

    def logprobability(self, v):
        raise NotImplementedError


@functools.lru_cache(typed=True)
class Gaussian(Codec):

    """TODOC"""

    # encoders for mu, logD, and u are all Gaussian, and will
    # typically share part of their weights (e.g. a first layer).

    def __init__(self, μlogDu_flat):
        """TODOC"""

        # Check the flattened parameters have the right shape
        concat_dim = int(μlogDu_flat.shape[-1])
        assert concat_dim % 3 == 0
        # Extract the separate parameters
        self.dim = concat_dim // 3
        outer_slices = [slice(None)] * (len(μlogDu_flat.shape) - 1)
        μ_flat = μlogDu_flat[outer_slices + slice(concat_dim)]
        logD_flat = μlogDu_flat[outer_slices + slice(concat_dim, 2 * concat_dim)]
        u_flat = μlogDu_flat[outer_slices + slice(2 * concat_dim, None)]

        # Prepare the D matrix
        D = tf.matrix_diag(K.exp(logD_flat))
        D_inv = tf.matrix_diag(K.exp(- logD_flat))
        D_inv_sqrt = tf.matrix_diag(K.exp(- .5 * logD_flat))

        # Some pre-computations
        u = K.expand_dims(u_flat, -1)
        uT = tf.matrix_transpose(u)
        uT_D_inv_u = right_squeeze2(uT @ D_inv @ u)
        η = 1.0 / (1.0 + uT_D_inv_u)

        self.μ = K.expand_dims(μ_flat, -1)
        self.logD_flat = logD_flat
        self.u = u
        self.η = η
        self.R = D_inv_sqrt - (((1 - K.sqrt(η)) / uT_D_inv_u) * (D_inv @ u @ uT @ D_inv_sqrt))
        self.C_inv = D + u @ uT
        self.C = D_inv - (η * (D_inv @ u @ uT @ D_inv))
        self.logdetC = K.log(η) - K.sum(logD_flat, axis=-1)

    # TOTEST
    def stochastic_value(self, n_samples):
        """TODOC"""

        with st.value_type(st.SampleValue(n_samples)):
            ε = st.StochasticTensor(tf.distributions.Normal(
                loc=np.zeros(self.μ.shape, dtype=np.float32),
                scale=np.ones(self.μ.shape, dtype=np.float32)))

        dims = list(range(len(ε.shape)))
        ε = tf.transpose(ε, perm=dims[:-3] + dims[-1:] + dims[-3:-1])

        return K.squeeze(expand_dims_tile(self.μ, -3, n_samples)
                         + expand_dims_tile(self.R, -3, n_samples) @ ε,
                         -1)

    # TOTEST
    def logprobability(self, v):
        """TODOC"""
        v = K.expand_dims(v, -1)
        assert v.shape == self.μ.shape
        return - .5 * (self.dim * np.log(2 * np.pi) + self.logdetC
                       + right_squeeze2(tf.matrix_transpose(v - self.μ)
                                        @ self.C_inv
                                        @ (v - self.μ)))

    # TOTEST
    def kl_to_normal(self):
        """TODOC"""
        return .5 * (tf.trace(self.C) - self.logdetC
                     + right_squeeze2(tf.matrix_transpose(self.μ) @ self.μ)
                     - self.dim)


class Bernouilli(Codec):
    pass
