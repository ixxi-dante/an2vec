import functools

import numpy as np

import tensorflow as tf
from keras import backend as K

from nw2vec.utils import right_squeeze2, expand_dims_tile


class Codec:

    def stochastic_value(self, n_samples):
        raise NotImplementedError

    def logprobability(self, v):
        raise NotImplementedError


class Gaussian(Codec):

    """TODOC"""

    def __init__(self, μlogDu_flat):
        """TODOC"""

        # Check the flattened parameters have the right shape
        concat_dim = int(μlogDu_flat.shape[-1])
        assert concat_dim % 3 == 0
        # Extract the separate parameters
        self.dim = concat_dim // 3
        outer_slices = [slice(None)] * (len(μlogDu_flat.shape) - 1)
        μ_flat = μlogDu_flat[outer_slices + [slice(self.dim)]]
        logD_flat = μlogDu_flat[outer_slices + [slice(self.dim, 2 * self.dim)]]
        u_flat = μlogDu_flat[outer_slices + [slice(2 * self.dim, None)]]

        # Prepare the D matrix
        D = tf.matrix_diag(K.exp(logD_flat))
        D_inv = tf.matrix_diag(K.exp(- logD_flat))
        D_inv_sqrt = tf.matrix_diag(K.exp(- .5 * logD_flat))

        # Some pre-computations
        u = K.expand_dims(u_flat, -1)
        uT = tf.matrix_transpose(u)
        uT_D_inv_u = uT @ D_inv @ u
        η = 1.0 / (1.0 + uT_D_inv_u)

        self.μ = K.expand_dims(μ_flat, -1)
        self.R = D_inv_sqrt - (((1 - K.sqrt(η)) / uT_D_inv_u) * (D_inv @ u @ uT @ D_inv_sqrt))
        self.C_inv = D + u @ uT
        self.C = D_inv - (η * (D_inv @ u @ uT @ D_inv))
        self.logdetC = right_squeeze2(K.log(η)) - K.sum(logD_flat, axis=-1)

    # TOTEST
    def stochastic_value(self, n_samples):
        """TODOC"""
        μ_shape = self.μ.shape.as_list()
        ε = tf.random_normal(μ_shape[:-2] + [n_samples] + μ_shape[-2:])
        return K.squeeze(expand_dims_tile(self.μ, -3, n_samples)
                         + expand_dims_tile(self.R, -3, n_samples) @ ε,
                         -1)

    # TOTEST
    def logprobability(self, v):
        """TODOC"""
        v = K.expand_dims(v, -1)
        # In lieu of assert v.shape == self.μ.shape
        v.shape.assert_is_compatible_with(self.μ.shape)
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


class SigmoidBernoulli(Codec):

    def __init__(self, logits):
        """TODOC"""
        self.logits = logits

    # TOTEST
    def logprobability(self, v):
        """TODOC"""
        # In lieu of assert v.shape == self.logits.shape
        v.shape.assert_is_compatible_with(self.logits.shape)
        return - K.sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=v, logits=self.logits),
                       axis=-1)


class Bernoulli(Codec):

    def __init__(self, probs):
        """TODOC"""
        self.probs = probs

    # TOTEST
    def logprobability(self, v):
        """TODOC"""
        # In lieu of assert v.shape == self.probs.shape
        v.shape.assert_is_compatible_with(self.probs.shape)
        return K.sum(v * K.log(self.probs) + (1.0 - v) * K.log(1 - self.probs), axis=-1)


@functools.lru_cache(typed=True)
def get(klass, *args, **kwargs):
    return klass(*args, **kwargs)
