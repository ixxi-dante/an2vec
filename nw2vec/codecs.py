import functools

import numpy as np

import tensorflow as tf
from keras import backend as K

from nw2vec.utils import right_squeeze2, expand_dims_tile, broadcast_left


class Codec:

    __losses__ = ['estimated_pred_loss', 'kl_to_normal_loss']

    def __init__(self, params):
        self.params = params

    def stochastic_value(self, n_samples):
        raise NotImplementedError

    def logprobability(self, v):
        raise NotImplementedError

    def kl_to_normal(self):
        raise NotImplementedError

    def estimated_pred_loss(self, y_true, y_pred):
        # `y_pred` has shape (batch, sampling[, ...]+), but `y_true` can
        # be less than that if values are repeated (in which case
        # `self.logprobability()` will broadcast it)
        if isinstance(y_pred, np.ndarray):
            assert (y_pred == self.params).all()
        else:
            assert y_pred == self.params
        logprobability = self.logprobability(y_true)
        # We're left with the batch axis + sampling axis. Whatever other
        # inner dimensions there were in `y_pred` should be averaged
        # over by `self.logprobability()`.
        assert len(logprobability.shape) == 2
        # This loss is *estimated*, i.e. based on a sample,
        # hence the average over axis 1 which is the sampling axis
        return - K.mean(logprobability, axis=1)

    def kl_to_normal_loss(self, y_true, y_pred):
        # `y_pred` has shape (batch, values), and `y_true` is ignored
        assert len(y_pred.shape) == 2
        assert y_pred == self.params
        kl_to_normal = self.kl_to_normal()
        assert len(kl_to_normal.shape) == 1
        return kl_to_normal


class Gaussian(Codec):

    """TODOC"""

    def __init__(self, params):
        """TODOC"""

        super(Gaussian, self).__init__(params)

        # Check the flattened parameters have the right shape
        concat_dim = int(params.shape[-1])
        assert concat_dim % 3 == 0
        # Extract the separate parameters
        self.dim = concat_dim // 3
        outer_slices = [slice(None)] * (len(params.shape) - 1)
        μ_flat = params[outer_slices + [slice(self.dim)]]
        logD_flat = params[outer_slices + [slice(self.dim, 2 * self.dim)]]
        u_flat = params[outer_slices + [slice(2 * self.dim, None)]]

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
        μ_shape = tf.shape(self.μ)
        ε = tf.random_normal(tf.concat([μ_shape[:-2], [n_samples], μ_shape[-2:]], 0),
                             dtype=K.floatx())
        return K.squeeze(expand_dims_tile(self.μ, -3, n_samples)
                         + expand_dims_tile(self.R, -3, n_samples) @ ε,
                         -1)

    # TOTEST
    def logprobability(self, v):
        """TODOC"""
        # Turn `v` into a column vector
        v = K.expand_dims(v, -1)
        # Check shapes and broadcast
        v = broadcast_left(v, self.μ)
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


class OrthogonalGaussian(Codec):

    """TODOC"""

    def __init__(self, params):
        """TODOC"""

        super(OrthogonalGaussian, self).__init__(params)

        # Check the flattened parameters have the right shape
        concat_dim = int(params.shape[-1])
        assert concat_dim % 2 == 0
        # Extract the separate parameters
        self.dim = concat_dim // 2
        outer_slices = [slice(None)] * (len(params.shape) - 1)
        μ_flat = params[outer_slices + [slice(self.dim)]]
        logS_flat = params[outer_slices + [slice(self.dim, 2 * self.dim)]]

        self.μ = K.expand_dims(μ_flat, -1)

        self.S_flat = K.exp(logS_flat)
        self.S = tf.matrix_diag(self.S_flat)
        self.logdetS = K.sum(logS_flat, axis=-1)

        self.S2_inv = tf.matrix_diag(K.exp(- 2 * logS_flat))
        self.traceS2 = K.sum(K.exp(2 * logS_flat), axis=-1)

    # TOTEST
    def stochastic_value(self, n_samples):
        """TODOC"""
        μ_shape = tf.shape(self.μ)
        ε = tf.random_normal(tf.concat([μ_shape[:-2], [n_samples], μ_shape[-2:]], 0),
                             dtype=K.floatx())
        return K.squeeze(expand_dims_tile(self.μ, -3, n_samples)
                         + expand_dims_tile(self.S, -3, n_samples) @ ε,
                         -1)

    # TOTEST
    def logprobability(self, v):
        """TODOC"""
        # Turn `v` into a column vector
        v = K.expand_dims(v, -1)
        # Check shapes and broadcast
        v = broadcast_left(v, self.μ)
        return - .5 * (self.dim * np.log(2 * np.pi) + 2 * self.logdetS
                       + right_squeeze2(tf.matrix_transpose(v - self.μ)
                                        @ self.S2_inv
                                        @ (v - self.μ)))

    # TOTEST
    def kl_to_normal(self):
        """TODOC"""
        return .5 * (self.traceS2 - 2 * self.logdetS
                     + right_squeeze2(tf.matrix_transpose(self.μ) @ self.μ)
                     - self.dim)


class SigmoidBernoulli(Codec):

    def __init__(self, params):
        """TODOC"""
        super(SigmoidBernoulli, self).__init__(params)
        self.logits = params

    # TOTEST
    def logprobability(self, v):
        """TODOC"""
        # This implementation is only valid for vectors as input

        # Check shapes and broadcast
        v = broadcast_left(v, self.logits)
        # `v` now has shape (batch, sampling, dim(vectors))
        assert len(v.shape) == 3  # If this fails, change it to a dynamic check

        sigmoid_cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(labels=v,
                                                                          logits=self.logits)
        return - K.sum(sigmoid_cross_entropies, axis=-1)


class SigmoidBernoulliAdjacency(Codec):

    def __init__(self, params):
        """TODOC"""
        super(SigmoidBernoulliAdjacency, self).__init__(params)
        self.logits = params

    # TOTEST
    def logprobability(self, adj):
        """TODOC"""
        # This implementation is only valid for an adjacency matrix as input, not for a vector

        # Check shapes and broadcast
        adj = broadcast_left(adj, self.logits)
        # `adj` now has shape (batch or 1, sampling, batch, batch)
        assert len(adj.shape) == 4  # If this fails, change it to a dynamic check

        weighted_sigmoid_cross_entropies = \
            tf.nn.sigmoid_cross_entropy_with_logits(labels=adj, logits=self.logits)

        return - K.mean(K.sum(weighted_sigmoid_cross_entropies, axis=-1), axis=-1)


class SigmoidBernoulliScaledAdjacency(Codec):

    def __init__(self, params):
        """TODOC"""
        super(SigmoidBernoulliScaledAdjacency, self).__init__(params)
        self.logits = params

    # TOTEST
    def logprobability(self, adj):
        """TODOC"""
        # This implementation is only valid for an adjacency matrix as input, not for a vector

        # Check shapes and broadcast
        adj = broadcast_left(adj, self.logits)
        # `adj` now has shape (batch or 1, sampling, batch, batch)
        assert len(adj.shape) == 4  # If this fails, change it to a dynamic check

        # Remove any potential diagonal elements for the density computation
        diagonal = tf.matrix_diag(tf.matrix_diag_part(adj))
        density = K.mean(adj - diagonal)
        weighted_sigmoid_cross_entropies = (
            .5
            * tf.nn.weighted_cross_entropy_with_logits(
                targets=adj, logits=self.logits, pos_weight=(1 / density) - 1)
            / (1 - density)
        )

        return - K.mean(K.sum(weighted_sigmoid_cross_entropies, axis=-1), axis=-1)


@functools.lru_cache(typed=True)
def get(codec_name, *args, **kwargs):
    codecs = available_codecs()
    assert codec_name in codecs
    return codecs[codec_name](*args, **kwargs)


def available_codecs():
    return {klass.__name__: klass for klass in Codec.__subclasses__()}


def get_loss(codec_name, loss_name):
    assert loss_name in Codec.__losses__
    assert codec_name in available_codecs().keys()

    def loss(y_true, y_pred):
        return getattr(get(codec_name, y_pred), loss_name)(y_true, y_pred)

    loss.__name__ = loss_fullname(codec_name, loss_name)
    return loss


def get_loss_by_fullname(loss_fullname):
    codec_name, loss_name = destructure_loss_fullname(loss_fullname)
    return get_loss(codec_name, loss_name)


def loss_fullname(codec_name, loss_name):
    return codec_name + '__' + loss_name


def destructure_loss_fullname(loss_fullname):
    codec_name, loss_name = loss_fullname.split('__')
    return codec_name, loss_name


def available_fullname_losses():
    codecs = available_codecs()
    return {loss_fullname(codec_name, loss_name): get_loss(codec_name, loss_name)
            for codec_name in codecs.keys()
            for loss_name in Codec.__losses__}
