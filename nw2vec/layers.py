import numpy as np
import tensorflow as tf
import keras
from keras import backend as K

from nw2vec import codecs


class GC(keras.layers.Layer):

    def __init__(self,
                 units,
                 activation=None,
                 use_gather=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GC, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_gather = use_gather
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.input_spec = [keras.engine.InputSpec(ndim=2),
                           keras.engine.InputSpec(ndim=2, axes={1: 1}),
                           keras.engine.InputSpec(ndim=2)]
        self.supports_masking = True

    def build(self, input_shapes):
        adj_shape, gather_shape, features_shape = input_shapes

        assert len(adj_shape) == 2
        assert len(gather_shape) == 2
        assert len(features_shape) == 2

        assert adj_shape[0] == adj_shape[1]
        assert adj_shape[0] >= gather_shape[0]
        assert gather_shape[1] == 1
        assert features_shape[0] == adj_shape[0]

        n_nodes = adj_shape[0]
        features_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(features_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec[0] = keras.engine.InputSpec(ndim=2, axes={0: n_nodes, 1: n_nodes})
        self.input_spec[2] = keras.engine.InputSpec(ndim=2, axes={0: n_nodes, 1: features_dim})
        super(GC, self).build(input_shapes)

    def call(self, inputs):
        adj, gather, features = inputs

        A_tilde = adj + tf.eye(tf.shape(adj)[0])
        D_tilde_inv_sqrt = tf.matrix_diag(1.0 / K.sqrt(K.sum(A_tilde, axis=1)))
        A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

        output = tf.matmul(A_hat, features @ self.kernel, a_is_sparse=True)
        if self.use_gather:
            output = tf.gather(output, K.squeeze(gather, 1), axis=0)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shapes):
        adj_shape, gather_shape, features_shape = input_shapes
        assert len(adj_shape) == 2
        assert len(gather_shape) == 2
        assert len(features_shape) == 2

        return (gather_shape[0] if self.use_gather else features_shape[0],
                self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_gather': self.use_gather,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint)
        }
        base_config = super(GC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Recover any numpy array arguments
        config = config.copy()
        for key in config:
            if isinstance(config[key], dict):
                if 'type' in config[key] and config[key]['type'] == 'ndarray':
                    config[key] = np.array(config[key]['value'])

        return cls(**config)


class Bilinear(keras.layers.Layer):

    def __init__(self,
                 bilin_axis,
                 batch_size,
                 fixed_kernel=None,
                 fixed_bias=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        # Validate arguments
        if fixed_bias is not None:
            assert use_bias

        super(Bilinear, self).__init__(**kwargs)
        self.bilin_axis = bilin_axis
        self.batch_size = batch_size
        self.fixed_kernel = fixed_kernel
        self.fixed_bias = fixed_bias
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.input_spec = [keras.engine.InputSpec(min_ndim=2, max_ndim=3,
                                                  axes={bilin_axis: self.batch_size})] * 2
        self.supports_masking = False

    def _process_input_shapes(self, input_shapes, check_concrete=True):
        if not isinstance(input_shapes, list) or len(input_shapes) != 2:
            raise ValueError('A `Bilinear` layer should be called '
                             'on a list of 2 inputs.')
        # The two tensors must have the same shape
        assert input_shapes[0] == input_shapes[1]
        shape = input_shapes[0]
        assert len(shape) == 2 or len(shape) == 3

        shape_is_fully_defined = None not in shape and tf.Dimension(None) not in shape
        if check_concrete:
            assert shape_is_fully_defined
        if shape_is_fully_defined:
            # Concretise axis for this fully defined shape
            bilin_axis = self.bilin_axis % len(shape)
        else:
            bilin_axis = self.bilin_axis
        # Reduction axis cannot be the bilinear axis
        assert len(shape) - 1 != bilin_axis
        if not shape_is_fully_defined:
            assert -1 != bilin_axis

        if shape_is_fully_defined and len(shape) == 3:
            assert bilin_axis in [0, 1]
            diag_axis = 1 - bilin_axis
        else:
            diag_axis = None
        return bilin_axis, diag_axis, shape[-1]

    def build(self, input_shapes):
        bilin_axis, _, input_dim = self._process_input_shapes(input_shapes, check_concrete=False)

        if self.fixed_kernel is not None:
            self.kernel = K.constant(self.fixed_kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, input_dim),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            if self.fixed_bias is not None:
                self.bias = K.constant(self.fixed_bias)
            else:
                self.bias = self.add_weight(shape=(1,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            # K.bias_add (in self.call()) requires something that has shape dim(output) - 1.
            # In this case we want the same bias added to all bilinear combinations, so we
            # tile our scalar into a vector, which gets added to all columns of the bilinear
            # matrix output
            self.bias = K.tile(self.bias, [self.batch_size])
        else:
            self.bias = None

        self.input_spec = [keras.layers.InputSpec(min_ndim=2, max_ndim=3,
                                                  axes={-1: input_dim,
                                                        bilin_axis: self.batch_size})] * 2
        super(Bilinear, self).build(input_shapes)

    def call(self, inputs):
        tensor0, tensor1 = inputs
        bilin_axis, diag_axis, _ = self._process_input_shapes([tensor0.shape, tensor1.shape])

        Q_tensor1 = tf.tensordot(tensor1, self.kernel, axes=[[-1], [1]])
        output = tf.tensordot(tensor0, Q_tensor1, axes=[[-1], [-1]])
        if diag_axis is not None:
            # Put the bilinear axes first and the diagonal axes last
            output = tf.transpose(output,
                                  perm=[bilin_axis, 2 + bilin_axis, diag_axis, 2 + diag_axis])
            # Take the diagonal elements for the diagonal axes
            output = tf.matrix_diag_part(output)
            # Put the reduced diagonal axis first, the bilinear axes last
            output = tf.transpose(output, perm=[2, 0, 1])

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shapes):
        bilin_axis, diag_axis, _ = self._process_input_shapes(input_shapes, check_concrete=False)
        shape = input_shapes[0]
        bilin_axes = [bilin_axis]
        # diag_axis can be None either because there is none (i.e. len(shape) == 2) or
        # because we don't know its dimension (i.e. shape[diag_axis] is None)
        diag_axes = [] if len(shape) == 2 else [diag_axis]
        return tuple([input_shapes[0][ax] if ax is not None else None
                      for ax in diag_axes + 2 * bilin_axes])

    def get_config(self):
        config = {
            'bilin_axis': self.bilin_axis,
            'batch_size': self.batch_size,
            'fixed_kernel': self.fixed_kernel,
            'fixed_bias': self.fixed_bias,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint)
        }
        base_config = super(Bilinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Recover any numpy array arguments
        config = config.copy()
        for key in config:
            if isinstance(config[key], dict):
                if 'type' in config[key] and config[key]['type'] == 'ndarray':
                    config[key] = np.array(config[key]['value'])

        return cls(**config)


class ParametrisedStochastic(keras.layers.Lambda):

    def __init__(self, codec_name, n_samples, **kwargs):
        self.codec_name = codec_name
        self.n_samples = n_samples

        def sampler(params):
            return codecs.get(codec_name, params).stochastic_value(n_samples)

        super(ParametrisedStochastic, self).__init__(sampler, **kwargs)

    def get_config(self):
        config = {
            'codec_name': self.codec_name,
            'n_samples': self.n_samples
        }
        # Skip the Lambda-specific config parameters as we recreate the Lambda layer ourselves
        base_config = super(keras.layers.Lambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Recover any numpy array arguments
        config = config.copy()
        for key in config:
            if isinstance(config[key], dict):
                if 'type' in config[key] and config[key]['type'] == 'ndarray':
                    config[key] = np.array(config[key]['value'])

        return cls(**config)
