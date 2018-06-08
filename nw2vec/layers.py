import sys
import inspect

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K

from nw2vec import codecs


class GC(keras.layers.Layer):

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 gather_mask=False,
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
        self.use_bias = use_bias
        self.gather_mask = gather_mask
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.input_spec = [keras.engine.InputSpec(ndim=2),
                           keras.engine.InputSpec(ndim=1),
                           keras.engine.InputSpec(ndim=2)]
        self.supports_masking = False

    def build(self, input_shapes):
        adj_shape, mask_shape, features_shape = input_shapes

        assert len(adj_shape) == 2
        assert len(mask_shape) == 1
        assert len(features_shape) == 2

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
        self.input_spec = [keras.engine.InputSpec(ndim=2, axes={0: n_nodes, 1: n_nodes}),
                           keras.engine.InputSpec(ndim=1, axes={0: n_nodes}),
                           keras.engine.InputSpec(ndim=2, axes={0: n_nodes, 1: features_dim})]
        super(GC, self).build(input_shapes)

    def call(self, inputs):
        adj, mask, features = inputs

        A_tilde = tf.sparse_add(adj, tf.eye(tf.shape(adj)[0]))
        D_tilde_out_inv_sqrt = tf.matrix_diag(1.0 / K.sqrt(K.sum(A_tilde, axis=0)))
        D_tilde_in_inv_sqrt = tf.matrix_diag(1.0 / K.sqrt(K.sum(A_tilde, axis=1)))
        A_hat = D_tilde_out_inv_sqrt @ A_tilde @ D_tilde_in_inv_sqrt

        output = tf.matmul(A_hat, features @ self.kernel, a_is_sparse=True)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        if self.gather_mask:
            output = tf.boolean_mask(output, mask)
        else:
            output = output * K.expand_dims(mask, -1)
        return output

    def compute_output_shape(self, input_shapes):
        adj_shape, mask_shape, features_shape = input_shapes
        assert len(adj_shape) == 2
        assert len(mask_shape) == 1
        assert len(features_shape) == 2

        return (None, self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'gather_mask': self.gather_mask,
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
        self.input_spec = [keras.engine.InputSpec(min_ndim=2, max_ndim=3)] * 2
        self.supports_masking = False

    def _process_input_shapes(self, input_shapes, check_concrete=True):
        if not isinstance(input_shapes, list) or len(input_shapes) != 2:
            raise ValueError('A `Bilinear` layer should be called '
                             'on a list of 2 inputs.')
        # The two tensors must have the same shape
        assert input_shapes[0] == input_shapes[1]
        shape = input_shapes[0]
        assert len(shape) in [2, 3]

        shapes_are_fully_defined = all([(None not in input_shape
                                         and tf.Dimension(None) not in input_shape)
                                        for input_shape in input_shapes])

        # Should we require fully defined shapes?
        if check_concrete:
            assert shapes_are_fully_defined

        # Get the bilinear axis
        if shapes_are_fully_defined:
            bilin_axis = self.bilin_axis % len(shape)
        else:
            bilin_axis = self.bilin_axis

        # Check the reduction axis is not the bilinear axis
        assert len(shape) - 1 != bilin_axis
        if not shapes_are_fully_defined:
            assert -1 != bilin_axis

        # Get the diag_axis if it exists
        if shapes_are_fully_defined and len(shape) == 3:
            assert bilin_axis in [0, 1]
            diag_axis = 1 - bilin_axis
            # The diag_axis dimensions must be the same for both tensors.
            # This is also dynamically checked for in `self.call()` for the case
            # where the shapes are not fully defined here.
            assert input_shapes[0][diag_axis] == input_shapes[1][diag_axis]
        else:
            diag_axis = None

        # Note that diag_axis can be `None` either because there is none (len(shape) == 2),
        # or because the shapes are not fully defined
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
                self.bias = self.add_weight(shape=(),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = [keras.layers.InputSpec(min_ndim=2, max_ndim=3,
                                                  axes={-1: input_dim})] * 2
        super(Bilinear, self).build(input_shapes)

    def call(self, inputs):
        tensor0, tensor1 = inputs
        bilin_axis, diag_axis, _ = self._process_input_shapes([tensor0.shape, tensor1.shape])

        Q_tensor1 = tf.tensordot(tensor1, self.kernel, axes=[[-1], [1]])
        output = tf.tensordot(tensor0, Q_tensor1, axes=[[-1], [-1]])
        if diag_axis is not None:
            # Dynamically check that both tensors have the same dimension in diag_axis
            with tf.control_dependencies(
                    [tf.assert_equal(tf.shape(tensor0)[diag_axis], tf.shape(tensor1)[diag_axis])]):
                # Put the bilinear axes first and the diagonal axes last
                output = tf.transpose(output,
                                      perm=[bilin_axis, 2 + bilin_axis, diag_axis, 2 + diag_axis])
                # Take the diagonal elements for the diagonal axes
                output = tf.matrix_diag_part(output)
                # Put the reduced diagonal axis first, the bilinear axes last
                output = tf.transpose(output, perm=[2, 0, 1])

        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        output = K.expand_dims(output, 0)
        return output

    def compute_output_shape(self, input_shapes):
        bilin_axis, diag_axis, _ = self._process_input_shapes(input_shapes, check_concrete=False)
        # diag_axis can be None either because there is none
        # (i.e. len(input_shapes[0]) == len(input_shapes[1]) == 2) or
        # because we don't know its dimension (i.e. shape[diag_axis] is None).
        # Here we only want to skip it if there is none, not if we don't know its dimension.
        axes = ([] if len(input_shapes[0]) == 2 else [(0, diag_axis)]
                + [(0, bilin_axis), (1, bilin_axis)])
        return ((1,) + tuple([input_shapes[tensor][ax] if ax is not None else None
                              for tensor, ax in axes]))

    def get_config(self):
        config = {
            'bilin_axis': self.bilin_axis,
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


def available_layers():
    return dict(
        inspect.getmembers(sys.modules[__name__],
                           lambda m: inspect.isclass(m) and issubclass(m, keras.layers.Layer))
   )
