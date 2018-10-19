import sys
import inspect

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K

from nw2vec import codecs
from nw2vec import utils


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

        A_tilde = tf.sparse_add(adj, tf.eye(tf.shape(adj)[0], dtype=K.floatx()))
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
                 call_impl='whileloop',  # set to "tensordot" to revert to the old implementation
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
        assert call_impl in ['whileloop', 'tensordot']
        self.call_impl = call_impl
        self.input_spec = [keras.engine.InputSpec(min_ndim=2, max_ndim=3)] * 2
        self.supports_masking = False

    def _process_input_shapes(self, input_shapes):
        if not isinstance(input_shapes, list) or len(input_shapes) != 2:
            raise ValueError('A `Bilinear` layer should be called '
                             'on a list of 2 inputs.')
        # The two tensors must have the same shape
        assert input_shapes[0] == input_shapes[1]
        shape = input_shapes[0]
        assert len(shape) in [2, 3]

        # Check the reduction axis is not the bilinear axis
        bilin_axis = self.bilin_axis % len(shape)
        if len(shape) == 3:
            assert bilin_axis in [0, 1]
        else:
            assert bilin_axis == 0

        # Get the diag_axis if it exists
        if len(shape) == 3:
            diag_axis = 1 - bilin_axis
        else:
            diag_axis = None

        return bilin_axis, diag_axis, shape[-1]

    def build(self, input_shapes):
        bilin_axis, _, input_dim = self._process_input_shapes(input_shapes)

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
        if self.call_impl == 'whileloop':
            return self.call_whileloop(inputs)
        else:
            return self.call_tensordot(inputs)

    def call_whileloop(self, inputs):
        tensor0, tensor1 = inputs
        bilin_axis, diag_axis, _ = self._process_input_shapes([tensor0.shape, tensor1.shape])
        assert bilin_axis in [0, 1]
        assert (diag_axis is None) or (diag_axis == 1 - bilin_axis)
        bilin_dim, diag_dim = tensor0.shape[bilin_axis], tensor0.shape[diag_axis]
        if isinstance(bilin_dim, tf.Dimension):
            bilin_dim = bilin_dim.value
        if isinstance(diag_dim, tf.Dimension):
            diag_dim = diag_dim.value

        K_tensor1 = tf.tensordot(tensor1, self.kernel, axes=[[-1], [1]])
        if diag_axis is None:
            assert len(tensor0.shape) == len(tensor1.shape) == 2
            output = tf.tensordot(tensor0, K_tensor1, axes=[[-1], [-1]])
        else:

            def diag_slice(k):
                idx = [0.0, 0.0, slice(None)]
                idx[diag_axis] = k
                idx[bilin_axis] = slice(None)
                return idx

            def counter(i, t):
                return i < diag_dim

            def body(i, t):
                idx = diag_slice(i)
                slice_dot = tf.tensordot(tensor0[idx], K_tensor1[idx], axes=[[-1], [-1]])
                return (i + 1, tf.concat([t, K.expand_dims(slice_dot, 0)], axis=0))

            i0 = K.constant(0, dtype=tf.int32)
            if bilin_dim is None:
                bilin_dim_dyn = tf.shape(tensor0)[bilin_axis]
                out0 = K.zeros((0, bilin_dim_dyn, bilin_dim_dyn))
            else:
                out0 = K.zeros((0, bilin_dim, bilin_dim))

            _, output = tf.while_loop(
                counter, body, loop_vars=[i0, out0],
                shape_invariants=[i0.get_shape(), tf.TensorShape([None, bilin_dim, bilin_dim])]
            )

        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        output = K.expand_dims(output, 0)
        return output

    def call_tensordot(self, inputs):
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
        bilin_axis, diag_axis, _ = self._process_input_shapes(input_shapes)
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


class InnerSlice(keras.layers.Lambda):

    def __init__(self, inner_slice, **kwargs):
        if isinstance(inner_slice, (tuple, list)):
            inner_slice = slice(*inner_slice)
        self.inner_slice = inner_slice

        def slicer(input):
            outer_slices = (slice(None),) * (len(input.shape) - 1)
            return input[outer_slices + (inner_slice,)]

        super(InnerSlice, self).__init__(slicer, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (utils.slice_size(self.inner_slice, input_shape[-1]),)

    def get_config(self):
        config = {
            'inner_slice': (self.inner_slice.start, self.inner_slice.stop, self.inner_slice.step),
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


class OverlappingConcatenate(keras.layers.Lambda):

    def __init__(self, overlap_size, reducer, **kwargs):
        assert reducer in ['mean', 'sum', 'max', 'min']
        self.overlap_size = overlap_size
        self.reducer = reducer

        def overlapper(inputs):
            input1, input2 = inputs
            shape1, shape2 = input1.shape.as_list(), input2.shape.as_list()

            assert len(shape1) == len(shape2)
            assert shape1[:-1] == shape2[:-1]
            assert overlap_size <= min(shape1[-1], shape2[-1])
            outer_slices = [slice(None)] * (len(shape1) - 1)

            cropped1 = input1[outer_slices + [slice(None, shape1[-1] - overlap_size)]]
            overlap1 = input1[outer_slices + [slice(shape1[-1] - overlap_size, None)]]
            cropped2 = input2[outer_slices + [slice(overlap_size, None)]]
            overlap2 = input2[outer_slices + [slice(None, overlap_size)]]

            overlap_parts = tf.stack([overlap1, overlap2], axis=0)
            reducer_fn = getattr(tf, 'reduce_' + reducer)
            overlap = reducer_fn(overlap_parts, axis=0)

            return tf.concat([cropped1, overlap, cropped2], axis=-1)

        super(OverlappingConcatenate, self).__init__(overlapper, **kwargs)

    def compute_output_shape(self, input_shapes):
        shape1, shape2 = input_shapes
        assert len(shape1) == len(shape2)
        assert shape1[:-1] == shape2[:-1]
        return shape1[:-1] + (shape1[-1] + shape2[-1] - self.overlap_size,)

    def get_config(self):
        config = {
            'overlap_size': self.overlap_size,
            'reducer': self.reducer,
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


class ParametrisedStochastic(keras.layers.Lambda):

    def __init__(self, codec_name, n_samples, **kwargs):
        self.codec_name = codec_name
        self.n_samples = n_samples

        def sampler(params):
            return codecs.get(codec_name, params).stochastic_value(n_samples)

        super(ParametrisedStochastic, self).__init__(sampler, **kwargs)

    def compute_output_shape(self, input_shape):
        codec_output_shape = codecs.available_codecs()[self.codec_name]\
            .compute_output_shape(input_shape)
        return codec_output_shape[:-1] + (self.n_samples,) + codec_output_shape[-1:]

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
