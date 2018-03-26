import tensorflow as tf
import keras
from keras import backend as K

from nw2vec import codecs
from nw2vec import utils


class GC(keras.layers.Layer):

    def __init__(self,
                 units,
                 adj,
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
        assert len(adj.shape) == 2 and adj.shape[0] == adj.shape[1]

        super(GC, self).__init__(**kwargs)
        self.units = units
        self.adj = adj
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.input_spec = keras.engine.InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        A_tilde = self.adj + tf.eye(self.adj.shape[0])
        D_tilde_inv_sqrt = tf.matrix_diag(1.0 / K.sqrt(K.sum(A_tilde, axis=1)))
        self.A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

        self.kernel = self.add_weight(shape=(input_dim, self.units),
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
        self.input_spec = keras.engine.InputSpec(min_ndim=2, axes={-1: input_dim,
                                                                   0: self.adj.shape[0]})
        super(GC, self).build(input_shape)

    def call(self, inputs):
        output = self.A_hat @ inputs @ self.kernel
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert len(input_shape) >= 2
        assert input_shape[-1] > 0
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        # FIXME: this will bail with numpy arrays
        config = {
            'units': self.units,
            'adj': self.adj,
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
        base_config = super(GC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
        self.supports_masking = True

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
        return utils.expand_dims_tile(output, 0, self.batch_size)

    def compute_output_shape(self, input_shapes):
        bilin_axis, diag_axis, _ = self._process_input_shapes(input_shapes, check_concrete=False)
        shape = input_shapes[0]
        bilin_axes = [bilin_axis]
        # diag_axis can be None either because there is none (i.e. len(shape) == 2) or
        # because we don't know its dimension (i.e. shape[diag_axis] is None)
        diag_axes = [] if len(shape) == 2 else [diag_axis]
        return (self.batch_size,) + tuple([input_shapes[0][ax] if ax is not None else None
                                           for ax in diag_axes + 2 * bilin_axes])

    def get_config(self):
        # FIXME: this will bail with numpy arrays
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


class ParametrisedStochastic(keras.layers.Lambda):

    def __init__(self, codec, n_samples, **kwargs):
        self.codec = codec

        def sampler(params):
            return codecs.get(codec, params).stochastic_value(n_samples)

        super(ParametrisedStochastic, self).__init__(sampler, **kwargs)

    # TODO: get_config
