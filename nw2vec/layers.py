import tensorflow as tf
import keras
from keras import backend as K

from nw2vec import codecs
from nw2vec import utils


# TODO: not only (not at all) a Dense layer
# class GC(keras.layers.Layer):
class GC(keras.layers.Dense):

    def __init__(self, units, adj, **kwargs):
        super(GC, self).__init__(units, **kwargs)


class Bilinear(keras.layers.Layer):

    def __init__(self,
                 bilin_axes,
                 batch_size,
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

        if isinstance(bilin_axes, int):
            bilin_axes = [bilin_axes]
        assert isinstance(bilin_axes, list)
        assert len(bilin_axes) >= 1

        super(Bilinear, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.bilin_axes = bilin_axes
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.input_spec = [keras.engine.InputSpec(min_ndim=2)] * 2
        self.supports_masking = True

    def _process_input_shapes(self, input_shapes, check_concrete=True):
        if not isinstance(input_shapes, list) or len(input_shapes) != 2:
            raise ValueError('A `Bilinear` layer should be called '
                             'on a list of 2 inputs.')
        # The two tensors must have the same shape
        assert input_shapes[0] == input_shapes[1]
        shape = input_shapes[0]

        shape_is_fully_defined = None not in shape and tf.Dimension(None) not in shape
        if check_concrete:
            assert shape_is_fully_defined
        if shape_is_fully_defined:
            # Concretise bilin_axes for this fully defined shape
            bilin_axes = [ax % len(shape) for ax in self.bilin_axes]
        else:
            bilin_axes = self.bilin_axes
        # Reduction axis cannot be in bilin_axes
        assert (len(shape) - 1) not in bilin_axes
        if not shape_is_fully_defined:
            assert -1 not in bilin_axes
        # Shape must be long enough to accomodate bilin_axes + reduction axis
        assert len(shape) >= len(bilin_axes) + 1

        if shape_is_fully_defined:
            diag_axes = list(sorted(set(range(len(shape) - 1)).difference(bilin_axes)))
        else:
            diag_axes = [None]
        return bilin_axes, diag_axes, shape[-1]

    def build(self, input_shapes):
        _, _, input_dim = self._process_input_shapes(input_shapes, check_concrete=False)

        self.kernel = self.add_weight(shape=(input_dim, input_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = [keras.layers.InputSpec(min_ndim=2,
                                                  axes={-1: input_dim,
                                                        0: self.batch_size})] * 2
        super(Bilinear, self).build(input_shapes)

    def call(self, inputs):
        tensor0, tensor1 = inputs
        bilin_axes, diag_axes, _ = self._process_input_shapes([tensor0.shape, tensor1.shape])

        Q_tensor0 = tf.tensordot(tensor0, self.kernel, axes=[[-1], [1]])
        output = tf.tensordot(tensor1, Q_tensor0, axes=[[-1], [-1]])
        # Put the bilinear axes first (in the order requested) and the diagonal axes last,
        # for each half of the tensordot axes
        n_bilin = len(bilin_axes)
        n_diag = len(diag_axes)
        output = tf.transpose(output,
                              perm=(bilin_axes + diag_axes
                                    + [ax + n_bilin + n_diag for ax in bilin_axes]
                                    + [ax + n_bilin + n_diag for ax in diag_axes]))
        # Take the diagonal elements for all the non-bilinear axis couples
        perm_base = list(range(n_bilin)) + list(range(n_bilin + 1, n_bilin + n_diag))
        for n_reduced, rem in enumerate(range(n_bilin + n_diag - 1, n_bilin - 1, -1)):
            perm = (perm_base[:rem]
                    + [rem + 1 + ax for ax in perm_base[:rem]]
                    + list(range(2 * (rem + 1), 2 * (rem + 1) + n_reduced))
                    + [n_bilin, rem + 1 + n_bilin])
            output = tf.transpose(output, perm=perm)
            assert output.shape[-1] == output.shape[-2]
            output = tf.matrix_diag_part(output)
        # Put the diagonal axes first, the bilinear axes last
        output = tf.transpose(output,
                              perm=(list(range(2 * n_bilin, 2 * n_bilin + n_diag))
                                    + list(range(2 * n_bilin))))

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return utils.expand_dims_tile(output, 0, self.batch_size)

    def compute_output_shape(self, input_shapes):
        bilin_axes, diag_axes, _ = self._process_input_shapes(input_shapes, check_concrete=False)
        return (self.batch_size,) + tuple([input_shapes[0][ax] if ax is not None else None
                                           for ax in diag_axes + 2 * bilin_axes])

    def get_config(self):
        config = {
            'batch_size': self.batch_size,
            'bilin_axes': self.bilin_axes,
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
