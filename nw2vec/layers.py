import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import _to_snake_case


class GC(keras.layers.Layer):
    pass


class Bilinear(keras.layers.Dot):

    def __init__(self, tensor1, tensor2, **kwargs):
        assert tensor1.shape == tensor2.shape

        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__
            name = _to_snake_case(prefix) + '_' + str(K.get_uid(prefix))
        self.name = name

        with tf.variable_scope(name):
            self.transform = keras.layers.Dense(tensor1.shape[-1], use_bias=False,
                                                name='transform')
            super(Bilinear, self).__init__(self.transform(tensor1), tensor2, **kwargs)


class ParametrisedStochastic(keras.layers.Lambda):

    def __init__(self, codec, n_samples, **kwargs):
        self.codec = codec

        def sampler(params):
            return codec(params).stochastic_value(n_samples)

        super(ParametrisedStochastic, self).__init__(sampler, **kwargs)
