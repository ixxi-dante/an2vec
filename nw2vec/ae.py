import keras
import tensorflow as tf

from nw2vec import layers
from nw2vec import codecs


class Model(keras.Model):

    def predict_on_fed_batch(self, x, feeds={}):
        # Get the model's `feed_dict`
        if (not hasattr(self, '_function_kwargs')
                or not isinstance(self._function_kwargs, dict)
                or not isinstance(self._function_kwargs.get('feed_dict'), dict)):
            if hasattr(self, 'predict_function'):
                raise ValueError(("Model {} has a `predict_function` but "
                                  "`_function_kwargs` is absent or is `None`, "
                                  "or has no `feed_dict` dict. Most likely this "
                                  "model has been compiled or has run a "
                                  "prediction without a feed_dict.").format(self))
            else:
                # No `feed_dict`, but also no `predict_function`,
                # so we can safely add a `_function_kwargs` with a `feed_dict`
                # and it will be used upon compilation of the `predict_function`
                # (which happens at the next call to one of the model.predict*
                # methods).
                self._function_kwargs = {'feed_dict': {}}
        feed_dict = self._function_kwargs['feed_dict']

        # Find the model's name for each feed tensor
        feed_layers = [layer for layer in self.layers
                       if (isinstance(layer, keras.layers.InputLayer)
                           and not layer.is_placeholder)]
        feeds_to_tensors = {}
        for feed_layer in feed_layers:
            assert len(feed_layer._inbound_nodes) == 1
            assert len(feed_layer._inbound_nodes[0].input_tensors) == 1
            feeds_to_tensors[feed_layer.name] = feed_layer._inbound_nodes[0].input_tensors[0].name

        # `feeds` should provide values for all the input tensors we found
        assert set(feeds_to_tensors.keys()) == set(feeds.keys())

        # Set the model's `feed_dict`, renaming the feed names
        # according to the model's inner names
        feed_dict.clear()
        feed_dict.update({tensor_name: feeds[feed_name]
                          for feed_name, tensor_name in feeds_to_tensors.items()})

        # Run the prediction and clean up so no other calls inadvertently use this `feed_dict`
        out = self.predict_on_batch(x)
        feed_dict.clear()
        return out


def gc_layer_with_placeholders(dim, name, gc_kwargs, inlayer):
    adj = keras.layers.Input(tensor=tf.placeholder(tf.float32, shape=(None, None),
                                                   name=name + '_adj'),
                             name=name + '_adj')
    gather = keras.layers.Input(tensor=tf.placeholder(tf.int32, shape=(None,),
                                                      name=name + '_gather'),
                                name=name + '_gather')
    gc = layers.GC(dim, name=name, **gc_kwargs)([adj, gather, inlayer])
    return [adj, gather], gc


def build_q(dims, use_bias=False):
    dim_data, dim_l1, dim_ξ = dims

    q_input = keras.layers.Input(shape=(dim_data,), name='q_input')
    # CANDO: change activation
    q_layer1_placeholders, q_layer1 = gc_layer_with_placeholders(
        dim_l1, 'q_layer1', {'use_bias': use_bias, 'activation': 'relu'}, q_input)
    q_μ_flat_placeholders, q_μ_flat = gc_layer_with_placeholders(
        dim_ξ, 'q_mu_flat', {'use_bias': use_bias}, q_layer1)
    q_logD_flat_placeholders, q_logD_flat = gc_layer_with_placeholders(
        dim_ξ, 'q_logD_flat', {'use_bias': use_bias}, q_layer1)
    q_u_flat_placeholders, q_u_flat = gc_layer_with_placeholders(
        dim_ξ, 'q_u_flat', {'use_bias': use_bias}, q_layer1)
    q_μlogDu_flat = keras.layers.Concatenate(name='q_mulogDu_flat')(
        [q_μ_flat, q_logD_flat, q_u_flat])
    q_model = Model(inputs=([q_input]
                            + q_layer1_placeholders
                            + q_μ_flat_placeholders
                            + q_logD_flat_placeholders
                            + q_u_flat_placeholders),
                    outputs=q_μlogDu_flat)

    return q_model, ('Gaussian',)


def build_p(dims, use_bias=False):
    dim_data, dim_l1, dim_ξ = dims

    p_input = keras.layers.Input(shape=(dim_ξ,), name='p_input')
    # CANDO: change activation
    p_layer1 = keras.layers.Dense(dim_l1, use_bias=use_bias, activation='relu',
                                  kernel_regularizer='l2', bias_regularizer='l2',
                                  name='p_layer1')(p_input)
    p_adj = layers.Bilinear(0, use_bias=use_bias,
                            kernel_regularizer='l2', bias_regularizer='l2',
                            name='p_adj')([p_layer1, p_layer1])
    p_v_μ_flat = keras.layers.Dense(dim_data, use_bias=use_bias,
                                    kernel_regularizer='l2', bias_regularizer='l2',
                                    name='p_v_mu_flat')(p_layer1)
    p_v_logD_flat = keras.layers.Dense(dim_data, use_bias=use_bias,
                                       kernel_regularizer='l2', bias_regularizer='l2',
                                       name='p_v_logD_flat')(p_layer1)
    p_v_u_flat = keras.layers.Dense(dim_data, use_bias=use_bias,
                                    kernel_regularizer='l2', bias_regularizer='l2',
                                    name='p_v_u_flat')(p_layer1)
    p_v_μlogDu_flat = keras.layers.Concatenate(name='p_v_mulogDu_flat')(
        [p_v_μ_flat, p_v_logD_flat, p_v_u_flat])
    p_model = Model(inputs=p_input, outputs=[p_adj, p_v_μlogDu_flat])

    return p_model, ('SigmoidBernoulli', 'Gaussian')


def build_vae(q_model_codecs, p_model_codecs, n_ξ_samples, loss_weights):
    """TODOC"""
    q, q_codecs = q_model_codecs
    assert len(q_codecs) == 1
    q_codec = q_codecs[0]
    del q_codecs
    p, p_codecs = p_model_codecs

    # Wire up the model
    ξ = layers.ParametrisedStochastic(q_codec, n_ξ_samples)(q.output)
    p_ξ = p(ξ)
    if not isinstance(p_ξ, list):
        p_ξ = [p_ξ]
    model = Model(inputs=q.input, outputs=[q.output] + p_ξ)

    # Compile the whole thing with losses
    model.compile('adam',  # CANDO: tune parameters
                  loss=([codecs.get_loss(q_codec, 'kl_to_normal_loss')]
                        + [codecs.get_loss(p_codec, 'estimated_pred_loss')
                           for p_codec in p_codecs]),
                  loss_weights=loss_weights,
                  feed_dict={},
                  # TODO: metrics
                  )

    return model, (q_codec,) + p_codecs


# Note: encoders are further regularised (Rezende et al. 2014, p.10):
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

# TODO:
# - we should weigh each contribution of decoders to the overall
#   loss according to its importance / number of contributing
#   items
# - minibatch (knowing that it will be model-specific
#   because of convolution and the like)
# - can the encoders incorporate side-features if the core
#   features are not distinctive enough?
