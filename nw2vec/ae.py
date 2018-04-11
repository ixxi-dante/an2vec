import keras

from nw2vec import layers
from nw2vec import codecs


def build_q(adj, dims):
    dim_data, dim_l1, dim_ξ = dims
    n_nodes = adj.shape[0]

    q_input = keras.layers.Input(batch_shape=(n_nodes, dim_data), name='q_input')
    # CANDO: change activation
    q_layer1 = layers.GC(dim_l1, adj, use_bias=True, activation='relu', name='q_layer1')(q_input)
    q_μ_flat = layers.GC(dim_ξ, adj, use_bias=True, name='q_mu_flat')(q_layer1)
    q_logD_flat = layers.GC(dim_ξ, adj, use_bias=True, name='q_logD_flat')(q_layer1)
    q_u_flat = layers.GC(dim_ξ, adj, use_bias=True, name='q_u_flat')(q_layer1)
    q_μlogDu_flat = keras.layers.Concatenate(name='q_mulogDu_flat')(
        [q_μ_flat, q_logD_flat, q_u_flat])
    q_model = keras.models.Model(inputs=q_input, outputs=q_μlogDu_flat)

    return q_model, ('Gaussian',)


def build_p(adj, dims):
    dim_data, dim_l1, dim_ξ = dims

    p_input = keras.layers.Input(shape=(dim_ξ,), name='p_input')
    # CANDO: change activation
    p_layer1 = keras.layers.Dense(dim_l1, use_bias=True, activation='relu',
                                  kernel_regularizer='l2', bias_regularizer='l2',
                                  name='p_layer1')(p_input)
    p_adj = layers.Bilinear(0, adj.shape[0], use_bias=False,
                            kernel_regularizer='l2', bias_regularizer='l2',
                            name='p_adj')([p_layer1, p_layer1])
    p_v_μ_flat = keras.layers.Dense(dim_data, use_bias=True,
                                    kernel_regularizer='l2', bias_regularizer='l2',
                                    name='p_v_mu_flat')(p_layer1)
    p_v_logD_flat = keras.layers.Dense(dim_data, use_bias=True,
                                       kernel_regularizer='l2', bias_regularizer='l2',
                                       name='p_v_logD_flat')(p_layer1)
    p_v_u_flat = keras.layers.Dense(dim_data, use_bias=True,
                                    kernel_regularizer='l2', bias_regularizer='l2',
                                    name='p_v_u_flat')(p_layer1)
    p_v_μlogDu_flat = keras.layers.Concatenate(name='p_v_mulogDu_flat')(
        [p_v_μ_flat, p_v_logD_flat, p_v_u_flat])
    p_model = keras.models.Model(inputs=p_input, outputs=[p_adj, p_v_μlogDu_flat])

    return p_model, ('SigmoidBernoulli', 'Gaussian')


def build_vae(q_model_codecs, p_model_codecs, n_ξ_samples, loss_weights, **kwargs):
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
    model = keras.models.Model(inputs=q.input, outputs=[q.output] + p_ξ)

    # Compile the whole thing with losses
    model.compile('adam',  # CANDO: tune parameters
                  loss=([codecs.get_loss(q_codec, 'kl_to_normal_loss')]
                        + [codecs.get_loss(p_codec, 'estimated_pred_loss')
                           for p_codec in p_codecs]),
                  loss_weights=loss_weights,
                  **kwargs,
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
