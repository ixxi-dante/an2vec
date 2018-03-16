import keras
from keras import backend as K

from nw2vec import layers
from nw2vec import utils
from nw2vec import codecs


# Example values for the parameters
# dim_data = 40    # Dimension base features for each node
# dim_l1 = 20      # Dimension of intermediate layers
# dim_z = 5        # Dimension of the embedding space
# n_ξ_samples = 10


def build_q_model(adj, dims):
    dim_data, dim_l1, dim_z = dims

    q_input = keras.layers.Input(shape=(dim_data,), name='q_input')
    # CANDO: change act
    # q_layer1 = layers.GC(dim_l1, adj, use_bias=True, act='relu', name='q_layer1')(q_input)
    # q_μ_flat = layers.GC(dim_z, adj, use_bias=True, name='q_μ_flat')(q_layer1)
    # q_logD_flat = layers.GC(dim_z, adj, use_bias=True, name='q_logD_flat')(q_layer1)
    # q_u_flat = layers.GC(dim_z, adj, use_bias=True, name='q_u_flat')(q_layer1)
    q_layer1 = keras.layers.Dense(dim_l1, adj, use_bias=True, act='relu',
                                  name='q_layer1')(q_input)
    q_μ_flat = keras.layers.Dense(dim_z, adj, use_bias=True, name='q_μ_flat')(q_layer1)
    q_logD_flat = keras.layers.Dense(dim_z, adj, use_bias=True, name='q_logD_flat')(q_layer1)
    q_u_flat = keras.layers.Dense(dim_z, adj, use_bias=True, name='q_u_flat')(q_layer1)
    q_μlogDu_flat = keras.layers.Concatenation(name='q_μlogDu_flat')(
        [q_μ_flat, q_logD_flat, q_u_flat])
    q_model = keras.models.Model(inputs=q_input, outputs=q_μlogDu_flat)

    return q_input, q_model, codecs.Gaussian


def build_p_model(dims):
    dim_data, dim_l1, dim_z = dims

    p_input = keras.layers.Input(shape=(dim_z,), name='p_input')
    # CANDO: change act
    p_layer1 = keras.layers.Dense(dim_l1, use_bias=True, act='relu',
                                  kernel_regularizer='l2', bias_regularizer='l2',
                                  name='p_layer1')(p_input)
    p_adj = layers.Bilinear(0, use_bias=False, act='sigmoid',
                            kernel_regularizer='l2', bias_regularizer='l2',
                            name='p_adj')(p_layer1)
    p_v_μ_flat = keras.layers.Dense(dim_data, use_bias=True,
                                    kernel_regularizer='l2', bias_regularizer='l2',
                                    name='p_v_μ_flat')(p_layer1)
    p_v_logD_flat = keras.layers.Dense(dim_data, use_bias=True,
                                       kernel_regularizer='l2', bias_regularizer='l2',
                                       name='p_v_logD_flat')(p_layer1)
    p_v_u_flat = keras.layers.Dense(dim_data, use_bias=True,
                                    kernel_regularizer='l2', bias_regularizer='l2',
                                    name='p_v_u_flat')(p_layer1)
    p_v_μlogDu_flat = keras.layers.Concatenation(name='p_v_μlogDu_flat')(
        [p_v_μ_flat, p_v_logD_flat, p_v_u_flat])
    p_model = keras.models.Model(inputs=p_input, outputs=[p_adj, p_v_μlogDu_flat])

    return p_input, p_model, (codecs.Bernoulli, codecs.Gaussian)


def vae(adj, q_input, q_model, q_codec, n_ξ_samples, p_model, p_codecs):
    # Wire up the model
    q = q_model(q_input)
    ξ = layers.ParametrisedStochastic(q_codec, n_ξ_samples)(q)
    model = keras.models.Model(inputs=q_input, outputs=[q] + p_model(ξ))

    # Define the losses
    def q_loss(y_true, y_pred):
        return codecs.get(q_codec, y_pred).kl_to_normal()

    p_adj_codec, p_v_codec = p_codecs

    def estimated_p_adj_loss(y_true, y_pred):
        batch_size = y_true.shape[0]
        assert adj.shape[0] == batch_size
        assert y_pred.shape == (n_ξ_samples, batch_size)
        y_pred_flat = K.reshape(y_pred, (n_ξ_samples, batch_size ** 2))
        sample_adj_flat = utils.expand_dims_tile(adj.flatten(), 0, n_ξ_samples)
        global_logprob = K.mean(codecs.get(p_adj_codec, y_pred_flat)
                                .logprobability(sample_adj_flat),
                                axis=-2)
        return - utils.expand_dims_tile(global_logprob, 0, batch_size)

    def estimated_p_v_loss(y_true, y_pred):
        y_true = utils.expand_dims_tile(y_true, -2, y_pred.shape[-2])
        return - K.mean(codecs.get(p_v_codec, y_pred).logprobability(y_true), axis=-2)

    # Compile the whole thing with losses
    model.compile('adam',  # CANDO: tune parameters
                  losses=[q_loss, estimated_p_adj_loss, estimated_p_v_loss],
                  loss_weights=[1.0, 1.0, 1.0],  # TODO: tune loss_weights
                  # TODO: metrics
                  )
    return model


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
