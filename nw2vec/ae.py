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
    # CANDO: change activation
    q_layer1 = layers.GC(dim_l1, adj, use_bias=True, activation='relu', name='q_layer1')(q_input)
    q_μ_flat = layers.GC(dim_z, adj, use_bias=True, name='q_mu_flat')(q_layer1)
    q_logD_flat = layers.GC(dim_z, adj, use_bias=True, name='q_logD_flat')(q_layer1)
    q_u_flat = layers.GC(dim_z, adj, use_bias=True, name='q_u_flat')(q_layer1)
    q_μlogDu_flat = keras.layers.Concatenate(name='q_mulogDu_flat')(
        [q_μ_flat, q_logD_flat, q_u_flat])
    q_model = keras.models.Model(inputs=q_input, outputs=q_μlogDu_flat)

    return q_input, q_model, codecs.Gaussian


def build_p_model(adj, dims):
    dim_data, dim_l1, dim_z = dims

    p_input = keras.layers.Input(shape=(dim_z,), name='p_input')
    # CANDO: change activation
    p_layer1 = keras.layers.Dense(dim_l1, use_bias=True, activation='relu',
                                  kernel_regularizer='l2', bias_regularizer='l2',
                                  name='p_layer1')(p_input)
    p_adj = layers.Bilinear(0, adj.shape[0], use_bias=False, activation='sigmoid',
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

    return p_input, p_model, (codecs.Bernoulli, codecs.Gaussian)


def vae(adj, q_input, q_model, q_codec, n_ξ_samples, p_model, p_codecs):
    # Wire up the model
    q = q_model(q_input)
    ξ = layers.ParametrisedStochastic(q_codec, n_ξ_samples)(q)
    model = keras.models.Model(inputs=q_input, outputs=[q] + p_model(ξ))

    # Define the losses
    def q_loss(_, pred):
        return codecs.get(q_codec, pred).kl_to_normal()

    p_adj_codec, p_v_codec = p_codecs

    def estimated_p_adj_loss(adj_true, adj_pred):
        return - K.mean(codecs.get(p_adj_codec, adj_pred).logprobability(adj_true), axis=1)

    def estimated_p_v_loss(v_true, v_pred):
        return - K.mean(codecs.get(p_v_codec, v_pred).logprobability(v_true), axis=1)

    # Compile the whole thing with losses
    model.compile('adam',  # CANDO: tune parameters
                  loss=[q_loss, estimated_p_adj_loss, estimated_p_v_loss],
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
