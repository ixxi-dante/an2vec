import keras
from keras import backend as K

from nw2vec import layers
from nw2vec import utils


# Set up the VAE

dim_data = 40    # Dimension base features for each node
dim_l1 = 20   # Dimension of intermediate layers
dim_z = 5    # Dimension of the embedding space
n_ξ_samples = 10
adj = []  # Adjacency matrix for convolutions

# Encoder tensor
q_input = keras.layers.Input(shape=(dim_data,), name='q_input')
# CANDO: change act
q_layer1 = layers.GC(dim_l1, adj, use_bias=True, act='relu', name='q_layer1')(q_input)
q_μ_flat = layers.GC(dim_z, adj, use_bias=True, name='q_μ_flat')(q_layer1)
q_logD_flat = layers.GC(dim_z, adj, use_bias=True, name='q_logD_flat')(q_layer1)
q_u_flat = layers.GC(dim_z, adj, use_bias=True, name='q_u_flat')(q_layer1)
q_μlogDu_flat = keras.layers.Concatenation(name='q_μlogDu_flat')([q_μ_flat, q_logD_flat, q_u_flat])
q_model = keras.models.Model(inputs=q_input, outputs=q_μlogDu_flat)

# Decoder model
p_input = keras.layers.Input(shape=(dim_z,), name='p_input')
# CANDO: change act
p_layer1 = keras.layers.Dense(dim_l1, use_bias=True, act='relu', name='p_layer1')(p_input)
p_adj = layers.Bilinear(adj.shape[0], name='p_adj')(p_layer1)
p_v_μ_flat = keras.layers.Dense(dim_data, use_bias=True, name='p_v_μ_flat')(p_layer1)
p_v_logD_flat = keras.layers.Dense(dim_data, use_bias=True, name='p_v_logD_flat')(p_layer1)
p_v_u_flat = keras.layers.Dense(dim_data, use_bias=True, name='p_v_u_flat')(p_layer1)
p_v_μlogDu_flat = keras.layers.Concatenation(name='p_v_μlogDu_flat')(
    [p_v_μ_flat, p_v_logD_flat, p_v_u_flat])
p_model = keras.models.Model(inputs=p_input, outputs=[p_adj, p_v_μlogDu_flat])


# ADD
# - encoder KL to normal
# - decoder regularizer ||θ||**2 / (2 * κ)


def vae(q_input, q_model, q_codec, n_ξ_samples, p_model, p_codecs):
    ξ = layers.ParametrisedStochastic(q_codec, n_ξ_samples)(q_model(q_input))
    model = keras.models.Model(inputs=q_input, outputs=p_model(ξ))

    def estimated_codec_loss(codec, y_true, y_pred):
        y_true = utils.expand_dims_tile(y_true, -2, y_pred.shape[-2])
        return K.mean(codec(y_pred).logprobability(y_true), axis=-2)

    model.compile('adam',  # CANDO: tune parameters
                  losses=[estimated_codec_loss(codec) for codec in p_codecs],
                  # TODO: loss_weights
                  # TODO: metrics
                  )
    return model


# decoders could be made of decoder_adj and decoder_features.
# decoder_adj should be Bernoulli, decoder_features should be
# Gaussian, and they could also share a first layer. They can be
# bilinear or MLP

# encoders are further regularised (Rezende et al. 2014, p.10):
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

# QUESTIONS:
#
# - what obective to use, and how does it depend on
# encoder/decoder
#
# - we should weigh each contribution of decoders to the overall
# loss according to its importance / number of contributing
# items
#
# - how much help does TF need for gradient descent
#
# - how do we minibatch (knowing that it will be model-specific
# because of convolution and the like)
#
# - can the encoders incorporate side-features if the core
# features are not distinctive enough?
