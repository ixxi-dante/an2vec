import keras
from keras import backend as K

from nw2vec import layers
from nw2vec import codecs


class ModelIO:

    def __init__(self, input, model, codecs):
        assert isinstance(codecs, (list, tuple))
        assert len(model.outputs) == len(codecs)
        self.input = input
        self.model = model
        self.codecs = tuple(codecs)

    @property
    def codec(self):
        assert len(self.codecs) == 1
        return self.codecs[0]

    @staticmethod
    def _kl_to_normal_loss_fn(codec):

        def loss_fn(y_true, y_pred):
            return codecs.get(codec, y_pred).kl_to_normal()

        return loss_fn

    def kl_to_normal_losses(self):
        return [self._kl_to_normal_loss_fn(codec) for codec in self.codecs]

    @staticmethod
    def _estimated_pred_loss_fn(codec):

        def loss_fn(y_true, y_pred):
            return - K.mean(codecs.get(codec, y_pred).logprobability(y_true), axis=1)

        return loss_fn

    def estimated_pred_losses(self):
        return [self._estimated_pred_loss_fn(codec) for codec in self.codecs]


def build_q_io(adj, dims):
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

    return ModelIO(q_input, q_model, (codecs.Gaussian,))


def build_p_io(adj, dims):
    dim_data, dim_l1, dim_z = dims

    p_input = keras.layers.Input(shape=(dim_z,), name='p_input')
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

    return ModelIO(p_input, p_model, (codecs.SigmoidBernoulli, codecs.Gaussian))


def build_vae_io(adj, q_io, p_io, n_ξ_samples, loss_weights):
    # Wire up the model
    ξ_params = q_io.model(q_io.input)
    ξ = layers.ParametrisedStochastic(q_io.codec, n_ξ_samples)(ξ_params)
    model = keras.models.Model(inputs=q_io.input, outputs=[ξ_params, p_io.model(ξ)])

    # Compile the whole thing with losses
    model.compile('adam',  # CANDO: tune parameters
                  loss=q_io.kl_to_normal_losses() + p_io.estimated_pred_losses(),
                  loss_weights=loss_weights,
                  # TODO: metrics
                  )

    return ModelIO(q_io.input, model, (q_io.codec,) + p_io.codecs)


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
