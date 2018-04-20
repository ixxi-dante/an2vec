from contextlib import contextmanager
import warnings

import keras
import tensorflow as tf

from keras.engine.topology import Layer
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras import callbacks as cbks

from nw2vec import layers
from nw2vec import codecs


class Model(keras.Model):

    def _get_feed_dict_and_translator(self):
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
            if hasattr(self, 'train_function'):
                raise ValueError(("Model {} has a `train_function` but "
                                  "`_function_kwargs` is absent or is `None`, "
                                  "or has no `feed_dict` dict. Most likely this "
                                  "model has been compiled or has run a "
                                  "training without a feed_dict.").format(self))
            if hasattr(self, 'test_function'):
                raise ValueError(("Model {} has a `test_function` but "
                                  "`_function_kwargs` is absent or is `None`, "
                                  "or has no `feed_dict` dict. Most likely this "
                                  "model has been compiled or has run a "
                                  "test without a feed_dict.").format(self))
            else:
                # No `feed_dict`, but also no `predict|train|test_function`,
                # so we can safely add a `_function_kwargs` with a `feed_dict`
                # and it will be used upon compilation of the `predict|train|test_function`
                # (which happens at the next call to one of the model.predict|train|test*
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

        return feed_dict, feeds_to_tensors

    @contextmanager
    def food(self, feeds):
        feed_dict, feeds_to_tensors = self._get_feed_dict_and_translator()

        # `feeds` should provide values for all the input tensors we found
        assert set(feeds_to_tensors.keys()) == set(feeds.keys())

        # Set the model's `feed_dict`, renaming the feed names
        # according to the model's inner names
        feed_dict.clear()
        feed_dict.update({tensor_name: feeds[feed_name]
                          for feed_name, tensor_name in feeds_to_tensors.items()})
        # Run the inner operation and clean up so no other calls inadvertently use this `feed_dict`
        yield
        feed_dict.clear()

    def train_on_fed_batch(self, x, y, feeds={}, **kwargs):
        with self.food(feeds):
            return self.train_on_batch(x, y, **kwargs)

    def predict_on_fed_batch(self, x, feeds={}):
        with self.food(feeds):
            return self.predict_on_batch(x)

    def fit_generator_feed(self,
                           generator,
                           steps_per_epoch=None,
                           epochs=1,
                           verbose=1,
                           callbacks=None,
                           validation_data=None,
                           validation_steps=None,
                           class_weight=None,
                           max_queue_size=10,
                           workers=1,
                           use_multiprocessing=False,
                           shuffle=True,
                           initial_epoch=0,
                           check_array_lengths=True):
        """Trains the model on data generated batch-by-batch by a Python generator
        or an instance of `Sequence`.

        See `Model.fig_generator()` for the full documentation.

        The only difference here is that the generator must also generate data for
        any native placeholders of the model.
        """
        # Disable validation, as we haven't converted the code for this yet.
        # All related code is commented with a `disabled:` prefix.
        if validation_data is not None:
            raise ValueError('Validation with a feeding generator is not yet supported')
        # The original (feed-modified) method starts here.

        wait_time = 0.01  # in seconds
        epoch = initial_epoch

        # disable: do_validation = bool(validation_data)
        self._make_train_function()
        # disable: if do_validation:
        # disable:     self._make_test_function()

        is_sequence = isinstance(generator, Sequence)
        if not is_sequence and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the`keras.utils.Sequence'
                            ' class.'))
        if steps_per_epoch is None:
            if is_sequence:
                steps_per_epoch = len(generator)
            else:
                raise ValueError('`steps_per_epoch=None` is only valid for a'
                                 ' generator based on the `keras.utils.Sequence`'
                                 ' class. Please specify `steps_per_epoch` or use'
                                 ' the `keras.utils.Sequence` class.')

        # disable: # python 2 has 'next', 3 has '__next__'
        # disable: # avoid any explicit version checks
        # disable: val_gen = (hasattr(validation_data, 'next') or
        # disable:            hasattr(validation_data, '__next__') or
        # disable:            isinstance(validation_data, Sequence))
        # disable: if (val_gen and not isinstance(validation_data, Sequence) and
        # disable:         not validation_steps):
        # disable:     raise ValueError('`validation_steps=None` is only valid for a'
        # disable:                      ' generator based on the `keras.utils.Sequence`'
        # disable:                      ' class. Please specify `validation_steps` or use'
        # disable:                      ' the `keras.utils.Sequence` class.')

        # Prepare display labels.
        out_labels = self.metrics_names
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        self.history = cbks.History()
        _callbacks = [cbks.BaseLogger(
            stateful_metrics=self.stateful_metric_names)]
        if verbose:
            _callbacks.append(
                cbks.ProgbarLogger(
                    count_mode='steps',
                    stateful_metrics=self.stateful_metric_names))
        _callbacks += (callbacks or []) + [self.history]
        callbacks = cbks.CallbackList(_callbacks)

        # it's possible to callback a different model than self:
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self
        callbacks.set_model(callback_model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            # disable: 'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        enqueuer = None
        # disable: val_enqueuer = None

        try:
            # disable: if do_validation and not val_gen:
            # disable:     # Prepare data for validation
            # disable:     if len(validation_data) == 2:
            # disable:         val_x, val_y = validation_data
            # disable:         val_sample_weight = None
            # disable:     elif len(validation_data) == 3:
            # disable:         val_x, val_y, val_sample_weight = validation_data
            # disable:     else:
            # disable:         raise ValueError('`validation_data` should be a tuple '
            # disable:                          '`(val_x, val_y, val_sample_weight)` '
            # disable:                          'or `(val_x, val_y)`. Found: ' +
            # disable:                          str(validation_data))
            # disable:     val_x, val_y, val_sample_weights = self._standardize_user_data(
            # disable:         val_x, val_y, val_sample_weight)
            # disable:     val_data = val_x + val_y + val_sample_weights
            # disable:     if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            # disable:         val_data += [0.]
            # disable:     for cbk in callbacks:
            # disable:         cbk.validation_data = val_data

            if workers > 0:
                if is_sequence:
                    enqueuer = OrderedEnqueuer(generator,
                                               use_multiprocessing=use_multiprocessing,
                                               shuffle=shuffle)
                else:
                    enqueuer = GeneratorEnqueuer(generator,
                                                 use_multiprocessing=use_multiprocessing,
                                                 wait_time=wait_time)
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enqueuer.get()
            else:
                if is_sequence:
                    output_generator = iter(generator)
                else:
                    output_generator = generator

            callback_model.stop_training = False
            # Construct epoch logs.
            epoch_logs = {}
            while epoch < epochs:
                for m in self.metrics:
                    if isinstance(m, Layer) and m.stateful:
                        m.reset_states()
                callbacks.on_epoch_begin(epoch)
                steps_done = 0
                batch_index = 0
                while steps_done < steps_per_epoch:
                    generator_output = next(output_generator)

                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, feeds, sample_weight)` '
                                         'or `(x, y, feeds)`. Found: ' +
                                         str(generator_output))

                    if len(generator_output) == 3:
                        x, y, feeds = generator_output
                        sample_weight = None
                    elif len(generator_output) == 4:
                        x, y, feeds, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, feeds, sample_weight)` '
                                         'or `(x, y, feeds)`. Found: ' +
                                         str(generator_output))
                    # build batch logs
                    batch_logs = {}
                    if x is None or len(x) == 0:
                        # Handle data tensors support when no input given
                        # step-size = 1 for data tensors
                        batch_size = 1
                    elif isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]
                    batch_logs['batch'] = batch_index
                    batch_logs['size'] = batch_size
                    callbacks.on_batch_begin(batch_index, batch_logs)

                    outs = self.train_on_fed_batch(x, y, feeds=feeds,
                                                   sample_weight=sample_weight,
                                                   class_weight=class_weight,
                                                   check_array_lengths=check_array_lengths)

                    if not isinstance(outs, list):
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_index, batch_logs)

                    batch_index += 1
                    steps_done += 1

                    # Epoch finished.
                    # disable: if steps_done >= steps_per_epoch and do_validation:
                    # disable:     if val_gen:
                    # disable:         val_outs = self.evaluate_generator(
                    # disable:             validation_data,
                    # disable:             validation_steps,
                    # disable:             workers=workers,
                    # disable:             use_multiprocessing=use_multiprocessing,
                    # disable:             max_queue_size=max_queue_size)
                    # disable:     else:
                    # disable:         # No need for try/except because
                    # disable:         # data has already been validated.
                    # disable:         val_outs = self.evaluate(
                    # disable:             val_x, val_y,
                    # disable:             batch_size=batch_size,
                    # disable:             sample_weight=val_sample_weights,
                    # disable:             verbose=0)
                    # disable:     if not isinstance(val_outs, list):
                    # disable:         val_outs = [val_outs]
                    # disable:     # Same labels assumed.
                    # disable:     for l, o in zip(out_labels, val_outs):
                    # disable:         epoch_logs['val_' + l] = o

                    if callback_model.stop_training:
                        break

                callbacks.on_epoch_end(epoch, epoch_logs)
                epoch += 1
                if callback_model.stop_training:
                    break

        finally:
            try:
                if enqueuer is not None:
                    enqueuer.stop()
            finally:
                pass
                # disable: if val_enqueuer is not None:
                # disable:     val_enqueuer.stop()

        callbacks.on_train_end()
        return self.history


def gc_layer_with_placeholders(dim, name, gc_kwargs, inlayer):
    adj = keras.layers.Input(tensor=tf.placeholder(tf.float32, shape=(None, None),
                                                   name=name + '_adj'),
                             name=name + '_adj')
    mask = keras.layers.Input(tensor=tf.placeholder(tf.float32, shape=(None,),
                                                    name=name + '_output_mask'),
                              name=name + '_output_mask')
    gc = layers.GC(dim, name=name, **gc_kwargs)([adj, mask, inlayer])
    return [adj, mask], gc


def build_q(dims, use_bias=False):
    dim_data, dim_l1, dim_ξ = dims

    q_input = keras.layers.Input(shape=(dim_data,), name='q_input')
    # CANDO: change activation
    q_layer1_placeholders, q_layer1 = gc_layer_with_placeholders(
        dim_l1, 'q_layer1', {'use_bias': use_bias, 'activation': 'relu'}, q_input)
    q_μ_flat_placeholders, q_μ_flat = gc_layer_with_placeholders(
        dim_ξ, 'q_mu_flat', {'use_bias': use_bias, 'gather_mask': True}, q_layer1)
    q_logD_flat_placeholders, q_logD_flat = gc_layer_with_placeholders(
        dim_ξ, 'q_logD_flat', {'use_bias': use_bias, 'gather_mask': True}, q_layer1)
    q_u_flat_placeholders, q_u_flat = gc_layer_with_placeholders(
        dim_ξ, 'q_u_flat', {'use_bias': use_bias, 'gather_mask': True}, q_layer1)
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

    return p_model, ('SigmoidBernoulliAdjacency', 'Gaussian')


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
