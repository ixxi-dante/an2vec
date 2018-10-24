from contextlib import contextmanager
import warnings
import itertools

import keras
from keras import backend as K
import tensorflow as tf

from keras.engine.topology import Layer
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras import callbacks as cbks
import numpy as np

from nw2vec import layers
from nw2vec import codecs


class ModelBatchCheckpoint(cbks.Callback):
    """Save the model after every batch.

    Copied and extended from keras.callbacks.ModelCheckpoint.

    `filepath` can contain named formatting options,
    which will be filled the value of `batch` and
    keys in `logs` (passed in `on_batch_end`).
    For example: if `filepath` is `weights.{batch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the batch number and
    the validation loss in the filename.

    Parameters
    ----------
    filepath: string, path to save the model file.
    monitor: quantity to monitor.
    verbose: verbosity mode, 0 or 1.
    save_best_only: if `save_best_only=True`,
        the latest best model according to
        the quantity monitored will not be overwritten.
    mode: one of {auto, min, max}.
        If `save_best_only=True`, the decision
        to overwrite the current save file is made
        based on either the maximization or the
        minimization of the monitored quantity. For `val_acc`,
        this should be `max`, for `val_loss` this should
        be `min`, etc. In `auto` mode, the direction is
        automatically inferred from the name of the monitored quantity.
    save_weights_only: if True, then only the model's weights will be
        saved (`model.save_weights(filepath)`), else the full model
        is saved (`model.save(filepath)`).
    period: Interval (number of batches) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelBatchCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.batches_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if 'batch' in logs:
            assert logs.pop('batch') == batch
        self.batches_since_last_save += 1
        if self.batches_since_last_save >= self.period:
            self.batches_since_last_save = 0
            filepath = self.filepath.format(epoch=self.current_epoch + 1, batch=batch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d / Batch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (self.current_epoch + 1, batch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d / Batch %05d: %s did not improve from %0.5f' %
                                  (self.current_epoch + 1, batch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d / Batch %05d: saving model to %s' %
                          (self.current_epoch + 1, batch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class Model(keras.Model):

    """An augmented version of `keras.Model` which knows how to handle custom
    fullbatching and minibatching.

    Parameters
    ----------
    inputs, outputs : normal input/output layers or tensors for `keras.Model`.
    fullbatcher : generator
        The generator used to train on full batches.
    minibatcher : generator
        The generator used to train on mini-batches.
    name : string
        The name of the model.

    """

    def __init__(self, inputs, outputs, fullbatcher=None, minibatcher=None, name=None):
        """Set the fullbatcher and minibatcher, then delegate to keras.Model.__init__."""
        self.fullbatcher = fullbatcher
        self.minibatcher = minibatcher
        return super(Model, self).__init__(inputs, outputs, name=name)

    def _get_feed_dict_and_translator(self):
        """Get the model's feed_dict and a map of feed names to placeholder tensor names.

        The feed_dict returned here is what `self.food()` uses; see that method
        for more details.

        The feed name to tensor name map simply lets us convert feed names, which
        correspond to the layer names of the model (see how the batcher functions
        generate feeds, in nw2vec.batching), into the underlying TF tensor names,
        which are slightly more precise than just the layer name.

        """

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
            tensor = feed_layer._inbound_nodes[0].input_tensors[0]
            if isinstance(tensor, tf.SparseTensor):
                feeds_to_tensors[feed_layer.name + '/indices'] = tensor.indices.name
                feeds_to_tensors[feed_layer.name + '/values'] = tensor.values.name
                feeds_to_tensors[feed_layer.name + '/dense_shape'] = tensor.dense_shape.name
            else:
                feeds_to_tensors[feed_layer.name] = tensor.name

        return feed_dict, feeds_to_tensors

    @contextmanager
    def food(self, feeds):
        """Abstract over our hack to provide placeholder feeds to keras training.

        Keras gives no direct access to the feed_dict dicts used when evaluating
        its models, so changing its values across batches (which is necessary for
        minibatching) is complicated.

        Our solution is to give the model an empty feed_dict upon compilation
        (see `build_vae()` below in this file), which is stored by Keras in the
        model, and which we can recover and modify whenever we want to set
        placeholder feeds for a particular evaluation.

        This method uses this hack to provide a handy context manager that
        you can use to feed placeholder values simply:

        ```
        with model.food(feeds):
            model.train_on_batch(x, y)
        ```

        """

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
        """Train on a batch with placeholder feeds."""
        with self.food(feeds):
            return self.train_on_batch(x, y, **kwargs)

    def predict_fullbatch(self, **kwargs):
        """Predict values on a full batch.

        The kwargs are passed to the fullbatcher function, so must provide
        what that function expects (see `nw2vec.batching.fullbatches()`).

        """

        if self.fullbatcher is None:
            raise ValueError("Model has no fullbatcher")

        x, _, feeds = next(self.fullbatcher(self, **kwargs))
        return self.predict_on_fed_batch(x, feeds=feeds)

    def predict_minibatches(self, **kwargs):
        """Predict values on a set of minibatches.

        The kwargs are passed to the minibatcher function, so must provide
        what that function expects (see `nw2vec,batching.pq_batches()').

        Additionally, kwargs must provide a `steps_per_epoch` key which
        defines how many minibatches to predict for.

        """

        if self.minibatcher is None:
            raise ValueError("Model has no minibatcher")

        steps_per_epoch = kwargs.pop('steps_per_epoch')

        out = []
        for x, _, feeds in itertools.islice(self.minibatcher(self, **kwargs),
                                            steps_per_epoch):
            out.append(self.predict_on_fed_batch(x, feeds=feeds))
        return out

    def predict_on_fed_batch(self, x, feeds={}):
        """Predict on a batch with placeholder feeds."""
        with self.food(feeds):
            return self.predict_on_batch(x)

    def fit_fullbatches(self, batcher_kws={}, shuffle=False, check_array_lengths=False, **kwargs):
        """Fit the model using the fullbatch generator.

        `batcher_kws` points the fullbatcher to the actual data to train on
        (which it uses to create the batches fed to the model). This function
        simply calls the fullbatcher and delegates to `self.fit_generator_feed()`,
        which receives all the `kwargs`. Read that method's documentation, as well
        as `keras.Model.fit_generator()` for details on available arguments.

        `shuffle` and `check_array_lengths` are here because they must be set to False
        for this technique to work properly (keras must not reshuffle our batches,
        and must not enforce too strict constraints on the shapes of the arrays produced
        by the batch generator.

        """

        return self.fit_generator_feed(self.fullbatcher(self, **batcher_kws), steps_per_epoch=1,
                                       shuffle=shuffle, check_array_lengths=check_array_lengths,
                                       **kwargs)

    def fit_minibatches(self, batcher_kws={}, shuffle=False, check_array_lengths=False, **kwargs):
        """Fit the model using the minibatch generator.

        `batcher_kws` points the minibatcher to the actual data to train on
        (which it uses to create the batches fed to the model). This function
        simply calls the minibatcher and delegates to `self.fit_generator_feed()`,
        which receives all the `kwargs`. Read that method's documentation, as well
        as `keras.Model.fit_generator()` for details on available arguments.

        `shuffle` and `check_array_lengths` are here because they must be set to False
        for this technique to work properly (keras must not reshuffle our batches,
        and must not enforce too strict constraints on the shapes of the arrays produced
        by the batch generator.

        """
        return self.fit_generator_feed(self.minibatcher(self, **batcher_kws),
                                       shuffle=shuffle, check_array_lengths=check_array_lengths,
                                       **kwargs)

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
        """Train the model on data generated batch-by-batch by a Python generator
        or an instance of `Sequence`.

        See `Model.fit_generator()` for the full documentation.

        The only difference here is that the generator must also generate data for
        any native placeholders of the model.

        Only use this if you know what you are doing (especially with the `shuffle`
        and `check_array_lengths` parameters). If not, prefer `self.fit_fullbatches()`
        or `self.fit_minibatches()`.

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
    """Generate a GC layer and its placeholders, and apply the GC layer to an input.

    Parameters
    ----------
    dim : int
        Output dimension of the GC layer (number of units).
    name : string
        Name of the layer.
    gc_kwargs : dict
        Keyword arguments passed to `nw2vec.layers.GC()` (such as
        `use_bias`, `activation` and so on).
    inlayer : keras tensor
        Input to the GC layer

    Returns
    -------
    placeholders : list of keras tensors
        List of placeholder tensors necessary for the GC layer.
    gc : keras tensor
        Output of the GC layer.

    """

    adj = keras.layers.Input(tensor=tf.sparse_placeholder(K.floatx(), shape=(None, None),
                                                          name=name + '_adj'),
                             sparse=True,
                             name=name + '_adj')
    mask = keras.layers.Input(tensor=tf.placeholder(K.floatx(), shape=(None,),
                                                    name=name + '_output_mask'),
                              name=name + '_output_mask')
    gc = layers.GC(dim, name=name, **gc_kwargs)([adj, mask, inlayer])
    return [adj, mask], gc


def build_q(dims, overlap=0, use_bias=False, fullbatcher=None, minibatcher=None):
    """Build the encoder part of a VAE.

    Parameters
    ----------
    dims : tuple of ints
        Tuple containing (dim_data, dim_l1, dim_ξa, dim_ξb), which are the dimensions
        of the successive layers in the encoder. `dim_data` is the dimension of the
        input features. `dim_l1` the dimension that the first layer outputs. `dim_ξa`
        and `dim_ξb` are the dimensions of the two parallel layers that are then combined
        to form the embeddings (each of these layers is in fact double, in order to
        generate μ and log(Σ), the parameters of the Gaussians from which we sample
        to create embeddings)
    overlap : int, optional
        Size of the overlap between the parallel layers which are combined to form the embeddings.
        The output of this encoder model will be of dimension `2 * (dim_ξa + dim_ξb - overlap)`
        (the multiplication by 2 is because both μ and log(Σ) are generated, with Σ diagonal.
        Defaults to 0.
    use_bias : bool, optional
        Whether or not to use bias in each of the layers of the encoder. Defaults to False.
    fullbatcher : generator, optional
        Generator used by the model to create fullbatches. Defaults to None, in which case
        the model will try to use the raw data when training and predicting (and this will
        not work given the way training is currently wired up, with a `target_func()` and all).
    minibatcher : generator, optional
        Generator used by the model to create minibatches. Defaults to None, in which case
        the model will fail to train or predict on minibatechs.

    Returns
    -------
    q_model : nw2vec.ae.Model
        The encoder model.
    q_codecs : tuple of strings
        Tuple of codec names used to interpret the encoder output. Currently a single codec.

    """

    # Extract arguments and check them.
    dim_data, dim_l1, dim_ξa, dim_ξb = dims
    assert overlap <= min(dim_ξa, dim_ξb)

    # Input layer which receives the node features.
    q_input = keras.layers.Input(shape=(dim_data,), name='q_input')

    # First layer: generate placeholders and output tensor
    # Placeholders are used to feed each layer in the model with adjacency matrices
    # and output masks, which are necessary for minibatching.
    q_layer1_placeholders, q_layer1 = gc_layer_with_placeholders(
        dim_l1, 'q_layer1', {'use_bias': use_bias, 'activation': 'relu'}, q_input)

    # Parallel second layers: generate placeholders for the parallel μ and log(Σ) values.
    q_μa_flat_placeholders, q_μa_flat = gc_layer_with_placeholders(
        dim_ξa, 'q_mua_flat', {'use_bias': use_bias, 'gather_mask': True}, q_layer1)
    q_logSa_flat_placeholders, q_logSa_flat = gc_layer_with_placeholders(
        dim_ξa, 'q_logSa_flat', {'use_bias': use_bias, 'gather_mask': True}, q_layer1)
    q_μb_flat_placeholders, q_μb_flat = gc_layer_with_placeholders(
        dim_ξb, 'q_mub_flat', {'use_bias': use_bias, 'gather_mask': True}, q_layer1)
    q_logSb_flat_placeholders, q_logSb_flat = gc_layer_with_placeholders(
        dim_ξb, 'q_logSb_flat', {'use_bias': use_bias, 'gather_mask': True}, q_layer1)

    # Concatenate the output of the previous parallel layers into a single tensor, with overlap.
    # The values that are overlapped are combined using the `reducer` argument
    # (can be `mean`, `max`, `min`, `sum`).
    q_μ_flat = layers.OverlappingConcatenate(name='q_mu_flat', overlap_size=overlap,
                                             reducer='mean')([q_μa_flat, q_μb_flat])
    q_logS_flat = layers.OverlappingConcatenate(name='q_logS_flat', overlap_size=overlap,
                                                reducer='mean')([q_logSa_flat, q_logSb_flat])

    # Concatenate μ and log(Σ) into a single tensor, which is the output of the model.
    q_μlogS_flat = keras.layers.Concatenate(name='q_mulogS_flat')([q_μ_flat, q_logS_flat])

    # Actually create the model, using all the placeholders created along the way.
    q_model = Model(inputs=([q_input]
                            + q_layer1_placeholders
                            + q_μa_flat_placeholders
                            + q_logSa_flat_placeholders
                            + q_μb_flat_placeholders
                            + q_logSb_flat_placeholders),
                    outputs=q_μlogS_flat,
                    fullbatcher=fullbatcher,
                    minibatcher=minibatcher)

    # Return the model and the name of the codec that should be used to interpret
    # the output of the model.
    return q_model, ('OrthogonalGaussian',)


def build_p_builder(dims, feature_codec='SigmoidBernoulli', adj_kernel=None, use_bias=False,
                    embedding_slices=None, with_l1=True, share_l1=True):
    """Create a function that applies the decoder to an input tensor.

    Parameters
    ----------
    dims : tuple of 4 ints
        The first two elements are `dim_data` (the dimension of the reconstructed output
        features) and `dim_l1` (the output dimension of the intermediate layer, if it
        is requested by 'with_l1'). The last two elements are ignored (but are usually
        the adj and feature embedding dimensions, as passed to `build_q()`).
    feature_codec : string, optional
        Name of the codec used to interpret the feature reconstruction. Must be one of
        'OrthogonalGaussian' or 'SigmoidBernoulli'. Defaults to 'SigmoidBernoulli'.
    adj_kernel : (dim_l1, dim_l1) or (dim_ξ_adj, dim_ξ_adj) 2D ndarray, optional
        If provided, represents a fixed kernel that will be used in the Bilinear adjacency
        reconstruction layer (in which case it must match the dimensions of its input,
        be it the adjacency embedding if `with_l1 == False`, or `dim_l1` if `with_l1 == True`).
        If not provided (which is the default), that kernel will be trained over.
    use_bias : bool, optional
        Whether or not to use bias to each of the layers in the decoder. Defaults to False.
    embedding_slices : 2-tuple of slice objects, optional
        The first slice defines which dimensions of the input to the decoder (i.e. the embeddings)
        are used for adjacency reconstruction. The second slice defines which dimensions are
        used for feature reconstruction. Defaults to (slice(None), slice(None)), which
        corresponds to using all embedding dimensions for both adjacency and feature
        reconstruction.
    with_l1 : bool, optional
        Whether or not to use an intermediate layer in decoding. Defaults to True.
    share_l1 : bool, optional
        If `with_l1=True`, this defines whether the adjacency flow and the feature
        flow share the same weights for their intermediate decoding layer. Defaults to False.

    Returns
    -------
    p_builder : function
        A function which, given an input tensor, produces the decoder's output on that tensor,
        and the codecs necessary to interpret those outputs. In practice, give this function
        to `build_vae()`, which will call it on the sampled embeddings.

    """

    # Validate arguments and set default values
    assert feature_codec in ['SigmoidBernoulli', 'OrthogonalGaussian', 'SoftmaxMultinomial']
    if embedding_slices is None:
        adj_embedding_slice, v_embedding_slice = [slice(None), slice(None)]
    else:
        adj_embedding_slice, v_embedding_slice = embedding_slices

    # Extract the dimensions we use.
    dim_data, dim_l1, _, _ = dims

    def p_builder(p_input):
        # Get slices of the embeddings for each prediction
        p_input_adj = layers.InnerSlice(adj_embedding_slice)(p_input)
        p_input_v = layers.InnerSlice(v_embedding_slice)(p_input)

        # Build layer1 for adj and features, if requested, and have
        # it shared between adjacency and features if requested.
        if share_l1:
            # Shared intermediate layer.
            assert with_l1, "Can't share l1 if l1 is not requested"
            p_layer1 = keras.layers.Dense(dim_l1, use_bias=use_bias, activation='relu',
                                          kernel_regularizer='l2', bias_regularizer='l2',
                                          name='p_layer1')
            p_penultimate_adj = p_layer1(p_input_adj)
            p_penultimate_v = p_layer1(p_input_v)
        else:
            if with_l1:
                # Unshared intermediate layer.
                p_penultimate_adj = keras.layers.Dense(
                    dim_l1, use_bias=use_bias, activation='relu',
                    kernel_regularizer='l2', bias_regularizer='l2',
                    name='p_layer1_adj'
                )(p_input_adj)
                p_penultimate_v = keras.layers.Dense(
                    dim_l1, use_bias=use_bias, activation='relu',
                    kernel_regularizer='l2', bias_regularizer='l2',
                    name='p_layer1_v'
                )(p_input_v)
            else:
                # No intermediate layer.
                p_penultimate_adj = p_input_adj
                p_penultimate_v = p_input_v

        # Prepare kwargs for the Bilinear adj decoder, then build it.
        adj_kwargs = {}
        if adj_kernel is not None:
            adj_kwargs['fixed_kernel'] = adj_kernel
        else:
            adj_kwargs['kernel_regularizer'] = 'l2'
        p_adj = layers.Bilinear(0, use_bias=use_bias, name='p_adj',
                                bias_regularizer='l2',
                                **adj_kwargs)([p_penultimate_adj, p_penultimate_adj])

        # Finally build the feature decoder according to the requested codec.
        if feature_codec in ['SigmoidBernoulli', 'SoftmaxMultinomial']:
            p_v = keras.layers.Dense(dim_data, use_bias=use_bias,
                                     kernel_regularizer='l2', bias_regularizer='l2',
                                     name='p_v')(p_penultimate_v)
        else:
            assert feature_codec == 'OrthogonalGaussian'
            p_v_μ_flat = keras.layers.Dense(dim_data, use_bias=use_bias,
                                            kernel_regularizer='l2', bias_regularizer='l2',
                                            name='p_v_mu_flat')(p_penultimate_v)
            p_v_logS_flat = keras.layers.Dense(dim_data, use_bias=use_bias,
                                               kernel_regularizer='l2', bias_regularizer='l2',
                                               name='p_v_logS_flat')(p_penultimate_v)
            p_v = keras.layers.Concatenate(name='p_v_mulogS_flat')([p_v_μ_flat, p_v_logS_flat])

        return ([p_adj, p_v], ('SigmoidBernoulliScaledAdjacency', feature_codec))

    return p_builder


def build_vae(q_model_codecs, p_builder, n_ξ_samples, loss_weights=None):
    """Build a full VAE given an encoder, a decoder-builder, a number of embedding samples,
    and weight losses.

    Returns
    -------
    model : nw2vec.ae.Model
        The VAE model.
    codoc_names : tuple of strings
        The list of codec names used to interpret the outputs of the VAE.

    """

    # Extract the encoder's codec.
    q, q_codecs = q_model_codecs
    assert len(q_codecs) == 1
    q_codec = q_codecs[0]
    del q_codecs

    # Generate embedding samples.
    ξ = layers.ParametrisedStochastic(q_codec, n_ξ_samples)(q.output)
    # Generate decodre output.
    p_outputs, p_codecs = p_builder(ξ)
    # Create the actual model.
    model = Model(inputs=q.input, outputs=[q.output] + p_outputs,
                  fullbatcher=q.fullbatcher, minibatcher=q.minibatcher)

    # Compile the whole thing with losses
    losses = ([codecs.get_loss(q_codec, 'kl_to_normal_loss')]
              + [codecs.get_loss(p_codec, 'estimated_pred_loss') for p_codec in p_codecs])
    if loss_weights is None:
        loss_weights = [1.0] * len(losses)
    model.compile(keras.optimizers.Adam(lr=.01),
                  loss=losses, loss_weights=loss_weights,
                  feed_dict={})

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
