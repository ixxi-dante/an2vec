from ._nw2vec_rust import array_split
from . import layers
from . import codecs


__all__ = ['array_split', 'custom_objects']


def custom_objects():
    available_codecs = codecs.available_codecs()
    available_losses = codecs.available_fullname_losses()
    available_layers = layers.available_layers()
    return dict(list(available_codecs.items())
                + list(available_losses.items())
                + list(available_layers.items()))
