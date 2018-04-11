import itertools

import numpy as np
import networkx as nx
import seaborn as sb
from matplotlib.patches import Wedge, Circle
from matplotlib.text import Text
import matplotlib.pyplot as plt

from nw2vec import utils


BORDER_PROP = .2
RADIUS_GAP = .2
THETA_GAP = .05
_network_layouts = {}


def nodes_patches(layout, labels, training_nodes=[], radius=.04):
    titles, xys = zip(*sorted(layout.items()))

    # Preprocess labels
    labels = np.array(labels) + 1e-6
    assert len(labels.shape) == 2
    labels /= labels.sum(1)[:, np.newaxis]
    palette_border = np.array(sb.color_palette(n_colors=labels.shape[1]))
    palette_circle = np.array(sb.color_palette(n_colors=labels.shape[1]))

    # Create wedges
    wedges = []
    # Scale thetas to leave space for the gaps
    thetas2 = labels.cumsum(1) * (1 - THETA_GAP * labels.shape[1])
    thetas1 = np.concatenate([np.zeros((labels.shape[0], 1)), thetas2[:, :-1]], axis=1)
    # Shift thetas to get the gaps
    thetas2 = thetas2 + np.arange(labels.shape[1]) * THETA_GAP
    thetas1 = thetas1 + np.arange(labels.shape[1]) * THETA_GAP
    for i, (xy, theta1, theta2) in enumerate(zip(utils.inner_repeat(xys, labels.shape[1]),
                                                 thetas1.flat, thetas2.flat)):
        wedges.append(Wedge(xy, radius,
                            theta1 * 360, theta2 * 360,
                            width=BORDER_PROP * radius,
                            color=palette_border[i % labels.shape[1]]))

    # Create central circles and titles
    circles = []
    texts = []
    color_predictions = palette_circle[np.argmax(labels, axis=1)]
    color_predictions[training_nodes] = [0, 0, 0]
    for xy, color, title in zip(xys, color_predictions, titles):
        circles.append(Circle(xy, radius * (1 - BORDER_PROP - RADIUS_GAP), color=color))
        texts.append(Text(x=xy[0], y=xy[1], text=str(title),
                          va='center', ha='center',
                          color='white'))

    return wedges + circles, texts


def update_nodes_patches(patches, labels, training_nodes=[]):
    # Preprocess labels
    labels = np.array(labels) + 1e-6
    assert len(labels.shape) == 2
    labels /= labels.sum(1)[:, np.newaxis]
    palette = np.array(sb.color_palette(n_colors=labels.shape[1]))

    # Extract wedges and circles
    wedges = list(filter(lambda p: isinstance(p, Wedge), patches))
    assert len(wedges) == np.prod(labels.shape)
    circles = list(filter(lambda p: isinstance(p, Circle), patches))
    assert len(circles) == labels.shape[0]

    # Update wedges
    # Scale thetas to leave space for the gaps
    thetas2 = labels.cumsum(1) * (1 - THETA_GAP * labels.shape[1])
    thetas1 = np.concatenate([np.zeros((labels.shape[0], 1)), thetas2[:, :-1]], axis=1)
    # Shift thetas to get the gaps
    thetas2 = thetas2 + np.arange(labels.shape[1]) * THETA_GAP
    thetas1 = thetas1 + np.arange(labels.shape[1]) * THETA_GAP
    for theta1, theta2, wedge in zip(thetas1.flat, thetas2.flat, wedges):
        wedge.set_theta1(theta1 * 360)
        wedge.set_theta2(theta2 * 360)

    # Update circles
    color_predictions = palette[np.argmax(labels, axis=1)]
    color_predictions[training_nodes] = [0, 0, 0]
    for circle, color in zip(circles, color_predictions):
        circle.set_color(color)


def draw_network(g, labels=None, training_nodes=[], ax=None, relayout=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.figure

    if relayout or g not in _network_layouts:
        _network_layouts[g] = nx.drawing.layout.spring_layout(g)
    layout = _network_layouts[g]

    if labels is None:
        assert len(training_nodes) == 0
        nx.draw_networkx(g, pos=layout, ax=ax, node_color='#65cb5e')
        node_patches = None
        edge_collection = None
        text_items = None
    else:
        # Plot nodes, edges and labels
        node_patches, text_items = nodes_patches(layout, labels, training_nodes=training_nodes)
        for artist in itertools.chain(node_patches, text_items):
            ax.add_artist(artist)
        edge_collection = nx.draw_networkx_edges(g, pos=layout, edge_color='grey', ax=ax)

    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect='equal')
    return (fig, ax), (node_patches, edge_collection, text_items)
