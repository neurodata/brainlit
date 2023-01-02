import pytest
import numpy as np
from brainlit.archive import dynamic_programming_viterbi2
from brainlit.preprocessing import image_process
import networkx as nx
from numpy.testing import (
    assert_array_equal,
)

####################
### input checks ###
####################


def test_viterbi2_bad_input():
    image = np.ones((10, 10, 10))
    labels = np.zeros((10, 10, 10), dtype=int)

    labels[0, 0, 0] = 1
    labels[9, 9, 9] = 3

    with pytest.raises(ValueError):
        mpnp = dynamic_programming_viterbi2.most_probable_neuron_path(image, labels)


def test_viterbi2_valid_input():
    image = np.zeros((10, 10, 10))
    labels = np.zeros((10, 10, 10), dtype=int)

    labels[0, 0, 0] = 1
    labels[9, 9, 9] = 2
    image[0, 0, 0] = 1
    image[9, 9, 9] = 1

    mpnp = dynamic_programming_viterbi2.most_probable_neuron_path(image, labels)


############################
### functionality checks ###
############################
image = 0.5 * np.ones((100, 100, 1))
image[50:55, 0:25, 0] = 0.91
image[50:55, 30:50, 0] = 0.92
image[45:50, 55:75, 0] = 0.93
image[60:65, 55:75, 0] = 0.94
image[45:60, 85:, 0] = 0.95

labels = np.zeros((100, 100, 1), dtype=int)
labels[50:55, 0:25, 0] = 1
labels[50:55, 30:50, 0] = 2
labels[45:50, 55:75, 0] = 3
labels[60:65, 55:75, 0] = 4
labels[45:60, 85:, 0] = 5

axon_coords = [[52, 2, 0]]
soma_coords = [[50, 90, 0]]

res = [0.1, 0.1, 0.1]


def test_viterbi2():
    _, axon_lbls = image_process.label_points(labels, axon_coords, res)
    axon_lbl = axon_lbls[0]
    _, soma_lbls = image_process.label_points(labels, soma_coords, res)
    soma_lbl = soma_lbls[0]

    mpnp = dynamic_programming_viterbi2.most_probable_neuron_path(
        image, labels, [soma_lbl], res
    )
    mpnp.compute_states()
    mpnp.compute_all_costs_dist(
        point_point_func=mpnp.point_point_dist, point_blob_func=mpnp.point_blob_dist
    )
    mpnp.compute_all_costs_int()
    mpnp.create_nx_graph()

    path_states = nx.shortest_path(mpnp.nxGraph, 0, 8, weight="weight")
    path_comps = [mpnp.state_to_comp[state][1] for state in path_states]

    assert_array_equal(path_comps, [1, 2, 3, 5])


def test_viterbi2_parallel():
    _, axon_lbls = image_process.label_points(labels, axon_coords, res)
    axon_lbl = axon_lbls[0]
    _, soma_lbls = image_process.label_points(labels, soma_coords, res)
    soma_lbl = soma_lbls[0]

    mpnp = dynamic_programming_viterbi2.most_probable_neuron_path(
        image, labels, [soma_lbl], res, parallel=2
    )
    mpnp.compute_states()
    mpnp.compute_all_costs_dist(
        point_point_func=mpnp.point_point_dist, point_blob_func=mpnp.point_blob_dist
    )
    mpnp.compute_all_costs_int()
    mpnp.create_nx_graph()

    path_states = nx.shortest_path(mpnp.nxGraph, 0, 8, weight="weight")
    path_comps = [mpnp.state_to_comp[state][1] for state in path_states]

    assert_array_equal(path_comps, [1, 2, 3, 5])
