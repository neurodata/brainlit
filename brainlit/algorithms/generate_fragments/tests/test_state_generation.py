import zarr
import numpy as np
from pathlib import Path
from brainlit.algorithms.generate_fragments.state_generation import state_generation
import networkx as nx
import pickle
from numpy.testing import (
    assert_array_equal,
)

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

soma_coords = [[50, 90, 0]]

res = [0.1, 0.1, 0.1]

############################
### functionality checks ###
############################

test_coords = np.hstack(
    (
        np.arange(100).reshape(100, 1),
        np.arange(100).reshape(100, 1),
        np.arange(100).reshape(100, 1),
    )
)


def test_state_generation_3d(tmp_path):
    im_file = str(tmp_path / "image.zarr")
    z_im = zarr.open(
        im_file, mode="w", shape=(100, 100, 1), chunks=(50, 50, 1), dtype="float"
    )
    z_im[:, :, :] = image
    lab_file = str(tmp_path / "fragments.zarr")
    z_lab = zarr.open(
        lab_file, mode="w", shape=(100, 100, 1), chunks=(50, 50, 1), dtype="int"
    )
    z_lab[:, :, :] = labels

    sg = state_generation(
        image_path=im_file,
        new_layers_dir=str(tmp_path),
        ilastik_program_path=None,
        ilastik_project_path=None,
        chunk_size=[50, 50, 1],
        soma_coords=soma_coords,
        resolution=res,
        prob_path=im_file,
        fragment_path=lab_file,
    )

    sg.compute_image_tiered()
    sg.compute_soma_lbls()
    assert_array_equal(sg.soma_lbls, [5])

    sg.compute_states("nb")
    sg.compute_states("pc")

    with open(sg.states_path, "rb") as handle:
        G = pickle.load(handle)
    for node in G.nodes:
        print(G.nodes[node])
    assert len(G.nodes) == 9  # 2 states per fragment plus one soma state

    sg.compute_edge_weights()
    sg.compute_bfs()


def test_state_generation_4d(tmp_path):
    im_file = str(tmp_path / "image.zarr")
    z_im = zarr.open(
        im_file, mode="w", shape=(2, 100, 100, 1), chunks=(50, 50, 1), dtype="float"
    )
    z_im[0, :, :, :] = image
    z_im[1, :, :, :] = np.zeros(z_im.shape[1:], dtype="float")
    probs_file = str(tmp_path / "probs.zarr")
    z_prob = zarr.open(
        probs_file, mode="w", shape=(100, 100, 1), chunks=(50, 50, 1), dtype="float"
    )
    z_prob[:, :, :] = image
    lab_file = str(tmp_path / "fragments.zarr")
    z_lab = zarr.open(
        lab_file, mode="w", shape=(100, 100, 1), chunks=(50, 50, 1), dtype="int"
    )
    z_lab[:, :, :] = labels

    sg = state_generation(
        image_path=im_file,
        new_layers_dir=str(tmp_path),
        ilastik_program_path=None,
        ilastik_project_path=None,
        chunk_size=[50, 50, 1],
        soma_coords=soma_coords,
        fg_channel=0,
        resolution=res,
        prob_path=probs_file,
        fragment_path=lab_file,
    )

    sg.compute_image_tiered()
    sg.compute_soma_lbls()
    assert_array_equal(sg.soma_lbls, [5])

    sg.compute_states("nb")
    sg.compute_states("pc")

    with open(sg.states_path, "rb") as handle:
        G = pickle.load(handle)
    for node in G.nodes:
        print(G.nodes[node])
    assert len(G.nodes) == 9  # 2 states per fragment plus one soma state

    sg.compute_edge_weights()
    sg.compute_bfs()
