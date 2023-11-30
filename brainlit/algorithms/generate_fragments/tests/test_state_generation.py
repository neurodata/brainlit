import zarr
import numpy as np
from pathlib import Path
from brainlit.algorithms.generate_fragments.state_generation import state_generation
import networkx as nx
import pickle
import pytest
import os
import shutil


soma_coords = []

res = [0.1, 0.1, 0.1]


def make_im_labels():
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

    return image, labels


@pytest.fixture(scope="session")
def init_3d_im_lab(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    image, labels = make_im_labels()

    im_file = data_dir / "image.zarr"
    z_im = zarr.open(
        im_file, mode="w", shape=(100, 100, 1), chunks=(50, 50, 1), dtype="float"
    )
    z_im[:, :, :] = image

    lab_file = data_dir / "fragments.zarr"
    z_lab = zarr.open(
        lab_file, mode="w", shape=(100, 100, 1), chunks=(50, 50, 1), dtype="int"
    )
    z_lab[:, :, :] = labels

    return data_dir, im_file, lab_file


@pytest.fixture(scope="session")
def init_4d_im_probs_lab(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    image, labels = make_im_labels()

    im_file = data_dir / "image.zarr"
    z_im = zarr.open(
        im_file, mode="w", shape=(2, 100, 100, 1), chunks=(50, 50, 1), dtype="float"
    )
    z_im[0, :, :, :] = image
    z_im[1, :, :, :] = np.zeros(z_im.shape[1:], dtype="float")

    probs_file = data_dir / "probs.zarr"
    z_prob = zarr.open(
        probs_file, mode="w", shape=(100, 100, 1), chunks=(50, 50, 1), dtype="float"
    )
    z_prob[:, :, :] = image

    lab_file = data_dir / "fragments.zarr"
    z_lab = zarr.open(
        lab_file, mode="w", shape=(100, 100, 1), chunks=(50, 50, 1), dtype="int"
    )
    z_lab[:, :, :] = labels

    return data_dir, im_file, probs_file, lab_file


####################
### input checks ###
####################


def test_state_generation_inputs(init_3d_im_lab):
    data_dir, im_file, lab_file = init_3d_im_lab

    im_file = data_dir / "im2d.zarr"
    zarr.open(im_file, mode="w", shape=(10, 10), chunks=(5, 5), dtype="float")

    with pytest.raises(ValueError):
        sg = state_generation(
            image_path=im_file,
            new_layers_dir=data_dir,
            ilastik_program_path=None,
            ilastik_project_path=None,
        )


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


def test_compute_frags(init_3d_im_lab):
    data_dir, im_file, lab_file = init_3d_im_lab

    sg = state_generation(
        image_path=im_file,
        new_layers_dir=data_dir,
        ilastik_program_path=None,
        ilastik_project_path=None,
        soma_coords=soma_coords,
        resolution=res,
        prob_path=im_file,
    )
    sg.compute_frags()
    z_lab = zarr.open(sg.fragment_path)
    assert len(np.unique(z_lab)) > 5


def test_state_generation_3d(init_3d_im_lab):
    data_dir, im_file, lab_file = init_3d_im_lab

    sg = state_generation(
        image_path=im_file,
        new_layers_dir=data_dir,
        ilastik_program_path=None,
        ilastik_project_path=None,
        soma_coords=soma_coords,
        resolution=res,
        prob_path=im_file,
        fragment_path=lab_file,
    )

    sg.compute_image_tiered()
    tiered_path = data_dir / "tiered.zarr"
    assert os.path.exists(tiered_path)
    shutil.rmtree(tiered_path)

    sg = state_generation(
        image_path=str(im_file),
        new_layers_dir=str(data_dir),
        ilastik_program_path=None,
        ilastik_project_path=None,
        soma_coords=soma_coords,
        resolution=res,
        prob_path=str(im_file),
        fragment_path=str(lab_file),
    )

    sg.compute_image_tiered()
    sg.compute_soma_lbls()

    sg.compute_states("nb")
    with pytest.raises(NotImplementedError):
        sg.compute_states("pc")

    with open(sg.states_path, "rb") as handle:
        G = pickle.load(handle)
    for node in G.nodes:
        print(G.nodes[node])
    assert len(G.nodes) == 10  # 2 states per fragment

    sg.compute_edge_weights()
    sg.compute_bfs()


def test_state_generation_4d(init_4d_im_probs_lab):
    data_dir, im_file, probs_file, lab_file = init_4d_im_probs_lab

    sg = state_generation(
        image_path=str(im_file),
        new_layers_dir=str(data_dir),
        ilastik_program_path=None,
        ilastik_project_path=None,
        soma_coords=soma_coords,
        fg_channel=0,
        resolution=res,
        prob_path=str(probs_file),
        fragment_path=str(lab_file),
    )

    sg.compute_image_tiered()
    sg.compute_soma_lbls()

    sg.compute_states("nb")
    with pytest.raises(NotImplementedError):
        sg.compute_states("pc")

    with open(sg.states_path, "rb") as handle:
        G = pickle.load(handle)
    for node in G.nodes:
        print(G.nodes[node])
    assert len(G.nodes) == 10  # 2 states per fragment

    sg.compute_edge_weights()
    sg.compute_bfs()
