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

z_im = zarr.zeros((100, 100, 1), chunks=(50, 50, 1), dtype="float")
z_im[:, :, :] = image
z_im_out = Path(__file__).parents[4] / "data" / "test_state_generation" / "image.zarr"
zarr.save(z_im_out, z_im)
z_lab = zarr.zeros((100, 100, 1), chunks=(50, 50, 1), dtype="int")
z_lab[:, :, :] = labels
z_frag_out = (
    Path(__file__).parents[4] / "data" / "test_state_generation" / "fragments.zarr"
)
zarr.save(z_frag_out, z_lab)

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


def test_state_generation():
    sg = state_generation(
        image_path=str(z_im_out),
        ilastik_program_path=None,
        ilastik_project_path=None,
        chunk_size=[50, 50, 1],
        soma_coords=soma_coords,
        resolution=res,
        prob_path=str(z_im_out),
        fragment_path=str(z_frag_out),
    )

    sg.compute_image_tiered()
    sg.compute_soma_lbls()
    assert_array_equal(sg.soma_lbls, [5])

    sg.compute_states()
    with open(sg.states_path, "rb") as handle:
        G = pickle.load(handle)
    for node in G.nodes:
        print(G.nodes[node])
    assert len(G.nodes) == 9  # 2 states per fragment plus one soma state

    sg._pc_endpoints_from_coords_neighbors(test_coords)

    sg.compute_edge_weights()
    sg.compute_bfs()
