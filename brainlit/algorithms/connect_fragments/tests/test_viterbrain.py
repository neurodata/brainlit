import pytest
import zarr
from pathlib import Path
import numpy as np
from brainlit.algorithms.connect_fragments.viterbrain import ViterBrain
from brainlit.preprocessing import image_process
import networkx as nx
from numpy.testing import (
    assert_array_equal,
)

labels = np.zeros((100, 100, 1), dtype=int)
labels[50:55, 0:25, 0] = 1
labels[50:55, 30:50, 0] = 2
labels[45:50, 55:75, 0] = 3
labels[60:65, 55:75, 0] = 4
labels[45:60, 85:, 0] = 5

tiered = np.zeros((100, 100, 1), dtype=int)

z_tier = zarr.zeros((100, 100, 1), chunks=(50, 50, 1), dtype="float")
z_tier[:, :, :] = tiered
z_tier_out = Path(__file__).parents[4] / "data" / "test_state_generation" / "image.zarr"
zarr.save(z_tier_out, z_tier)
z_lab = zarr.zeros((100, 100, 1), chunks=(50, 50, 1), dtype="int")
z_lab[:, :, :] = labels
z_frag_out = (
    Path(__file__).parents[4] / "data" / "test_state_generation" / "fragments.zarr"
)
zarr.save(z_frag_out, z_lab)

G = nx.DiGraph()

G.add_node(
    0,
    type="fragment",
    fragment=1,
    point1=[52, 0, 0],
    point2=[52, 24, 0],
    orientation1=[0, 1, 0],
    orientation2=[0, 1, 0],
    image_cost=0,
    twin=1,
)
G.add_node(
    1,
    type="fragment",
    fragment=1,
    point1=[52, 25, 0],
    point2=[52, 0, 0],
    orientation1=[0, -1, 0],
    orientation2=[0, -1, 0],
    image_cost=0,
    twin=0,
)
G.add_node(
    2,
    type="fragment",
    fragment=2,
    point1=[52, 30, 0],
    point2=[52, 49, 0],
    orientation1=[0, 1, 0],
    orientation2=[0, 1, 0],
    image_cost=0,
    twin=3,
)
G.add_node(
    3,
    type="fragment",
    fragment=2,
    point1=[52, 49, 0],
    point2=[52, 30, 0],
    orientation1=[0, -1, 0],
    orientation2=[0, -1, 0],
    image_cost=0,
    twin=2,
)
G.add_node(
    4,
    type="fragment",
    fragment=3,
    point1=[47, 55, 0],
    point2=[47, 74, 0],
    orientation1=[0, 1, 0],
    orientation2=[0, 1, 0],
    image_cost=0,
    twin=5,
)
G.add_node(
    5,
    type="fragment",
    fragment=3,
    point1=[47, 74, 0],
    point2=[47, 55, 0],
    orientation1=[0, -1, 0],
    orientation2=[0, -1, 0],
    image_cost=0,
    twin=4,
)
G.add_node(
    6,
    type="fragment",
    fragment=4,
    point1=[62, 55, 0],
    point2=[62, 74, 0],
    orientation1=[0, 1, 0],
    orientation2=[0, 1, 0],
    image_cost=0,
    twin=7,
)
G.add_node(
    7,
    type="fragment",
    fragment=4,
    point1=[62, 74, 0],
    point2=[62, 55, 0],
    orientation1=[0, -1, 0],
    orientation2=[0, -1, 0],
    image_cost=0,
    twin=6,
)
G.add_node(8, type="soma", fragment=5, soma_coords=np.argwhere(labels == 5))

vb = ViterBrain(
    G=G,
    tiered_path=str(z_tier_out),
    fragment_path=str(z_frag_out),
    resolution=[1, 1, 1],
    coef_curv=1,
    coef_dist=1,
    coef_int=1,
)

####################
### input checks ###
####################


def test_frag_frag_dist_bad_input():
    with pytest.raises(ValueError):
        vb.frag_frag_dist(
            pt1=[0, 0, 0], orientation1=[1, 1, 0], pt2=[1, 0, 0], orientation2=[1, 0, 0]
        )
    with pytest.raises(ValueError):
        vb.frag_frag_dist(
            pt1=[0, 0, 0], orientation1=[1, 0, 0], pt2=[1, 0, 0], orientation2=[1, 1, 0]
        )
    with pytest.raises(ValueError):
        vb.frag_frag_dist(
            pt1=[0, 0, 0],
            orientation1=[np.nan, 0, 0],
            pt2=[1, 0, 0],
            orientation2=[1, 1, 0],
        )
    with pytest.raises(ValueError):
        vb.frag_frag_dist(
            pt1=[0, 0, 0],
            orientation1=[np.nan, 0, 0],
            pt2=[1, 0, 0],
            orientation2=[np.nan, 1, 0],
        )


############################
### functionality checks ###
############################


def test_frag_frag_dist():
    cost = vb.frag_frag_dist(
        pt1=[0, 0, 0], orientation1=[1, 0, 0], pt2=[1, 0, 0], orientation2=[1, 0, 0]
    )
    assert cost == 1.0

    cost = vb.frag_frag_dist(
        pt1=[0, 0, 0], orientation1=[1, 0, 0], pt2=[2, 0, 0], orientation2=[1, 0, 0]
    )
    assert cost == 4.0

    cost = vb.frag_frag_dist(
        pt1=[0, 0, 0], orientation1=[1, 0, 0], pt2=[1, 0, 0], orientation2=[0, 1, 0]
    )
    assert cost == 1.5

    cost = vb.frag_frag_dist(
        pt1=[0, 0, 0], orientation1=[0, 1, 0], pt2=[1, 0, 0], orientation2=[0, 1, 0]
    )
    assert cost == 2.0

    cost = vb.frag_frag_dist(
        pt1=[0, 0, 0], orientation1=[1, 0, 0], pt2=[2, 0, 0], orientation2=[-1, 0, 0]
    )
    assert cost == np.inf


def test_viterbrain():
    vb.compute_all_costs_dist(vb.frag_frag_dist, vb.frag_soma_dist)
    vb.compute_all_costs_int()
    assert_array_equal(nx.shortest_path(vb.nxGraph, source=0, target=8), [0, 2, 4, 8])
