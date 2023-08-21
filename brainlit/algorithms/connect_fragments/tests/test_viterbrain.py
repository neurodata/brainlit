from asyncio import create_task
import pytest
import zarr
from pathlib import Path
import numpy as np
from brainlit.algorithms.connect_fragments.viterbrain import (
    ViterBrain,
    explain_viterbrain,
)
from brainlit.preprocessing import image_process
import networkx as nx
from numpy.testing import (
    assert_array_equal,
)
import copy


@pytest.fixture(scope="session")
def create_vb(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")

    labels = np.zeros((100, 100, 2), dtype=int)
    labels[50:55, 0:25, :] = 1
    labels[50:55, 30:50, :] = 2
    labels[45:50, 55:75, :] = 3
    labels[60:65, 55:75, :] = 4
    labels[45:60, 85:, :] = 5

    tiered = np.zeros((100, 100, 2), dtype=int)

    z_tier = zarr.zeros((100, 100, 2), chunks=(50, 50, 2), dtype="float")
    z_tier[:, :, :] = tiered
    z_tier_out = data_dir / "test_state_generation" / "image.zarr"
    zarr.save(z_tier_out, z_tier)
    z_lab = zarr.zeros((100, 100, 2), chunks=(50, 50, 2), dtype="int")
    z_lab[:, :, :] = labels
    z_frag_out = data_dir / "test_state_generation" / "fragments.zarr"
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
        point1=[52, 24, 0],
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

    return vb


####################
### input checks ###
####################


def test_frag_frag_dist_bad_input(create_vb):
    vb_og = create_vb
    vb = copy.deepcopy(vb_og)

    vb.nxGraph.nodes[0]["point2"] = [0, 0, 0]
    vb.nxGraph.nodes[0]["orientation2"] = [1, 1, 0]
    vb.nxGraph.nodes[1]["point1"] = [1, 0, 0]
    vb.nxGraph.nodes[1]["orientation1"] = [1, 0, 0]
    with pytest.raises(ValueError):
        vb.frag_frag_dist(state1=0, state2=1)
    vb.nxGraph.nodes[0]["point2"] = [0, 0, 0]
    vb.nxGraph.nodes[0]["orientation2"] = [1, 0, 0]
    vb.nxGraph.nodes[1]["point1"] = [1, 0, 0]
    vb.nxGraph.nodes[1]["orientation1"] = [1, 1, 0]
    with pytest.raises(ValueError):
        vb.frag_frag_dist(state1=0, state2=1)
    vb.nxGraph.nodes[0]["point2"] = [0, 0, 0]
    vb.nxGraph.nodes[0]["orientation2"] = [np.nan, 0, 0]
    vb.nxGraph.nodes[1]["point1"] = [1, 0, 0]
    vb.nxGraph.nodes[1]["orientation1"] = [1, 0, 0]
    with pytest.raises(ValueError):
        vb.frag_frag_dist(state1=0, state2=1)
    vb.nxGraph.nodes[0]["point2"] = [0, 0, 0]
    vb.nxGraph.nodes[0]["orientation2"] = [1, 0, 0]
    vb.nxGraph.nodes[1]["point1"] = [0, 0, 0]
    vb.nxGraph.nodes[1]["orientation1"] = [1, 0, 0]
    with pytest.raises(ValueError):
        vb.frag_frag_dist(state1=0, state2=1)


def test_explain_viterbrain(create_vb):
    vb = create_vb

    vb.compute_all_costs_dist(vb.frag_frag_dist, vb.frag_soma_dist)
    vb.compute_all_costs_int(vb._line_int)
    explain_viterbrain(vb, c1=[52, 0, 0], c2=[50, 90, 0])


def test_shortest_path(create_vb):
    vb = create_vb

    vb.compute_all_costs_dist(vb.frag_frag_dist, vb.frag_soma_dist)
    vb.compute_all_costs_int(vb._line_int)
    vb.shortest_path([52, 0, 0], [50, 90, 0])


############################
### functionality checks ###
############################


def test_frag_frag_dist(create_vb):
    vb = create_vb

    cost = vb.frag_frag_dist_coord(
        pt1=[0, 0, 0], orientation1=[1, 0, 0], pt2=[1, 0, 0], orientation2=[1, 0, 0]
    )
    assert cost == 1.0

    cost = vb.frag_frag_dist_coord(
        pt1=[0, 0, 0], orientation1=[1, 0, 0], pt2=[2, 0, 0], orientation2=[1, 0, 0]
    )
    assert cost == 4.0

    cost = vb.frag_frag_dist_coord(
        pt1=[0, 0, 0], orientation1=[1, 0, 0], pt2=[1, 0, 0], orientation2=[0, 1, 0]
    )
    assert cost == 1.5

    cost = vb.frag_frag_dist_coord(
        pt1=[0, 0, 0], orientation1=[0, 1, 0], pt2=[1, 0, 0], orientation2=[0, 1, 0]
    )
    assert cost == 2.0

    cost = vb.frag_frag_dist_coord(
        pt1=[0, 0, 0], orientation1=[1, 0, 0], pt2=[2, 0, 0], orientation2=[-1, 0, 0]
    )
    assert cost == np.inf


def test_frag_frag_dist_simple(create_vb):
    vb = create_vb
    cost = vb.frag_frag_dist_simple(state1=0, state2=2)
    assert cost == 24 + 6**2

    cost = vb.frag_frag_dist_simple(state1=0, state2=6)
    assert cost == np.inf


def test_line_int_zero(create_vb):
    vb = create_vb

    cost = vb._line_int_zero(state1=0, state2=6)
    assert cost == 0


def test_viterbrain(create_vb):
    vb = create_vb

    vb.compute_all_costs_dist(vb.frag_frag_dist, vb.frag_soma_dist)
    vb.compute_all_costs_int(vb._line_int)
    assert_array_equal(nx.shortest_path(vb.nxGraph, source=0, target=8), [0, 2, 4, 8])
    assert len(vb.shortest_path([49, 10, 0], [44, 90, 0])) == 8
