from asyncio import create_task
import pytest
import zarr
from pathlib import Path
import numpy as np
from brainlit.algorithms.connect_fragments.viterbrain import (
    ViterBrain,
    explain_viterbrain,
    _curv_dist,
    _dist_simple,
    _compute_dist_cost,
    _line_int_zero,
    _line_int_coord,
    _compute_int_cost,
)
from brainlit.preprocessing import image_process
import networkx as nx
from numpy.testing import (
    assert_array_equal,
)
import copy
from networkx import NetworkXNoPath


@pytest.fixture(scope="session")
def create_im_tiered(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")

    im_path = data_dir / "image_tiered.zarr"

    z_tiered = zarr.open(
        im_path, mode="w", shape=(10, 10, 10), chunks=(5, 5, 5), dtype="uint16"
    )
    ra = np.zeros((10, 10, 10), dtype="uint16")
    ra[5, 5, :] = np.arange(10)

    z_tiered[:, :, :] = ra[:, :, :]

    return im_path


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
    G.add_node(
        8,
        type="fragment",
        fragment=5,
        point1=[52, 85, 0],
        point2=[52, 99, 0],
        orientation1=[0, 1, 0],
        orientation2=[0, 1, 0],
        image_cost=0,
        twin=9,
    )
    G.add_node(
        9,
        type="fragment",
        fragment=5,
        point1=[52, 99, 0],
        point2=[52, 85, 0],
        orientation1=[0, -1, 0],
        orientation2=[0, -1, 0],
        image_cost=0,
        twin=8,
    )

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


@pytest.fixture(scope="session")
def create_vb_soma(tmp_path_factory):
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
    G.add_node(
        8,
        type="soma",
        fragment=5,
        soma_coords=np.argwhere(labels == 5),
    )

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


def test_compute_dist_cost(create_vb):
    vb_og = create_vb
    vb = copy.deepcopy(vb_og)

    # distance between two states of same fragment
    vb.nxGraph.nodes[0]["point2"] = [0, 0, 0]
    vb.nxGraph.nodes[0]["orientation2"] = [1, 1, 0]
    vb.nxGraph.nodes[1]["point1"] = [1, 0, 0]
    vb.nxGraph.nodes[1]["orientation1"] = [1, 0, 0]
    result = _compute_dist_cost(
        ((0, vb.nxGraph.nodes[0]), (1, vb.nxGraph.nodes[1])), res=[1, 1, 1]
    )
    assert result[2] == np.inf


def test_curv_dist_bad_input(create_vb):
    # orientation1 of state2 is not unit length
    with pytest.raises(ValueError):
        _curv_dist(
            res=[1, 1, 1],
            pt1=[0, 0, 0],
            orientation1=[1, 0, 0],
            pt2=[1, 0, 0],
            orientation2=[1, 1, 0],
        )

    # nan orientation will give nan curvature
    with pytest.raises(ValueError):
        _curv_dist(
            res=[1, 1, 1],
            pt1=[0, 0, 0],
            orientation1=[np.nan, 0, 0],
            pt2=[1, 0, 0],
            orientation2=[1, 0, 0],
        )

    # identical points will give nan curvature
    with pytest.raises(ValueError):
        _curv_dist(
            res=[1, 1, 1],
            pt1=[0, 0, 0],
            orientation1=[1, 0, 0],
            pt2=[0, 0, 0],
            orientation2=[1, 0, 0],
        )


def test_explain_viterbrain(create_vb):
    vb = create_vb

    vb.compute_all_costs_dist()
    vb.compute_all_costs_int()
    explain_viterbrain(vb, c1=[52, 0, 0], c2=[50, 90, 0])


def test_shortest_path(create_vb):
    vb = create_vb

    vb.compute_all_costs_dist()
    vb.compute_all_costs_int()
    vb.shortest_path([52, 0, 0], [50, 90, 0])


############################
### functionality checks ###
############################


def test_frag_frag_dist(create_vb):
    vb = create_vb

    dist, k_cost = _curv_dist(
        res=[1, 1, 1],
        pt1=[0, 0, 0],
        orientation1=[1, 0, 0],
        pt2=[1, 0, 0],
        orientation2=[1, 0, 0],
    )
    assert dist == 1.0
    assert k_cost == 0

    dist, k_cost = _curv_dist(
        res=[1, 1, 1],
        pt1=[0, 0, 0],
        orientation1=[1, 0, 0],
        pt2=[2, 0, 0],
        orientation2=[1, 0, 0],
    )
    assert dist == 2.0
    assert k_cost == 0

    dist, k_cost = _curv_dist(
        res=[1, 1, 1],
        pt1=[0, 0, 0],
        orientation1=[1, 0, 0],
        pt2=[1, 0, 0],
        orientation2=[0, 1, 0],
    )
    assert dist == 1.0
    assert k_cost == 0.5

    dist, k_cost = _curv_dist(
        res=[1, 1, 1],
        pt1=[0, 0, 0],
        orientation1=[0, 1, 0],
        pt2=[1, 0, 0],
        orientation2=[0, 1, 0],
    )
    assert dist == 1.0
    assert k_cost == 1.0

    dist, k_cost = _curv_dist(
        res=[2, 1, 3],
        pt1=[0, 0, 0],
        orientation1=[0, 1, 0],
        pt2=[1, 0, 0],
        orientation2=[0, 1, 0],
    )
    assert dist == 2.0
    assert k_cost == 1.0

    dist, k_cost = _curv_dist(
        res=[1, 1, 1],
        pt1=[0, 0, 0],
        orientation1=[1, 0, 0],
        pt2=[16, 0, 0],
        orientation2=[1, 0, 0],
    )
    assert dist == np.inf
    assert k_cost == np.inf

    dist, k_cost = _curv_dist(
        res=[1, 1, 1],
        pt1=[0, 0, 0],
        orientation1=[1, 0, 0],
        pt2=[2, 0, 0],
        orientation2=[-1, 0, 0],
    )
    assert dist == np.inf
    assert k_cost == np.inf


def test_frag_frag_dist_simple(create_vb):
    vb = create_vb
    G = vb.nxGraph
    cost = _dist_simple(
        res=[1, 1, 1],
        pt0=G.nodes[0]["point1"],
        pt1=G.nodes[0]["point2"],
        orientation1=G.nodes[0]["orientation2"],
        pt2=G.nodes[2]["point1"],
        orientation2=G.nodes[2]["orientation1"],
    )
    assert cost == 24 + 6**2

    cost = _dist_simple(
        res=[1, 1, 1],
        pt0=G.nodes[0]["point1"],
        pt1=G.nodes[0]["point2"],
        orientation1=G.nodes[0]["orientation2"],
        pt2=G.nodes[6]["point1"],
        orientation2=G.nodes[6]["orientation1"],
    )
    assert cost == np.inf


def test_frag_soma_dist(create_vb_soma):
    vb = create_vb_soma

    _, nonline_point = vb.frag_soma_dist(
        point=[62, 74, 0], orientation=[1, 0, 0], soma_lbl=5
    )
    assert_array_equal(nonline_point, [59, 85, 0])


def test_line_int_zero(create_vb):
    vb = create_vb
    G = vb.nxGraph
    cost = _line_int_zero(G.nodes[0]["point2"], G.nodes[6]["point1"], tiered_path="")
    assert cost == 0


def test_line_int_coord(create_im_tiered):
    im_path = create_im_tiered
    sum = _line_int_coord([5, 5, 0], [5, 5, 9], im_path)
    assert sum == 36


def test_compute_int_cost(create_vb):
    vb = create_vb
    G = vb.nxGraph
    s1, s2 = 0, 1

    state1_data = (s1, G.nodes[s1])
    state2_data = (s2, G.nodes[s2])

    s1, s2, cost = _compute_int_cost((state1_data, state2_data), "")

    assert cost == np.inf


def test_viterbrain(create_vb):
    vb = create_vb

    vb.compute_all_costs_dist()
    vb.compute_all_costs_int()
    assert_array_equal(nx.shortest_path(vb.nxGraph, source=0, target=8), [0, 2, 4, 8])
    assert (
        len(vb.shortest_path([49, 10, 0], [44, 90, 0])) == 9
    )  # start point + state 0 endpoints + state 2 endpoints + state 4 endpoints + state 8 start point + end point
