import pytest
import numpy as np
from brainlit.algorithms.connect_fragments.viterbrain import ViterBrain
from brainlit.preprocessing import image_process
import networkx as nx
from numpy.testing import (
    assert_array_equal,
)

G = nx.DiGraph()
vb = ViterBrain(
    G=G,
    tiered_path="",
    fragment_path="",
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
