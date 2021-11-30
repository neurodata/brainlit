import pytest
import numpy as np
from brainlit.algorithms.connect_fragments.viterbrain import ViterBrain
from brainlit.preprocessing import image_process
import networkx as nx
from numpy.testing import (
    assert_array_equal,
)

####################
### input checks ###
####################

def test_frag_frag_dist_bad_input():
    with pytest.raises(ValueError):
        ViterBrain.frag_frag_dist(pt1=[0,0,0], orientation1=[1,1,0], pt2=[1,0,0], orientation2=[1,0,0])
    with pytest.raises(ValueError):
        ViterBrain.frag_frag_dist(pt1=[0,0,0], orientation1=[1,0,0], pt2=[1,0,0], orientation2=[1,1,0])
    with pytest.raises(ValueError):
        ViterBrain.frag_frag_dist(pt1=[0,0,0], orientation1=[np.nan,0,0], pt2=[1,0,0], orientation2=[1,1,0])
    with pytest.raises(ValueError):
        ViterBrain.frag_frag_dist(pt1=[0,0,0], orientation1=[np.nan,0,0], pt2=[1,0,0], orientation2=[np.nan,1,0])

def test_frag_soma_dist_bad_input():
    with pytest.raises(ValueError):
        ViterBrain.frag_frag_dist(point=[0,0,0], orientation=[1,1,0], pt2=[1,0,0], orientation2=[1,0,0])
    with pytest.raises(ValueError):
        ViterBrain.frag_frag_dist(pt1=[0,0,0], orientation1=[1,0,0], pt2=[1,0,0], orientation2=[1,1,0])
    with pytest.raises(ValueError):
        ViterBrain.frag_frag_dist(pt1=[0,0,0], orientation1=[np.nan,0,0], pt2=[1,0,0], orientation2=[1,1,0])
    with pytest.raises(ValueError):
        ViterBrain.frag_frag_dist(pt1=[0,0,0], orientation1=[np.nan,0,0], pt2=[1,0,0], orientation2=[np.nan,1,0])
