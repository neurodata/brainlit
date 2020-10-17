import pytest
from brainlit.algorithms.connect_fragments.fit_spline import GeometricGraph
import networkx as nx
from pathlib import Path
from brainlit.utils.swc import read_swc_offset
import numpy as np

def test_fit_spline_tree_invariant()
G=GeometricGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,1)
