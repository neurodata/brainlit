import pytest
from brainlit.algorithms.connect_fragments import fit_spline
import networkx as nx


def test_closed_shape()
'''
tree should not include closed shapes
'''
G=GeometricGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,1)

def test_multiple_roots()
'''
only allow geometric graph with one root
'''
G=GeometricGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_edge(1,2)
G.add_edge(1,3)
G.add_edge(3,5)
G.add_edge(4,3)
