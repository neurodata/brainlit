import pytest
from brainlit.algorithms.connect_fragments.fit_spline import GeometricGraph
import networkx as nx


def test_loc():
    """Nodes should be defined in loc attribute
    """

    neuron=GeometricGraph()
    neuron.add_node(1)
    neuron.add_node(2)
    neuron.add_node(3)
    neuron.add_edge(1,2)
    neuron.add_edge(2,3)
    neuron.add_edge(3,1)
    spline_tree = neuron.fit_spline_tree_invariant()

#def test_find_longet_path():
    """only allow geometric graph to have one root
    """

    #G=GeometricGraph()
    #G.add_node(1)
    #G.add_node(2)
    #G.add_node(3)
    #G.add_node(4)
    #G.add_node(5)
    #G.add_edge(1,2)
    #G.add_edge(1,3)
    #G.add_edge(3,5)
    #G.add_edge(4,3)
    #H=G.fit_spline_tree_invariant()
    #if nx.is_directed(G)==False:
    #    raise ValueError("graph is not directed")
