import pytest
from brainlit.algorithms.connect_fragments.fit_spline import GeometricGraph
import networkx as nx
import numpy as np

def test_loc():
    """Nodes should be defined in 'loc' attribute
       pytest test_fsti.py -k test_loc
    """

    neuron=GeometricGraph()
    #add nodes
    neuron.add_node(1,location=np.array([100,100,200]))
    neuron.add_node(2,location=np.array([200,0,200]))
    neuron.add_node(3,location=np.array([200,300,200]))
    neuron.add_node(4,location=np.array([300,400,200]))
    neuron.add_node(5,location=np.array([100,500,200]))
    #define soma
    soma=[100,100,200]
    #add edges
    neuron.add_edge(2,1)
    neuron.add_edge(3,2)
    neuron.add_edge(4,2)
    neuron.add_edge(5,2)
    #build spline tree using fit_spline_tree_invariant()
    spline_tree = neuron.fit_spline_tree_invariant()

def test_loc_np():
    """Nodes 'loc' should be a numpy.array
       pytest test_fsti.py -k test_loc_np
    """

    neuron=GeometricGraph()
    #add nodes
    neuron.add_node(1,loc=[100,100,200])
    neuron.add_node(2,loc=[200,0,200])
    neuron.add_node(3,loc=[200,300,200])
    neuron.add_node(4,loc=[300,400,200])
    neuron.add_node(5,loc=[100,500,200])
    #define soma
    soma=[100,100,200]
    #add edges
    neuron.add_edge(2,1)
    neuron.add_edge(3,2)
    neuron.add_edge(4,2)
    neuron.add_edge(5,2)
    #build spline tree using fit_spline_tree_invariant()
    spline_tree = neuron.fit_spline_tree_invariant()

def test_loc_3d():
    """Nodes 'loc' should be 3-dimensional
       pytest test_fsti.py -k test_loc_3d
    """

    neuron=GeometricGraph()
    #add nodes
    neuron.add_node(1,loc=np.array([100,100]))
    neuron.add_node(2,loc=np.array([200,0]))
    neuron.add_node(3,loc=np.array([200,300]))
    neuron.add_node(4,loc=np.array([300,400]))
    neuron.add_node(5,loc=np.array([100,500]))

    #define soma
    soma=[100,100]
    #add edges
    neuron.add_edge(2,1)
    neuron.add_edge(3,2)
    neuron.add_edge(4,2)
    neuron.add_edge(5,2)
    #build spline tree using fit_spline_tree_invariant()
    spline_tree = neuron.fit_spline_tree_invariant()

def test_edgcov():
    """check if every node is assigned to at least one edge
       pytest test_fsti.py -k test_edgcov
    """

    neuron=GeometricGraph()
    #add nodes
    neuron.add_node(1,loc=np.array([100,100,200]))
    neuron.add_node(2,loc=np.array([200,0,200]))
    neuron.add_node(3,loc=np.array([200,300,200]))
    neuron.add_node(4,loc=np.array([300,400,200]))
    neuron.add_node(5,loc=np.array([100,500,200]))
    #define soma
    soma=[100,100,200]
    #add edges
    neuron.add_edge(2,1)
    neuron.add_edge(3,2)
    neuron.add_edge(4,3)
    #build spline tree using fit_spline_tree_invariant()
    spline_tree = neuron.fit_spline_tree_invariant()

def test_cycle():
    """check if the geometric graph contains undirected cycle(s)
       pytest test_fsti.py -k test_cycle
    """

    neuron=GeometricGraph()
    #add nodes
    neuron.add_node(1,loc=np.array([100,100,200]))
    neuron.add_node(2,loc=np.array([200,0,200]))
    neuron.add_node(3,loc=np.array([200,300,200]))
    neuron.add_node(4,loc=np.array([300,400,200]))
    neuron.add_node(5,loc=np.array([100,500,200]))
    #define soma
    soma=[100,100,200]
    #add edges
    neuron.add_edge(2,1)
    neuron.add_edge(3,2)
    neuron.add_edge(4,3)
    neuron.add_edge(5,4)
    neuron.add_edge(3,5)
    #build spline tree using fit_spline_tree_invariant()
    spline_tree = neuron.fit_spline_tree_invariant()

def test_disconnect():
    """check if the geometric graph contains disconnected segment(s)
       pytest test_fsti.py -k test_disconnect
    """

    neuron=GeometricGraph()
    #add nodes
    neuron.add_node(1,loc=np.array([100,100,200]))
    neuron.add_node(2,loc=np.array([200,0,200]))
    neuron.add_node(3,loc=np.array([200,300,200]))
    neuron.add_node(4,loc=np.array([300,400,200]))
    neuron.add_node(5,loc=np.array([100,500,200]))
    #define soma
    soma=[100,100,200]
    #add edges
    neuron.add_edge(2,1)
    neuron.add_edge(3,4)
    neuron.add_edge(3,5)
    #build spline tree using fit_spline_tree_invariant()
    spline_tree = neuron.fit_spline_tree_invariant()