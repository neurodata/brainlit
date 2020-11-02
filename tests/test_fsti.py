import pytest
from brainlit.algorithms.connect_fragments.fit_spline import GeometricGraph
import networkx as nx
import numpy as np

def test_loc():
    """Nodes should be defined in 'loc' attribute
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
    with pytest.raises(KeyError):
        spline_tree = neuron.fit_spline_tree_invariant()

def test_loc_np():
    """Nodes 'loc' should be a numpy.array
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
    with pytest.raises(TypeError):
        spline_tree = neuron.fit_spline_tree_invariant()

def test_loc_3d():
    """Nodes 'loc' should be 3-dimensional
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
    with pytest.raises(ValueError):
        spline_tree = neuron.fit_spline_tree_invariant()

def test_edgcov():
    """check if every node is assigned to at least one edge
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
    with pytest.raises(ValueError, match=r"The graph is not edge-covering"):
        spline_tree = neuron.fit_spline_tree_invariant()
    

def test_cycle():
    """check if the geometric graph contains undirected cycle(s)
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
    with pytest.raises(ValueError):
        spline_tree = neuron.fit_spline_tree_invariant()

def test_disconnect():
    """check if the geometric graph contains disconnected segment(s)
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
    with pytest.raises(ValueError):
        spline_tree = neuron.fit_spline_tree_invariant()

def test_DupNodLoc():
    """check if every node is unique in location
    """

    neuron=GeometricGraph()
    #add nodes
    neuron.add_node(1,loc=np.array([100,100,200]))
    neuron.add_node(2,loc=np.array([200,0,200]))
    neuron.add_node(3,loc=np.array([200,300,200]))
    neuron.add_node(4,loc=np.array([300,400,200]))
    neuron.add_node(5,loc=np.array([100,500,200]))
    neuron.add_node(6,loc=np.array([100,500,200]))
    #define soma
    soma=[100,100,200]
    #add edges
    neuron.add_edge(2,1)
    neuron.add_edge(2,3)
    neuron.add_edge(3,4)
    neuron.add_edge(3,5)
    neuron.add_edge(3,6)
    #build spline tree using fit_spline_tree_invariant()
    with pytest.raises(ValueError):
        spline_tree = neuron.fit_spline_tree_invariant()    