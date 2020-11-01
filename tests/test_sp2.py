import pytest
import numpy as np
from brainlit.algorithms.connect_fragments.fit_spline import GeometricGraph
import networkx as nx
import matplotlib.pyplot as plt


#adding nodes and edges
#1. nodes should be defined under nodes
#2. loc should be np.array not list
#3. nodes should be added with two arguments: samp, loc
#4. edges should be added in pairs: samp, parent


neuron=GeometricGraph()
neuron.add_node(1,loc=np.array([100,100,200]))
neuron.add_node(2,loc=np.array([200,0,200]))
neuron.add_node(3,loc=np.array([200,300,200]))
neuron.add_node(4,loc=np.array([300,400,200]))
neuron.add_node(5,loc=np.array([100,500,200]))
neuron.add_node(6,loc=np.array([100,500,200]))
soma=[100,100,200]
#neuron.add_edge(1,-1)
neuron.add_edge(1,2)
neuron.add_edge(2,3)
neuron.add_edge(3,4)
neuron.add_edge(3,5)
#neuron.add_edge(3,5)
#neuron.add_edge(1,5)
#neuron.add_edge(1,4)

spline_tree = neuron.fit_spline_tree_invariant()

#print(type(neuron.nodes[1]["loc"]))
#print(neuron.nodes)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

spline_tree = neuron.fit_spline_tree_invariant()
#spline_tree1 = spline_tree.to_undirected()

for node in spline_tree.nodes:
    path = spline_tree.nodes[node]["path"]
    locs = np.zeros((len(path),3))
    for p,point in enumerate(path):
        locs[p,:] = neuron.nodes[point]["loc"]
    ax.scatter(locs[:,0], locs[:,1], locs[:,2], c='b', s=9)
    ax.plot(locs[:,0], locs[:,1], locs[:,2], 'b-')
ax.scatter(soma[0],soma[1],soma[2], c='r', s=500)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# Hide grid lines
ax.grid(False)

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')
plt.show()
