import pytest
import numpy as np
from brainlit.algorithms.connect_fragments.fit_spline import GeometricGraph
import networkx as nx

# nodes should be added one by one (not as an array)
neuron=GeometricGraph()
neuron.add_node(1,loc=[100,100,200])
neuron.add_node(2,loc=[200,0,200])
neuron.add_node(3,loc=[200,300,200])
neuron.add_node(4,loc=[300,400,200])
neuron.add_node(5,loc=[100,500,200])

neuron.add_edge(1,-1)
neuron.add_edge(2,1)
neuron.add_edge(3,2)
neuron.add_edge(4,2)
neuron.add_edge(5,2)
#x=(0,2,5,6,8)
#y=(0,2,10,5,2)
#z=(0,0,0,0,0)
#samp=(1,2,3,4,5)
#par=(-1,1,2,3,4)
print(neuron.nodes[1]["loc"])
spline_tree = neuron.fit_spline_tree_invariant()
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
spline_tree = neuron.fit_spline_tree_invariant()
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
"""