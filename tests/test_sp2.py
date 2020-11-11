import pytest
import numpy as np
from brainlit.algorithms.connect_fragments.fit_spline import GeometricGraph
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep


# adding nodes and edges
# 1. nodes should be defined under nodes
# 2. loc should be np.array not list
# 3. nodes should be added with two arguments: samp, loc
# 4. edges should be added in pairs: samp, parent
soma = [0, -100, 200]

neuron = GeometricGraph()
# add nodes
neuron.add_node(1, loc=np.array([100, 0, 200]))
neuron.add_node(2, loc=np.array([100, 100, 200]))
neuron.add_node(3, loc=np.array([0, 200, 200]))
neuron.add_node(4, loc=np.array([200, 300, 200]))
# add edges
neuron.add_edge(1, 2)
neuron.add_edge(2, 3)
neuron.add_edge(2, 4)
# first path parameters created by `splprep`
path = [1, 2, 4]
x = np.zeros((len(path), 3))
for row, node in enumerate(path):
    x[row, :] = neuron.nodes[node]["loc"]
m = x.shape[0]
diffs = np.diff(x, axis=0)
diffs = np.linalg.norm(diffs, axis=1)
diffs = np.cumsum(diffs)
diffs = np.concatenate(([0], diffs))
k = np.amin([m - 1, 5])
tck_scipy, u_scipy = splprep([x[:, 0], x[:, 1], x[:, 2]], u=diffs, k=k)
print("tck sci", tck_scipy)
# print("u sci",u_scipy)
# first path created by `fit_spline_tree_invariant`
spline_tree = neuron.fit_spline_tree_invariant()
spline = spline_tree.nodes[0]["spline"]
print("tck fit", spline[0])
u_fit = spline[1][0]
tck_fit = spline[0][0]

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

spline_tree = neuron.fit_spline_tree_invariant()

PATHS=[]
for node in spline_tree.nodes:
    path = spline_tree.nodes[node]["path"]
    PATHS.append(path)
print(PATHS)


locs = np.zeros((len(path),3))
for p,point in enumerate(path):
    locs[p,:] = neuron_long4.nodes[point]["loc"]
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

print(PATHS)

#spline
for node in spline_tree.nodes:
    #print(len(spline_tree.nodes))
    path = spline_tree.nodes[node]["path"]
    locs = np.zeros((len(path),3))
    for p,point in enumerate(path):
        locs[p,:] = neuron_long4.nodes[point]["loc"]

    spline = spline_tree.nodes[node]["spline"]
    u = spline[1]
    print("old u:",u)
    u = np.arange(u[0], u[-1]+0.9, 1)
    print("new u:",u)
    tck = spline[0]
    pts = splev(u, tck)
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    ax.scatter(locs[:,0], locs[:,1], locs[:,2], 'blue')
    ax.plot(pts[0], pts[1], pts[2], 'red')
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
