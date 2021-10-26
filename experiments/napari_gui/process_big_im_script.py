from tqdm import tqdm
import pickle
from pathlib import Path
import os

from skimage import io, measure
import numpy as np 
import h5py
from brainlit.preprocessing import image_process
from brainlit.algorithms.connect_fragments import dynamic_programming_viterbi2
import scipy.ndimage as ndi 
from sklearn.metrics import pairwise_distances_argmin_min
import napari
import networkx as nx
import time


# data_dir = "/data/tathey1/mouselight/"
# im_path = os.path.join(data_dir, "250.tif")

# probs_path = os.path.join(data_dir, "250_probs.tif")
# # f = h5py.File(probs_path, 'r')


# res = [0.3,0.3,1]
# soma_coords = [[417, 417, 125]]

# t1 = time.perf_counter()
# print("reading files")

# im_og = io.imread(im_path, plugin="tifffile")
# im_processed = io.imread(probs_path, plugin="tifffile")
# # im_processed = f.get('exported_data')
# # im_processed = im_processed[:,:,:,1]

# print(f"image shape: {im_og.shape}")
# print(f"read files in {time.perf_counter()-t1} seconds")
# t1 = time.perf_counter()

# threshold = 0.9
# labels = measure.label(im_processed > threshold)

# print(f"labeled image in {time.perf_counter()-t1} seconds")
# t1 = time.perf_counter()

# new_labels = image_process.split_frags(soma_coords, labels, im_processed, threshold, res, verbose=True)

# io.imsave("/data/tathey1/mouselight/1_labels.tif", new_labels)

# print(f"made fragments in {time.perf_counter()-t1} seconds")
# print(f"{np.unique(new_labels).size} fragments")
# t1 = time.perf_counter()

# _, soma_lbls = image_process.label_points(new_labels, soma_coords, res)
# soma_lbl = soma_lbls[0]
# soma_mask = new_labels == soma_lbl

# mpnp = dynamic_programming_viterbi2.most_probable_neuron_path(image=im_og.astype(float), labels=new_labels, soma_lbls=soma_lbls, resolution=(0.3, 0.3, 1), coef_dist=10, coef_curv=1000)

# print(f"made viterbi object in {time.perf_counter()-t1} seconds")
# with open("/data/tathey1/mouselight/1_viterbi_begin.pkl", 'wb') as handle:
#     pickle.dump(mpnp, handle)
# t1 = time.perf_counter()

# mpnp.frags_to_lines()

# print(f"made states in {time.perf_counter()-t1} seconds")
# with open("/data/tathey1/mouselight/1_viterbi_states.pkl", 'wb') as handle:
#     pickle.dump(mpnp, handle)
# t1 = time.perf_counter()

# mpnp.reset_dists(type="all")
# mpnp.compute_all_costs_dist(point_point_func=mpnp.point_point_dist, point_blob_func=mpnp.point_blob_dist)

# print(f"made dist cost in {time.perf_counter()-t1} seconds")
# t1 = time.perf_counter()

# mpnp.compute_all_costs_int()

# print(f"made int cost in {time.perf_counter()-t1} seconds")
# t1 = time.perf_counter()

# mpnp.create_nx_graph()

# with open("/data/tathey1/mouselight/1_viterbi_nx.pkl", 'wb') as handle:
#     pickle.dump(mpnp, handle)

# print(f"made graph in {time.perf_counter()-t1} seconds")

with open("/data/tathey1/mouselight/1_viterbi_nx.pkl", 'rb') as handle:
    mpnp = pickle.load(handle)
t1 = time.perf_counter()

#path_states = nx.shortest_path(mpnp.nxGraph, 1, 11, weight='weight')

a = sorted(mpnp.nxGraph.degree, key=lambda x: x[1], reverse=True)
nx.bfs_edges(mpnp.nxGraph, a[0][0])

print(f"computed path in {time.perf_counter()-t1} seconds")
t1 = time.perf_counter()

with open("/data/tathey1/mouselight/1_viterbi.pkl", 'wb') as handle:
    pickle.dump(mpnp, handle)

print(f"saved viterbi in {time.perf_counter()-t1} seconds")
t1 = time.perf_counter()

