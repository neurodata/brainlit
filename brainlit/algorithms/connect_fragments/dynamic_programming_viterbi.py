# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 21:47:47 2021

@author: frede
"""

import numpy as np
from itertools import combinations
from graspy.match import GraphMatch as GMP
import networkx as nx
import scipy.ndimage as ndi
from sklearn.metrics import pairwise_distances_argmin_min
import warnings
import math
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.manifold import SpectralEmbedding
from mouselight_code.src.swc2voxel import Bresenham3D

# from mouselight_code.src.visualize import napari_viewer
from scipy import spatial
from scipy.special import logsumexp
from tqdm import tqdm
from sklearn.decomposition import PCA
from skimage import measure
from skimage import morphology
import time

class viterbi_algorithm:
    def __init__(self, image, labels, soma_labels, resolution):

        num_components = np.amax(labels)
        self.num_components = num_components
        self.image = image
        self.labels = labels
        self.somas = soma_labels
        
        self.cost_mat_dist = np.ones((num_components+1, num_components+1)) * -1
        self.cost_mat_int = np.ones((num_components+1, num_components+1)) * -1

        self.res = resolution

        np.fill_diagonal(self.cost_mat_dist, np.inf)
        np.fill_diagonal(self.cost_mat_int, np.inf)

        # following object contains voxel coordinates of connection points
        # two elements in tuple refers to two objects
        # 0th and 1st dim refer to object numbers, 2nd axis is the voxel coordinates
        self.connection_mat = (
            np.zeros((num_components+1, num_components+1, 3), dtype=int),
            np.zeros((num_components+1, num_components+1, 3), dtype=int),
        )

        self.end_points = None
        self.not_lines = None
