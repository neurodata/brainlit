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

import matplotlib.pyplot as plt

class viterbi_algorithm:
    def __init__(self, image, labels, soma_labels, resolution=[1,1,1]):

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
    
    def compute_bounds(self, label, pad):
        """ Currently zmin and zmax are hardcoded as 0,1 for this simple image """
        """compute coordinates of bounding box around a masked object, with given padding
        Args:
            label (np.array): mask of the object
            pad (float): padding around object in um
        Returns:
            [ints]: integer coordinates of bounding box
        """
        labels = self.labels
        res = self.res

        r = np.any(label, axis=(1, 2))
        c = np.any(label, axis=(0, 2))
        z = np.any(label, axis=(0, 1))
        rmin, rmax = np.where(r)[0][[0, -1]]
        rmin = np.amax((0, math.floor(rmin - pad / res[0])))
        rmax = np.amin((labels.shape[0], math.ceil(rmax + (pad + 1) / res[0])))
        cmin, cmax = np.where(c)[0][[0, -1]]
        cmin = np.amax((0, math.floor(cmin - (pad) / res[1])))
        cmax = np.amin((labels.shape[1], math.ceil(cmax + (pad + 1) / res[1])))
        #zmin, zmax = np.where(z)[0][[0, -1]]
        #zmin = np.amax((0, math.floor(zmin - (pad) / res[2])))
        #zmax = np.amin((labels.shape[2], math.ceil(zmax + (pad + 1) / res[2])))
        zmin = 0
        zmax = 2
        return int(rmin), int(rmax), int(cmin), int(cmax), int(zmin), int(zmax)
    
        
    def frags_to_lines_le_skel(self, soma_lbls=[]):
        """Relies on the assumption that self.labels has values as if it came from measure.label"""
        labels = self.labels

        end_points = {}

        # Note: we want label 1 onwards, because 0 is background
        for component in np.unique(labels)[1:]:
            print(component)
            # Skip if it is a soma
            if component in soma_lbls:
                continue
            
            # Mask the current component
            mask = labels == component
            
            # The mask is relatively sparse, so we need to cut out only the 
            # relevant regions with labels
            rmin, rmax, cmin, cmax, zmin, zmax = self.compute_bounds(mask, pad=1)
            mask = mask[rmin:rmax,cmin:cmax,zmin:zmax]

            skel = morphology.skeletonize_3d(mask)

            coords_mask = np.argwhere(mask)

            coords_skel = np.argwhere(skel)
            print(coords_mask)
            if len(coords_skel) < 4:
                coords = coords_mask
            else:
                coords = coords_skel

            embedding = SpectralEmbedding(n_components=1).fit_transform(coords)

            amax = np.argmax(embedding)
            amin = np.argmin(embedding)
            a = coords[amax, :]
            b = coords[amin, :]

            end_points[component] = (a, b)
            
        print(f"{len(end_points.keys())} out of {len(np.unique(labels)[1:])} are lines")

        self.end_points = end_points
        components = set(np.unique(labels)[1:])
        components_lines = set(end_points.keys())
        self.not_lines = components.difference(components_lines)

    
    def line_line_dist(self, lbl1, lbl2):
        """
        Args:
            lbl1 ([type]): [non-soma component]
            lbl2 ([type]): [non-soma component]
        """
        if lbl1 == lbl2:
            raise ValueError(f"Cannot compute distance between {lbl1} and {lbl2}")
        
        ends1 = self.end_points[lbl1]
        ends2 = self.end_points[lbl2]
        
        # Compute the euclidean distance between each endpoint
        d1 = np.linalg.norm(np.subtract(ends1[0], ends2[0]))
        d2 = np.linalg.norm(np.subtract(ends1[0], ends2[1]))
        d3 = np.linalg.norm(np.subtract(ends1[1], ends2[0]))
        d4 = np.linalg.norm(np.subtract(ends1[1], ends2[1]))
        
        idx = np.argmin([d1, d2, d3, d4])

        if idx == 0:
            loc1, loc2 = ends1[0], ends2[0]
        elif idx == 1:
            loc1, loc2 = ends1[0], ends2[1]
        elif idx == 2:
            loc1, loc2 = ends1[1], ends2[0]
        elif idx == 3:
            loc1, loc2 = ends1[1], ends2[1]

        self.connection_mat[0][lbl1, lbl2, :] = loc1
        self.connection_mat[1][lbl1, lbl2, :] = loc2

        dist_cost = np.amin([d1, d2, d3, d4])
        return dist_cost, loc1, loc2
    
    def line_blob_dist(self, lbl1, lbl2):
        """
        Args:
            lbl1 ([type]): [line component]
            lbl2 ([type]): [soma component]
        """
        if lbl1 == lbl2:
            raise ValueError(f"Cannot compute distance between {lbl1} and {lbl2}")

        labels = self.labels

        connection_mat = self.connection_mat

        # lbl1 is a line, lbl2 is a soma or blob
        line_pts = self.end_points[lbl1]
        blob_lbl = lbl2

        lowest_cost = np.inf

        label_nonline = labels == blob_lbl
        # for all endpoints of the line
        for endpt in line_pts:
            if blob_lbl in self.somas.keys():
                # Soma is represented as a single point in space,
                # cast as a list with 1 tuple object
                coords = [self.somas[blob_lbl]]
            else:
                coords = np.argwhere(
                    label_nonline ^ ndi.morphology.binary_erosion(label_nonline)
                )
                self.soma_locs[blob_lbl] = coords

            dists = np.linalg.norm(np.subtract(coords, endpt))
            dist_cost = np.amin(dists)
            
            # find minimum based on distance cost
            if dist_cost < lowest_cost:
                lowest_cost = dist_cost
                endpt_lowest = endpt
                blob_lowest = coords[np.argmin(dists)]

        # set connection_mat points
        connection_mat[0][lbl1, lbl2, :] = endpt_lowest
        if labels[endpt_lowest] != lbl1:
            raise ValueError(
                f"Lowest cost point: {endpt_lowest} has label {labels[endpt_lowest]}, not {lbl1}"
            )
        connection_mat[1][lbl1, lbl2, :] = blob_lowest
        if (labels[blob_lowest] != lbl2):
            raise ValueError("Error in setting connection_mat")

        return lowest_cost, endpt_lowest, blob_lowest

