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
        self.sigma = 1/20
    
    def viterbi_frag(self, start_lbl, K, somas):
        """Run Viterbi algorithm on image that has been masked into connected components.
        Args:
            start_lbl (int): starting component
            K (int): number of iterations
            somas (list): list of components that are cell bodies
        Returns:
            [type]: [description]
        """

        # Initialize dictionary of paths
        # Start at state 0
        # Dictionary value:
        #   - First element of tuple is the state
        #   - Second is a tuple that contains path length and path to state
        paths_k = {start_lbl: (0, [start_lbl])}

        all_paths = []

        for step in np.arange(K):
            all_paths.append(paths_k)

            paths_k, closest_state = self.viterbi_frag_next_layer(
                paths_k,
                somas,
            )

        sort_paths = sorted(paths_k.items(), key=lambda x: x[1][0])
        top_paths = [entry[1] for entry in sort_paths[:1]]
        return top_paths[0], sort_paths
    
    def viterbi_frag_next_layer(self, paths_k, somas):
        num_components = self.num_components

        # This dictionary will store the paths for the next level
        paths_k1 = {}
        # Init closest state and cost to that state
        closest_state = -1
        closest_state_len = np.inf
        # For each possible current state
        for state in range(1, num_components + 1):
            shortest_path = []
            shortest_length = np.Inf
            # For each possible previous state
            for prev_state in paths_k.keys():
                path = paths_k[prev_state][1].copy()
                # Calculate the cost to traverse
                length = paths_k[prev_state][0] + self.path_cost(
                    prev_state, state, path, somas
                )

                path.append(state)

                if length < shortest_length:
                    shortest_length = length
                    shortest_path = path
                    
            if shortest_length < closest_state_len:
                closest_state = state
                closest_state_len = shortest_length
                
            paths_k1[state] = (shortest_length, shortest_path)

        return paths_k1, closest_state
    
    def path_cost(self, prev_state, state, path, somas):
        cost_dist = self.cost_mat_dist[prev_state, state]
        cost_int = self.cost_mat_int[prev_state, state]

        if self.path_has_connection([prev_state, state], path):
            cost_int = np.inf
 
        total_cost = cost_dist + cost_int
        print(prev_state, state)
        print(f"{cost_dist} + {cost_int} = {total_cost}");
        return total_cost
    

    def path_has_connection(self,connection,path):
        l = len(connection)
        
        # If the path is < 2 nodes long
        if l > len(path):
            return False

        for i in range(len(path) - l + 1):
            if path[i : i + l] == connection:
                print("Connection duplicate")
                return True
        return False
    
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
    
        
    def frags_to_lines_le_skel(self, nonline_labels=[]):
        """Relies on the assumption that self.labels has values as if it came from measure.label"""
        labels = self.labels

        end_points = {}

        # Note: we want label 1 onwards, because 0 is background
        for component in np.unique(labels)[1:]:
            # Skip if it is a soma
            if component in nonline_labels:
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
            
            if len(coords_skel) < 4:
                coords = coords_mask
            else:
                coords = coords_skel

            embedding = SpectralEmbedding(n_components=1).fit_transform(coords)
            amax = np.argmax(embedding)
            amin = np.argmin(embedding)
            a = coords[amax, :]
            b = coords[amin, :]
            a = np.add(a,[rmin, cmin, zmin])
            b = np.add(b,[rmin, cmin, zmin])

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

        dist_cost = np.amin([d1, d2, d3, d4])
        return dist_cost, loc1, loc2
    
    def line_blob_dist(self, lbl1, lbl2):
        """
        Args:
            lbl1 ([type]): [line component]
            lbl2 ([type]): [soma/blob component]
        """
        if lbl1 == lbl2:
            raise ValueError(f"Cannot compute distance between {lbl1} and {lbl2}")

        labels = self.labels

        # lbl1 is a line, lbl2 is a soma or blob
        line_pts = self.end_points[lbl1]
        blob = lbl2

        lowest_cost = np.inf

        label_nonline = labels == blob
        # for all endpoints of the line
        for endpt in line_pts:
            if blob in self.somas.keys():
                # Soma is represented as a single point in space,
                # cast as a list with 1 tuple object
                coords = self.somas[blob]
            else:
                # Cut out only the boundary of the blob for distance computation
                coords = np.argwhere(
                    label_nonline ^ ndi.morphology.binary_erosion(label_nonline)
                )

            dists = np.linalg.norm(np.subtract(coords, endpt),axis=1)
            dist_cost = np.amin(dists)
            
            # find minimum based on distance cost
            if dist_cost < lowest_cost:
                lowest_cost = dist_cost
                endpt_lowest = endpt
                blob_lowest = coords[np.argmin(dists)]

        if labels[endpt_lowest[0],endpt_lowest[1],endpt_lowest[2]] != lbl1:
            raise ValueError(
                f"Lowest cost point: {endpt_lowest} has label {labels[endpt_lowest]}, not {lbl1}"
            )

        return lowest_cost, endpt_lowest, blob_lowest

    def line_int(self, loc1, loc2, lbl1, lbl2):

        # Use bresenham3D to "draw" a line in 3D
        xlist, ylist, zlist = Bresenham3D(
            loc1[0], loc1[1], loc1[2], loc2[0], loc2[1], loc2[2]
        )
        # Calculate the intensity cost along the line
        ints = self.image[xlist, ylist, zlist]

        # remove first and last voxels, which are part of foreground
        ints = ints[1:-1]
        #print(lbl1,lbl2,ints)
        mu1 = 2  # np.mean(image[labels == lbl1])

        # Need to check about this
        #int_cost = (mu1 ** 2 - 2 * mu1 * np.mean(ints)) / self.sigma
        
        int_cost = 1/(np.mean(ints)+1)
        
        return int_cost
    
    def compute_all_dists(self):
        for lbl1 in range(1, self.num_components + 1):
            for lbl2 in range(lbl1, self.num_components + 1):
                
                skip_connection = False
                
                if lbl2 == lbl1:
                    continue
                
                if lbl1 in self.end_points.keys() and lbl2 in self.end_points.keys():
                    # Line to line
                    dist, loc1, loc2 = self.line_line_dist(lbl1, lbl2)
                    int_cost = self.line_int(loc1, loc2, lbl1, lbl2)
                    
                # One of them is a blob
                elif lbl1 in self.end_points.keys():
                    # lbl1 to soma (lbl2)
                    dist, loc1, loc2 = self.line_blob_dist(lbl1, lbl2)
                    int_cost = self.line_int(loc1, loc2, lbl1, lbl2)

                elif lbl2 in self.end_points.keys():
                    # lbl2 to soma (lbl1)
                    dist, loc2, loc1 = self.line_blob_dist(lbl2, lbl1)
                    int_cost = self.line_int(loc1, loc2, lbl1, lbl2)

                # Both are blobs
                else:
                    dist = np.inf
                    int_cost = np.inf
                    skip_connection = True
                    
                # Distance cost is symmetric
                self.cost_mat_dist[lbl1, lbl2] = dist
                self.cost_mat_dist[lbl2, lbl1] = dist
            
                # Int cost is symmetric
                self.cost_mat_int[lbl1, lbl2] = int_cost
                self.cost_mat_int[lbl2, lbl1] = int_cost
                
                if not skip_connection:
                    # Set the forward connection
                    self.connection_mat[0][lbl1, lbl2] = loc1
                    self.connection_mat[1][lbl1, lbl2] = loc2
    
                    # Set the backward connection
                    self.connection_mat[0][lbl2, lbl1] = loc2
                    self.connection_mat[1][lbl2, lbl1] = loc1

                if dist+int_cost < 0:
                    warnings.warn(
                        f"Negative cost between {lbl1} to {lbl2} from: dist - {dist}, intensity - {int_cost}"
                    )

        for soma in self.somas.keys():
            # Going from soma to anything else is impossible, 
            # as we want to connect foreground to somas
            self.cost_mat_dist[soma, :] = np.inf
            self.cost_mat_int[soma, :] = np.inf
            # Soma to soma is 0
            self.cost_mat_dist[soma, soma] = 0
            self.cost_mat_int[soma, soma] = 0
            # Connection from soma outwards should be 0'd
            self.connection_mat[0][soma, :] = [0,0,0]
            self.connection_mat[1][soma, :] = [0,0,0]

            self.connection_mat[0][soma, :] = [0,0,0]
            self.connection_mat[1][soma, :] = [0,0,0]


        for lbl1 in range(1, self.num_components + 1):
            if lbl1 in self.somas.keys():
                denom = 0
            else:
                denom = logsumexp(-1 * self.cost_mat_dist[lbl1, 1:])
            self.cost_mat_dist[lbl1, 1:] = self.cost_mat_dist[lbl1, 1:] + denom

