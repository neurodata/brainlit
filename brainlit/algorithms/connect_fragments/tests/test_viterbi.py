# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 23:47:23 2021

@author: frede
"""
import numpy as np
import pandas as pd
import tifffile as tf
import brainlit
import pytest
from brainlit.algorithms.connect_fragments.dynamic_programming_viterbi import viterbi_algorithm
from pathlib import Path
from skimage import io, measure
from grid_generator import grid_gen, grid_gen3D

# To be removed later, only for testing purposes
import matplotlib.pyplot as plt

def testInit():   
    # alg.x == x will return a boolean matrix. The sum should be equal to
    #   dim1 * dim2 * dim3 if every position is True
    assert(np.sum(alg.image == img) == 10 * 10 * 2)
    assert(np.sum(alg.labels == lbls) == 10 * 10 * 2)
    
def testLineToLine():
    # Test in 2D, pseudo-3D
    # Compare the output of line_line_dist with manually computed distances
    assert(np.sqrt((0-3)**2+(2-3)**2) == alg.line_line_dist(1,3)[0])
    assert(np.sqrt((0-3)**2+(2-3)**2) == alg.line_line_dist(3,1)[0])
    assert(np.sqrt((6-7)**2+(5-7)**2) == alg.line_line_dist(3,4)[0])
    assert(np.sqrt((6-7)**2+(5-7)**2) == alg.line_line_dist(4,3)[0])
    with pytest.raises(ValueError):
        alg.line_line_dist(2,2)

    # Test in 2D, pseudo-3D, with resolution scaling of [0.5,0.5,0.5]
    # Compare the output of line_line_dist with manually computed distances
    assert(np.sqrt(((0-3)/2)**2+((2-3)/2)**2) == alg_half.line_line_dist(1,3)[0])
    assert(np.sqrt(((0-3)/2)**2+((2-3)/2)**2) == alg_half.line_line_dist(3,1)[0])
    assert(np.sqrt(((6-7)/2)**2+((5-7)/2)**2) == alg_half.line_line_dist(3,4)[0])
    assert(np.sqrt(((6-7)/2)**2+((5-7)/2)**2) == alg_half.line_line_dist(4,3)[0])
    with pytest.raises(ValueError):
        alg_half.line_line_dist(2,2)

    # Test in 3D
    # Compare the output of line_line_dist with manually computed distances
    assert(np.sqrt((3-96)**2+(0-99)**2+(0-99)**2) == alg1_3D.line_line_dist(1,3)[0])
    assert(np.sqrt((96-3)**2+(99-0)**2+(99-0)**2) == alg1_3D.line_line_dist(3,1)[0])

    with pytest.raises(ValueError):
        alg1_3D.line_line_dist(2,2)
    
    # Test in 3D, with resolution scaling of [0.5,0.5,0.5]
    # Compare the output of line_line_dist with manually computed distances
    assert(np.sqrt(((3-96)/2)**2+((0-99)/2)**2+((0-99)/2)**2) == alg1_3D_half.line_line_dist(1,3)[0])
    assert(np.sqrt(((96-3)/2)**2+((99-0)/2)**2+((99-0)/2)**2) == alg1_3D_half.line_line_dist(3,1)[0])

    with pytest.raises(ValueError):
        alg1_3D_half.line_line_dist(2,2)
    
def testLineToBlob():
    # Test in 2D, pseudo-3D
    # Compare the output of line-blob dist with manually computed distances
    # To a soma
    assert(np.sqrt((8-9)**2+(7-9)**2) == alg.line_blob_dist(4,5)[0])
    # Should go from an endpoint to the closest position on the big blob
    assert(np.sqrt((9-3)**2+(9-15)**2) == alg2.line_blob_dist(2,1)[0])
    with pytest.raises(ValueError):
        alg.line_blob_dist(2,2)
        
    # Test in 2D, pseudo-3D, with resolution scaling of [0.5,0.5,0.5]
    # Compare the output of line-blob dist with manually computed distances
    # To a soma
    assert(np.sqrt(((8-9)/2)**2+((7-9)/2)**2) == alg_half.line_blob_dist(4,5)[0])
    # Should go from an endpoint to the closest position on the big blob
    assert(np.sqrt(((9-3)/2)**2+((9-15)/2)**2) == alg2_half.line_blob_dist(2,1)[0])
    with pytest.raises(ValueError):
        alg.line_blob_dist(2,2)
        
    # Test in 3D
    # Should go from an endpoint to the closest position on the blob
    assert(np.sqrt((4-21)**2+(0-21)**2+(0-21)**2) == alg2_3D.line_blob_dist(1,3)[0])
    with pytest.raises(ValueError):
        alg2_3D.line_blob_dist(2,2)

    # Test in 3D, with resolution scaling of [0.5,0.5,0.5]
    # Should go from an endpoint to the closest position on the blob
    assert(np.sqrt(((4-21)/2)**2+((0-21)/2)**2+((0-21)/2)**2) == alg2_3D_half.line_blob_dist(1,3)[0])
    with pytest.raises(ValueError):
        alg2_3D_half.line_blob_dist(2,2)


def testPathIntensityCost():
    # Currently a stub, need to double-check what the proper form of the 
    # intensity score should be
    assert(1 == alg2.line_int((9,9,0),(12,12,0),2,3))
    
    print(alg.line_int((0,2,0),(3,3,0),1,3))
    
def testConnections():
    # Test if we are getting the correct connections
    alg.compute_all_dists()
    c = alg.connection_mat
    
    # 1 to x, x to 1
    np.testing.assert_equal(c[0][1][2], [0,2,0])
    np.testing.assert_equal(c[1][1][2], [0,9,0])
    np.testing.assert_equal(c[0][2][1], [0,9,0])
    np.testing.assert_equal(c[1][2][1], [0,2,0])
    
    np.testing.assert_equal(c[0][1][3], [0,2,0])
    np.testing.assert_equal(c[1][1][3], [3,3,0])
    np.testing.assert_equal(c[0][3][1], [3,3,0])
    np.testing.assert_equal(c[1][3][1], [0,2,0])
    
    np.testing.assert_equal(c[0][1][4], [0,2,0])
    np.testing.assert_equal(c[1][1][4], [7,7,0])
    np.testing.assert_equal(c[0][4][1], [7,7,0])
    np.testing.assert_equal(c[1][4][1], [0,2,0])
    
    # 2 to x, x to 2
    np.testing.assert_equal(c[0][2][3], [0,9,0])
    np.testing.assert_equal(c[1][2][3], [3,3,0])
    np.testing.assert_equal(c[0][3][2], [3,3,0])
    np.testing.assert_equal(c[1][3][2], [0,9,0])
    
    np.testing.assert_equal(c[0][2][4], [0,9,0])
    np.testing.assert_equal(c[1][2][4], [7,8,0])
    np.testing.assert_equal(c[0][4][2], [7,8,0])
    np.testing.assert_equal(c[1][4][2], [0,9,0])

    # 3 to x, x to 3
    np.testing.assert_equal(c[0][3][4], [6,5,0])
    np.testing.assert_equal(c[1][3][4], [7,7,0])
    np.testing.assert_equal(c[0][4][3], [7,7,0])
    np.testing.assert_equal(c[1][4][3], [6,5,0])
    
    # 4 to x, x to 4 is covered by other test blocks
    
    # SOMA CONNECTIONS
    # 5 to x, x to 5 
    np.testing.assert_equal(c[0][1][5], [0,2,0])
    np.testing.assert_equal(c[1][1][5], [9,9,0])
    np.testing.assert_equal(c[0][5][1], [0,0,0])
    np.testing.assert_equal(c[1][5][1], [0,0,0])
        
    np.testing.assert_equal(c[0][3][5], [6,5,0])
    np.testing.assert_equal(c[1][3][5], [9,9,0])
    np.testing.assert_equal(c[0][5][3], [0,0,0])
    np.testing.assert_equal(c[1][5][3], [0,0,0])
    
    np.testing.assert_equal(c[0][4][5], [7,8,0])
    np.testing.assert_equal(c[1][4][5], [9,9,0])
    np.testing.assert_equal(c[0][5][4], [0,0,0])
    np.testing.assert_equal(c[1][5][4], [0,0,0])
    
        # Cannot go from blob 2 to soma 5
    np.testing.assert_equal(c[0][2][5], [0,0,0])
    np.testing.assert_equal(c[1][2][5], [0,0,0])
    np.testing.assert_equal(c[0][5][2], [0,0,0])
    np.testing.assert_equal(c[1][5][2], [0,0,0])
    
    # Cost matrix checks
    for i in range(5):
        for j in range(5):
            if i in alg.somas.keys():
                assert(np.inf == alg.cost_mat_dist[i,j])
                assert(np.inf == alg.cost_mat_int[i,j])
            elif i == j:
                assert(np.inf == alg.cost_mat_dist[i,j])
                assert(np.inf == alg.cost_mat_int[i,j])

    # For manual debugging purposes
    connect_points = set()
    for i in range(1):
        for j in range(1,5):
            for k in range(1,5):
                if not (np.equal([0,0,0], c[i][j][k])).all():
                    connect_points.add(tuple(c[i][j][k]))

def testEndpoints():
    alg3.frags_to_lines_le_skel([2])
    # Labels 1 and 3 are lines for this 100x100 example
    
    ids1 = map(id, np.array(alg3.end_points[1]))
    ids3 = map(id, np.array(alg3.end_points[3]))
    
    # Check if both endpoints are found
    # Note that the manually typed endpoints were identified visually
    # in this simple example.
    assert(id(np.array([1,1,0])) in ids1)
    assert(id(np.array([48,48,0])) in ids1)
    
    assert(id(np.array([51,51,0])) in ids3)
    assert(id(np.array([97,97,0])) in ids3)
    
    
    print("Endpoints Tests: ", alg3.end_points)

def testEndpoints3D():
    alg3_3D.frags_to_lines_le_skel([4])
    # Labels 1 and 3 are lines for this 100x100 example
    
    ids1 = map(id, np.array(alg3_3D.end_points[1]))
    ids2 = map(id, np.array(alg3_3D.end_points[2]))
    ids3 = map(id, np.array(alg3_3D.end_points[3]))
    #ids4 = map(id, np.array(alg3_3D.end_points[4]))
    #ids5 = map(id, np.array(alg3_3D.end_points[2]))
                
    
    # Check if both endpoints are found
    # Note that the manually typed endpoints were identified visually
    # in this simple example.
    #assert(id(np.array([1,1,0])) in ids1)
    #assert(id(np.array([48,48,0])) in ids1)
    
    #assert(id(np.array([51,51,0])) in ids3)
    #assert(id(np.array([97,97,0])) in ids3)
    
    
    print("Endpoints Tests: ", alg3_3D.end_points)


def testViterbi():
    def print_path(alg, path):
        c = alg.connection_mat
        path_lbls = path[1]
        for i in range(len(path_lbls)-1):
            from_lbl = path_lbls[i]
            to_lbl = path_lbls[i+1]
            
            print(f"From {from_lbl} to {to_lbl}: {c[0][from_lbl][to_lbl]}, {c[1][from_lbl][to_lbl]}")
    
    print()
    
    print("========== Viterbi 2D ==========")
    
    alg.compute_all_dists()
    top_path, sorted_paths = alg.viterbi_frag(1, K=4, somas=alg.somas)
    print("------- PATHS -------")
    print(sorted_paths)
    print("----- BEST PATH -----")
    print(top_path)
    print_path(alg, top_path)   
    #print("----- BUGGY PATH -----")
    #print(sorted_paths[1][1])
    #print_path(alg, sorted_paths[1][1])
    
    print()
    
    alg1.compute_all_dists()
    print("_______ WITH DIFFERENT INTENSITY DATA _______")
    top_path1, sorted_paths1 = alg1.viterbi_frag(1, K=4, somas=alg1.somas)
    print("------- PATHS -------")
    print(sorted_paths1)
    print("----- BEST PATH -----")
    print(top_path1)
    print_path(alg1, top_path1)
    #print("----- BUGGY PATH -----")
    #print(sorted_paths1[1][1])
    #print_path(alg1, sorted_paths1[1][1])
    
    print("__________ K=10 TEST __________")
    top_path, sorted_paths = alg.viterbi_frag(1, K=10, somas=alg.somas)
    print("------- PATHS -------")
    print(sorted_paths)
    print("----- BEST PATH -----")
    print(top_path)
    print_path(alg, top_path)   
    
    print()

def testViterbi3D():
    def print_path(alg, path):
        c = alg.connection_mat
        path_lbls = path[1]
        for i in range(len(path_lbls)-1):
            from_lbl = path_lbls[i]
            to_lbl = path_lbls[i+1]
            
            print(f"From {from_lbl} to {to_lbl}: {c[0][from_lbl][to_lbl]}, {c[1][from_lbl][to_lbl]}")
    print()
    
    print("========== Viterbi 3D ==========")
    
    alg4_3D.compute_all_dists()
    top_path, sorted_paths = alg4_3D.viterbi_frag(1, K=6, somas=alg4_3D.somas)
    print("------- PATHS -------")
    print(sorted_paths)
    print("----- BEST PATH -----")
    print(top_path)
    print_path(alg4_3D, top_path)
    
    print()
    
    print("__________ K=10 TEST __________")
    top_path, sorted_paths = alg4_3D.viterbi_frag(1, K=10, somas=alg4_3D.somas)
    print("------- PATHS -------")
    print(sorted_paths)
    print("----- BEST PATH -----")
    print(top_path)
    print_path(alg4_3D, top_path)
    
''' Setting up the image environment '''

img, lbls, _ , somas = grid_gen(10)
plt.figure()
plt.imshow(img[:,:,0])
alg = viterbi_algorithm(img, lbls, somas, [1,1,1])
# For the timebeing, manually do the endpoints
# NOTE: no endpoints for 2 or 5 because they are "blobs"
endpoints = {}
endpoints[1] = ((0,0,0),(0,2,0))
endpoints[3] = ((3,3,0),(6,5,0))
endpoints[4] = ((7,7,0),(7,8,0))
#alg.end_points = alg.frags_to_lines_le_skel()
alg.end_points = endpoints

alg_half = viterbi_algorithm(img,lbls,somas,[0.5,0.5,0.5])
alg_half.end_points = endpoints

img2, lbls2, _ , somas2 = grid_gen(20)
alg2 = viterbi_algorithm(img2, lbls2, somas2, [1,1,1])
# manually do the endpoints
# NOTE: no endpoints for 1 or 4 because they are blobs
endpoints2 = {}
endpoints2[2] = ((1,1,0),(9,9,0))
endpoints2[3] = ((12,12,0),(15,15,0))
alg2.end_points = endpoints2

alg2_half = viterbi_algorithm(img2, lbls2, somas2, [0.5,0.5,0.5])
alg2_half.end_points = endpoints2

# 100x100 example for endpoints
img3, lbls3, _, somas3 = grid_gen(100)
alg3 = viterbi_algorithm(img3, lbls3, somas3, [1,1,1])

img1_3D, lbls1_3D, _, somas1_3D = grid_gen3D(100)
alg1_3D = viterbi_algorithm(img1_3D, lbls1_3D, somas1_3D, [1,1,1])
endpts1_3D = {}
endpts1_3D[1] = ((0,0,0),(3,0,0))
endpts1_3D[3] = ((99,99,99),(96,99,99))
alg1_3D.end_points = endpts1_3D

alg1_3D_half = viterbi_algorithm(img1_3D, lbls1_3D, somas1_3D, [0.5,0.5,0.5])
alg1_3D_half.end_points = endpts1_3D


img2_3D, lbls2_3D, _, somas2_3D,= grid_gen3D(25)
alg2_3D = viterbi_algorithm(img2_3D, lbls2_3D, somas2_3D, [1,1,1])
endpts2_3D = {}
endpts2_3D[1] = ((0,0,0),(4,0,0))
alg2_3D.end_points = endpts2_3D
alg2_3D_half = viterbi_algorithm(img2_3D, lbls2_3D, somas2_3D, [0.5,0.5,0.5])
alg2_3D_half.end_points = endpts2_3D

#100x100 example for endpoints
img3_3D, lbls3_3D, _, somas3_3D = grid_gen3D(101)
alg3_3D = viterbi_algorithm(img3_3D, lbls3_3D, somas3_3D, [1,1,1])

#10x10 high intensity loop test
img1, lbls1, _ , somas1 = grid_gen(102)
plt.figure()
plt.imshow(img1[:,:,0])
plt.figure()
plt.imshow(img[:,:,0])
alg1 = viterbi_algorithm(img1, lbls1, somas1, [1,1,1])
# For the timebeing, manually do the endpoints
# NOTE: no endpoints for 2 or 5 because they are "blobs"
endpoints = {}
endpoints[1] = ((0,0,0),(0,2,0))
endpoints[3] = ((3,3,0),(6,5,0))
endpoints[4] = ((7,7,0),(7,8,0))
#alg.end_points = alg.frags_to_lines_le_skel()
alg1.end_points = endpoints


#100x100 viterbi test
img4_3D, lbls4_3D, _, somas4_3D = grid_gen3D(15)

alg4_3D = viterbi_algorithm(img4_3D, lbls4_3D, somas4_3D, [1,1,1])
# For the timebeing, manually do the endpoints
# NOTE: no endpoints for 2 or 5 because they are "blobs"
endpoints = {}
endpoints[1] = ((0,0,0),(3,0,0))
endpoints[2] = ((4,3,0),(4,7,2))
endpoints[3] = ((6,8,4),(7,10,7))
endpoints[4] = ((8,12,7),(12,13,11))
#alg.end_points = alg.frags_to_lines_le_skel()
alg4_3D.end_points = endpoints
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_pts1 = (np.nonzero(lbls4_3D==1))
plot_pts2 = (np.nonzero(lbls4_3D==2))
plot_pts3 = (np.nonzero(lbls4_3D==3))
plot_pts4 = (np.nonzero(lbls4_3D==4))
plot_pts5 = (np.nonzero(lbls4_3D==5))

ax.scatter(plot_pts1[0],plot_pts1[1],plot_pts1[2])
ax.scatter(plot_pts2[0],plot_pts2[1],plot_pts2[2])
ax.scatter(plot_pts3[0],plot_pts3[1],plot_pts3[2])
ax.scatter(plot_pts4[0],plot_pts4[1],plot_pts4[2])
ax.scatter(plot_pts5[0],plot_pts5[1],plot_pts5[2])


testInit()
testLineToLine()
testLineToBlob()
testPathIntensityCost()
testConnections()
testEndpoints()
testEndpoints3D()
testViterbi()
testViterbi3D()
#print(lbls)