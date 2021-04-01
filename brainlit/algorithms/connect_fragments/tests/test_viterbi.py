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

# To be removed later, only for testing purposes
import matplotlib.pyplot as plt

def testInit():   
    # alg.x == x will return a boolean matrix. The sum should be equal to
    #   dim1 * dim2 * dim3 if every position is True
    assert(np.sum(alg.image == img) == 10 * 10 * 2)
    assert(np.sum(alg.labels == lbls) == 10 * 10 * 2)
    
def testLineToLine():
    # Compare the output of line_line_dist with manually computed distances
    assert(np.sqrt((0-3)**2+(2-3)**2) == alg.line_line_dist(1,3)[0])
    assert(np.sqrt((0-3)**2+(2-3)**2) == alg.line_line_dist(3,1)[0])
    assert(np.sqrt((6-7)**2+(5-7)**2) == alg.line_line_dist(3,4)[0])
    assert(np.sqrt((6-7)**2+(5-7)**2) == alg.line_line_dist(4,3)[0])
    with pytest.raises(ValueError):
        alg.line_line_dist(2,2)


def testLineToLine3D():
    # Compare the output of line_line_dist with manually computed distances
    assert(np.sqrt((3-96)**2+(0-99)**2+(0-99)**2) == alg1_3D.line_line_dist(1,3)[0])
    assert(np.sqrt((96-3)**2+(99-0)**2+(99-0)**2) == alg1_3D.line_line_dist(3,1)[0])

    with pytest.raises(ValueError):
        alg1_3D.line_line_dist(2,2)
    
def testLineToBlob():
    # Compare the output of line-blob dist with manually computed distances
    
    # To a soma
    assert(np.sqrt((8-9)**2+(7-9)**2) == alg.line_blob_dist(4,5)[0])

    # Should go from an endpoint to the closest position on the big blob
    assert(np.sqrt((9-3)**2+(9-15)**2) == alg2.line_blob_dist(2,1)[0])
    with pytest.raises(ValueError):
        alg.line_blob_dist(2,2)
        
def testLineToBlob3D():
    print( alg2_3D.line_blob_dist(1,3))
    # Should go from an endpoint to the closest position on the blob
    assert(np.sqrt((4-21)**2+(0-21)**2+(0-21)**2) == alg2_3D.line_blob_dist(1,3)[0])
    with pytest.raises(ValueError):
        alg2_3D.line_blob_dist(2,2)


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

def grid_gen(grid_id=10):
    if grid_id == 10:
        ''' Generates a 10x10x2 grid with a small assortment of labels'''
        # 10x10x2 image, with 1 color channel greyscale
        grid = np.zeros((10,10,2))
        labels = np.zeros((10,10,2))
        grid[0,9,0] = 255
        grid[0,0,0] = 255
        grid[0,1,0] = 255
        grid[0,2,0] = 255
        grid[3,3,0] = 255
        grid[4,3,0] = 255
        grid[5,3,0] = 255
        grid[5,4,0] = 255
        grid[6,4,0] = 255
        grid[6,5,0] = 255
        grid[7,7,0] = 255
        grid[7,8,0] = 255
        grid[9,9,0] = 255 # Soma
        
        # Add this back in later. For simplicity, only the 0th layer of grid has labels
        #grid[:,:,1] = grid[:,:,0] 
        # Label the components
        labels, num = measure.label(grid, return_num=True)

        # Create intensity data
        grid[0,9,0] = 255 # Frag
        
        grid[0,0,0] = 150 # Frag
        grid[0,1,0] = 150 # Frag
        grid[0,2,0] = 150 # Frag
        
        grid[1,2,0] = 150
        grid[1,3,0] = 150
        grid[2,3,0] = 175
                
        grid[3,3,0] = 200 # Frag
        grid[4,3,0] = 200 # Frag
        grid[5,3,0] = 200 # Frag
        grid[5,4,0] = 200 # Frag
        grid[6,4,0] = 200 # Frag
        grid[6,5,0] = 200 # Frag
        
        grid[6,6,0] = 215 
        grid[7,6,0] = 215 

        grid[7,7,0] = 225 # Frag
        grid[7,8,0] = 225 # Frag
        
        
        grid[8,8,0] = 240
        grid[8,9,0] = 250

        grid[9,9,0] = 255 # Soma        
        
        

        # We'll have the bottom-right corner, which is labeled 5, be the soma
        somas = {5:[(9,9,0)]}
    
    if grid_id == 102:
        ''' Generates a 10x10x2 grid with a small assortment of labels'''
        # 10x10x2 image, with 1 color channel greyscale
        grid = np.zeros((10,10,2))
        labels = np.zeros((10,10,2))
        grid[0,9,0] = 255
        grid[0,0,0] = 255
        grid[0,1,0] = 255
        grid[0,2,0] = 255
        grid[3,3,0] = 255
        grid[4,3,0] = 255
        grid[5,3,0] = 255
        grid[5,4,0] = 255
        grid[6,4,0] = 255
        grid[6,5,0] = 255
        grid[7,7,0] = 255
        grid[7,8,0] = 255
        grid[9,9,0] = 255 # Soma
        
        # Add this back in later. For simplicity, only the 0th layer of grid has labels
        #grid[:,:,1] = grid[:,:,0] 
        # Label the components
        labels, num = measure.label(grid, return_num=True)

        # Create intensity data, similar to the other grid10 but it has an
        # intensity anomaly
        grid[0,9,0] = 255 # Blob
        
        grid[0,0,0] = 150 # Frag
        grid[0,1,0] = 150 # Frag
        grid[0,2,0] = 150 # Frag
        
        grid[1,2,0] = 150
        grid[1,3,0] = 150
        grid[2,3,0] = 175
                
        grid[3,3,0] = 255 # Frag
        grid[4,3,0] = 255 # Frag
        grid[5,3,0] = 255 # Frag
        grid[5,4,0] = 255 # Frag
        grid[6,4,0] = 255 # Frag
        grid[6,5,0] = 255 # Frag
        
        grid[6,6,0] = 40 
        grid[7,6,0] = 40 

        grid[7,7,0] = 50 # Frag
        grid[7,8,0] = 50 # Frag
        
        
        grid[8,8,0] = 50
        grid[8,9,0] = 50

        grid[9,9,0] = 255 # Soma        
        


        # We'll have the bottom-right corner, which is labeled 5, be the soma
        somas = {5:[(9,9,0)]}
    
    if grid_id == 20:
        ''' Generates a 20x20x2 grid with a small assortment of labels'''
        # 20x20x2 image, with 1 color channel greyscale
        grid = np.zeros((20,20,2))
        labels = np.zeros((20,20,2))
        # Create some diagonal lines
        for i in range(1,10):
            grid[i,i,0] = 200
        
        # Create some diagonal lines
        for i in range(12,16):
            grid[i,i,0] = 175
        
        # Create a "blob"
        grid[0,16,0] = 225
        grid[0,17,0] = 225
        grid[0,18,0] = 225
        for i in range(1,4):
            grid[i,15,0] = 225
            grid[i,16,0] = 225
            grid[i,17,0] = 225
            grid[i,18,0] = 225
            grid[i,19,0] = 225
        grid[4,16,0] = 225
        grid[4,17,0] = 225
        grid[4,18,0] = 225
        
        # Label a soma
        grid[19,19,0] = 255
        # Add this back in later. For simplicity, only the 0th layer of grid has labels
        #grid[:,:,1] = grid[:,:,0] 
        # Label the components
        labels, num = measure.label(grid, return_num=True)
        
        # We'll have the bottom-right corner, which is labeled 4, be the soma
        somas = {4:[(19,19,0)]}
        
    if grid_id == 100:
        ''' Generates a 100x100x2 grid with 2 labels along the diagonal'''
        ''' and one soma '''
        # 100x100x2 image, with 1 color channel greyscale
        grid = np.zeros((100,100,2))
        labels = np.zeros((100,100,2))
        # Create some diagonal lines
        for i in range(1,49):
            grid[i,i,0] = 200
        
        # Create some diagonal lines
        for i in range(51,98):
            grid[i,i,0] = 175
        # Label a soma
        grid[10,30,0] = 255
        # Add this back in later. For simplicity, only the 0th layer of grid has labels
        #grid[:,:,1] = grid[:,:,0] 
        # Label the components
        labels, num = measure.label(grid, return_num=True)
        somas = {2:[(10,30,0)]}
    
    plt.figure()
    plt.imshow(labels[:,:,0])
    return grid, labels, num, somas

def grid_gen3D(grid_id=10):
    if grid_id == 100:
        grid = np.zeros((100,100,100))
        labels = np.zeros((100,100,100))
        grid[0,0,0] = 255
        grid[1,0,0] = 255
        grid[2,0,0] = 255
        grid[3,0,0] = 255
        
        grid[99,99,99] = 255
        grid[98,99,99] = 255
        grid[97,99,99] = 255
        grid[96,99,99] = 255
        
        # Mark a soma
        grid[0,0,99] = 255
        
        somas = {2:[(0,0,99)]}
        labels, num = measure.label(grid, return_num=True)
    
    if grid_id == 25:
        grid = np.zeros((25,25,25))
        labels = np.zeros((25,25,25))
        
        # Create a line
        for i in range(0,5):
            grid[i,0,0] = 255
        
        # Create a cube for a blob
        for i in range(20,25):
            for j in range(20,25):
                for k in range(20,25):
                    grid[i,i,i] = 255
                    grid[i,i,i] = 255
                    grid[i,i,i] = 255
                    grid[i,i,i] = 255
                    grid[i,i,i] = 255
        
        # Remove the corners to make it more blob-like
        grid[20,20,20] = 0
        grid[20,24,20] = 0
        grid[20,20,24] = 0
        grid[20,24,24] = 0
        grid[24,20,24] = 0
        grid[24,20,20] = 0
        grid[24,20,24] = 0
        grid[24,24,24] = 0 
        grid[24,24,24] = 0
        
        # Mark a soma
        grid[0,0,24] = 255
        
        somas = {2:[(0,0,24)]}
        
        labels, num = measure.label(grid, return_num=True)
    
    if grid_id == 101:
        grid = np.zeros((100,100,100))
        labels = np.zeros((100,100,100))
        for i in range(1,90):
            grid[i,0,0] = 240
        for j in range(5,90):
            grid[0,j,0] = 200
        for k in range(10,70):
            grid[0,0,k] = 160
                    

        grid[99,99,99] = 255
        labels, num = measure.label(grid, return_num=True)
        print("Soma:",labels[99,99,99])
        somas = {3:[(99,99,99)]}
    
    if grid_id == 15:
        grid = np.zeros((100,100,100))
        labels = np.zeros((100,100,100))
        for i in range(0,4):
            grid[i,0,0] = 150

        grid[4,3,0] = 125
        grid[4,3,1] = 125
        grid[4,4,1] = 125
        grid[4,4,2] = 125
        grid[4,5,2] = 125
        grid[4,6,2] = 125
        grid[4,7,2] = 125
        
        grid[6,8,4] = 175
        grid[6,9,4] = 175
        grid[6,10,4] = 175
        grid[7,10,4] = 175
        grid[7,10,5] = 175
        grid[7,10,6] = 175
        grid[7,10,7] = 175
        
        grid[8,12,7] = 225
        grid[8,12,8] = 225
        grid[8,12,9] = 225
        grid[8,12,10] = 225
        grid[9,12,10] = 225
        grid[10,12,10] = 225
        grid[11,13,10] = 225
        grid[12,13,10] = 225
        grid[12,13,11] = 225
        
        #for j in range(0,12):
        #    grid[0,j,12] = 100
        
        grid[14,14,14] = 255
        labels, num = measure.label(grid, return_num=True)
        #print("labels: ", np.unique(labels))
        #print("lbl1:" , np.argwhere(labels==1))
        #print("lbl2:" , np.argwhere(labels==2))
        #print("lbl3:" , np.argwhere(labels==3))
        #print("lbl4:" , np.argwhere(labels==4))
        #print("lbl5:" , np.argwhere(labels==5))
        somas = {5:[(14,14,14)]}

        # Adding intensity data
        grid[3,1,0] = 130
        grid[3,2,0] = 130
        grid[3,3,0] = 130

        grid[4,7,2] = 150
        grid[4,8,2] = 150
        grid[4,8,3] = 150
        grid[5,8,3] = 150
        grid[6,8,3] = 150
        
        grid[7,10,7] = 200
        grid[8,10,7] = 200
        grid[8,11,7] = 200
        
        grid[12,14,11] = 245
        grid[13,14,11] = 245
        grid[13,14,12] = 245
        grid[14,14,13] = 245

    return grid, labels, num, somas
    
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

img2, lbls2, _ , somas2 = grid_gen(20)
alg2 = viterbi_algorithm(img2, lbls2, somas2, [1,1,1])

# I tried doing spectral embedding, the endpoints produced were not correct
#alg2.frags_to_lines_le_skel([1,4])
#print("Endpoints Tests: ", alg2.end_points)

# For the timebeing, manually do the endpoints
# NOTE: no endpoints for 1 or 4 because they are blobs
endpoints2 = {}
endpoints2[2] = ((1,1,0),(9,9,0))
endpoints2[3] = ((12,12,0),(15,15,0))
alg2.end_points = endpoints2

# 100x100 example for endpoints
img3, lbls3, _, somas3 = grid_gen(100)
alg3 = viterbi_algorithm(img3, lbls3, somas3, [1,1,1])

img1_3D, lbls1_3D, _, somas1_3D = grid_gen3D(100)
alg1_3D = viterbi_algorithm(img1_3D, lbls1_3D, somas1_3D, [1,1,1])
endpts1_3D = {}
endpts1_3D[1] = ((0,0,0),(3,0,0))
endpts1_3D[3] = ((99,99,99),(96,99,99))
alg1_3D.end_points = endpts1_3D

img2_3D, lbls2_3D, _, somas2_3D,= grid_gen3D(25)
alg2_3D = viterbi_algorithm(img2_3D, lbls2_3D, somas2_3D, [1,1,1])
endpts2_3D = {}
endpts2_3D[1] = ((0,0,0),(4,0,0))
alg2_3D.end_points = endpts2_3D

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
testLineToLine3D()
testLineToBlob()
testLineToBlob3D()
testPathIntensityCost()
testConnections()
testEndpoints()
testEndpoints3D()
testViterbi()
testViterbi3D()
#print(lbls)