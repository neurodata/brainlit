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
    
def testLineToBlob():
    # Compare the output of line_line_dist with manually computed distances
    assert(np.sqrt((8-9)**2+(7-9)**2) == alg.line_blob_dist(4,5)[0])

    # Should go from an endpoint to the closest position on the blob
    assert(np.sqrt((9-3)**2+(9-15)**2) == alg2.line_blob_dist(2,1)[0])
    with pytest.raises(ValueError):
        alg.line_blob_dist(2,2)

def testPathIntensityCost():
    # Currently a stub, need to double-check what the proper form of the 
    # intensity score should be
    assert(0 == alg2.line_int((9,9,0),(12,12,0),2,3))
    
def testConnections():
    # Test if we are getting the correct connections
    alg.compute_all_dists(alg.somas)
    c = alg.connection_mat
    print(alg.connection_mat[0][3][1])
    print(alg.connection_mat[1][3][1])
    
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
    # NOTE: All of the following 5 to x fail currently.
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
    
    # For manual debugging purposes
    connect_points = set()
    for i in range(1):
        for j in range(1,5):
            for k in range(1,5):
                if not (np.equal([0,0,0], c[i][j][k])).all():
                    connect_points.add(tuple(c[i][j][k]))


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
        grid[9,9,0] = 255
        
        # Add this back in later. For simplicity, only the 0th layer of grid has labels
        #grid[:,:,1] = grid[:,:,0] 
        # Label the components
        labels, num = measure.label(grid, return_num=True)
        plt.imshow(labels[:,:,0])
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
        plt.imshow(labels[:,:,0])
        # We'll have the bottom-right corner, which is labeled 4, be the soma
        somas = {4:[(19,19,0)]}
    
    return grid, labels, num, somas

img, lbls, _ , somas = grid_gen(10)
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
alg2.frags_to_lines_le_skel([1,4])
print(alg2.end_points)

# For the timebeing, manually do the endpoints
# NOTE: no endpoints for 1 or 4 because they are blobs
endpoints2 = {}
endpoints2[2] = ((1,1,0),(9,9,0))
endpoints2[3] = ((12,12,0),(15,15,0))
alg2.end_points = endpoints2

testInit()
testLineToLine()
testLineToBlob()
testPathIntensityCost()
testConnections()