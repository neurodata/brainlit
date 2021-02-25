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

''' Setting up the image environment '''

def grid_gen():
    # 10x10x2 image, with 1 color channel greyscale
    # The 3rd dimension will simply be a copy of the 1st dimension
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
    somas = {5:(9,9,0)}
    return grid, labels, num, somas

img, lbls, _ , somas = grid_gen()
alg = viterbi_algorithm(img, lbls, somas, [1,1])
# For the timebeing, manually do the endpoints
# NOTE: no endpoints for 2 or 5 because they are "blobs"
endpoints = {}
endpoints[1] = ((0,0,0),(0,2,0))
endpoints[3] = ((3,3,0),(6,5,0))
endpoints[4] = ((7,7,0),(7,8,0))
alg.end_points = endpoints
    

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
    
def testLineToSoma():
    # Compare the output of line_line_dist with manually computed distances
    assert(np.sqrt((8-9)**2+(7-9)**2) == alg.line_blob_dist(4,5))

testInit()
testLineToLine()
testLineToSoma()