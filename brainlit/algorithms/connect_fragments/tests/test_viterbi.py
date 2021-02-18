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

# 10x10 image, with 1 color channel greyscale
grid = np.zeros((10,10,1))
grid[0,9] = 255
grid[0,0] = 255
grid[0,1] = 255
grid[0,2] = 255
grid[3,3] = 255
grid[4,3] = 255
grid[5,3] = 255
grid[5,4] = 255
grid[6,4] = 255
grid[6,5] = 255
grid[7,7] = 255
grid[7,8] = 255
grid[9,9] = 255

# Label the components
labels, num = measure.label(grid, return_num=True)
plt.imshow(labels[:,:,0])

# We'll have the bottom-right corner, which is labeled 5, be the soma
somas = [5]
