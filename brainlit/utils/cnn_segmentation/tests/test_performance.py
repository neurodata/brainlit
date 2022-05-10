
import pytest

import numpy as np
import torch

from brainlit.utils.cnn_segmentation import performance
from numpy.testing import (
    assert_array_equal,
)

############################
### functionality checks ###
############################

def test_get_metrics():
    pred_list = [torch.from_numpy(np.zeros(shape=(4, 4, 4))), torch.from_numpy(np.ones(shape=(4, 4, 4)))]
    y_list = [torch.from_numpy(np.ones(shape=(4, 4, 4))), torch.from_numpy(np.ones(shape=(4, 4, 4)))]

    acc_list, precision_list, recall_list, percent_nonzero = performance.get_metrics(pred_list, y_list)
    
    acc_true = [0.0, 100.0]
    precision_true = [0.0, 100.0]
    recall_true = [0.0, 100.0]
    percent_nonzero_true = [0.0, 100.0]
    
    assert_array_equal(acc_list, acc_true)
    assert_array_equal(precision_list, precision_true)
    assert_array_equal(recall_list, recall_true)
    assert_array_equal(percent_nonzero, percent_nonzero_true)
    


    
