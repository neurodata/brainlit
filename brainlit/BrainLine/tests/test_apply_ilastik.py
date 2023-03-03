import h5py
import numpy as np
from brainlit.BrainLine.apply_ilastik import plot_results
import os
import pytest
import matplotlib.pyplot as plt

def test_ApplyIlastik():
    # need to have ilastik downlaoded
    pass


@pytest.fixture(scope="session")
def axon_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    brain_dir = data_dir / "braintest"
    brain_dir.mkdir()
    val_dir = brain_dir / "val"
    val_dir.mkdir()

    labels = np.ones((2,10,10,10), dtype = np.uint16)
    labels[0,0,0,:] = 2
    labels_path = val_dir / "subvol-image_3channel_Labels.h5"
    with h5py.File(str(labels_path), "w") as f:
        dset = f.create_dataset("exported_data", data=labels)

    im_probs = np.zeros((2,10,10,10))
    im_probs[0,0,0,:5] = 0.9
    for fname in ["subvol.h5", "subvol_Probabilities.h5"]:
        path = val_dir / fname
        with h5py.File(path, "w") as f:
            f.create_dataset("exported_data", data = im_probs)

    return data_dir


def test_plot_results_axon(axon_data_dir):
    data_dir_str = str(axon_data_dir)
    test_max_fscore, test_best_threshold = plot_results(data_dir=data_dir_str, brain_id="test", object_type = "axon", positive_channel=0, show_plot=False)
    plt.close('all')

    true_prec = 1
    true_rec = 0.5
    true_max_fscore = 2 * true_prec * true_rec / (true_prec + true_rec)

    assert test_best_threshold < 0.9
    assert true_max_fscore == test_max_fscore
    

@pytest.fixture(scope="session")
def soma_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    brain_dir = data_dir / "braintest"
    brain_dir.mkdir()
    val_dir = brain_dir / "val"
    val_dir.mkdir()

    im_probs = np.zeros((2,10,10,10))
    im_probs[1,:,:,:7] = 0.9
    for fname in ["subvol1_pos", "subvol2_pos", "subvol3_neg"]:
        fname_im = fname + ".h5"
        im_path = val_dir / fname_im
        with h5py.File(str(im_path), "w") as f:
            pass

        fname_prob = fname + "_Probabilities.h5"
        path = val_dir / fname_prob
        with h5py.File(path, "w") as f:
            f.create_dataset("exported_data", data = im_probs)

    return data_dir


def test_plot_results_soma(soma_data_dir):
    data_dir_str = str(soma_data_dir)
    test_max_fscore, test_best_threshold = plot_results(data_dir=data_dir_str, brain_id="test", object_type = "soma", positive_channel=1, show_plot=False, doubles=["subvol2_pos.h5"])
    plt.close('all')

    true_prec = 2/3
    true_rec = 1
    true_max_fscore = 2 * true_prec * true_rec / (true_prec + true_rec)

    assert test_best_threshold < 0.9
    assert true_max_fscore == test_max_fscore