import h5py
import numpy as np
from brainlit.BrainLine.apply_ilastik import (
    plot_results,
    examine_threshold,
    ApplyIlastik,
    ApplyIlastik_LargeImage
)
import os
import pytest
import matplotlib.pyplot as plt
from pathlib import Path


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

    labels = np.ones((2, 10, 10, 10), dtype=np.uint16)
    labels[0, 0, 0, :] = 2
    labels_path = val_dir / "subvol-image_3channel_Labels.h5"
    with h5py.File(str(labels_path), "w") as f:
        dset = f.create_dataset("exported_data", data=labels)

    im_probs = np.zeros((2, 10, 10, 10))
    im_probs[0, 0, 0, :5] = 0.9
    path = val_dir / "subvol_Probabilities.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("exported_data", data=im_probs)

    im = np.zeros((3, 10, 10, 10))
    im[0, 0, 0, :5] = 0.9
    path = val_dir / "subvol.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("image_3channel", data=im)

    return data_dir


def test_move_results(axon_data_dir):
    data_dir_str = str(axon_data_dir)
    apl = ApplyIlastik(
        ilastk_path="test",
        project_path="test",
        brains_path=data_dir_str,
        brains=["test"],
    )
    apl.move_results()


def test_plot_results_axon(axon_data_dir):
    data_dir_str = str(axon_data_dir)
    test_max_fscore, test_best_threshold = plot_results(
        data_dir=data_dir_str,
        brain_ids=["test"],
        object_type="axon",
        positive_channel=0,
        show_plot=False,
    )

    true_prec = 1
    true_rec = 0.5
    true_max_fscore = 2 * true_prec * true_rec / (true_prec + true_rec)

    assert test_best_threshold < 0.9
    assert true_max_fscore == test_max_fscore


def test_examine_threshold_axon(axon_data_dir):
    data_dir_str = str(axon_data_dir)
    examine_threshold(
        data_dir=data_dir_str,
        brain_id="test",
        threshold=0.5,
        object_type="axon",
        positive_channel=0,
        show_plot=False,
    )


@pytest.fixture(scope="session")
def soma_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    brain_dir = data_dir / "braintest"
    brain_dir.mkdir()
    val_dir = brain_dir / "val"
    val_dir.mkdir()

    im_probs = np.zeros((2, 10, 10, 10))
    im = np.zeros((3, 10, 10, 10))
    im_probs[1, :, :, :7] = 0.9
    for fname in ["subvol1_pos", "subvol2_pos", "subvol3_neg"]:
        fname_im = fname + ".h5"
        im_path = val_dir / fname_im
        with h5py.File(str(im_path), "w") as f:
            f.create_dataset("image_3channel", data=im)

        fname_prob = fname + "_Probabilities.h5"
        path = val_dir / fname_prob
        with h5py.File(path, "w") as f:
            f.create_dataset("exported_data", data=im_probs)

    return data_dir


def test_plot_results_soma(soma_data_dir):
    data_dir_str = str(soma_data_dir)
    test_max_fscore, test_best_threshold = plot_results(
        data_dir=data_dir_str,
        brain_ids=["test"],
        object_type="soma",
        positive_channel=1,
        show_plot=False,
        doubles=["subvol2_pos.h5"],
    )
    plt.close("all")

    true_prec = 2 / 3
    true_rec = 1
    true_max_fscore = 2 * true_prec * true_rec / (true_prec + true_rec)

    assert test_best_threshold < 0.9
    assert true_max_fscore == test_max_fscore


def test_examine_threshold_soma(soma_data_dir):
    data_dir_str = str(soma_data_dir)
    examine_threshold(
        data_dir=data_dir_str,
        brain_id="test",
        threshold=0.5,
        object_type="soma",
        positive_channel=1,
        doubles=["subvol2_pos.h5"],
        show_plot=False,
    )



def test_ApplyIlastik_LargeImage():
    data_file = (
        Path(os.path.abspath(__file__)).parents[3]
        / "docs"
        / "notebooks"
        / "pipelines"
        / "BrainLine"
        / "axon_data.json"
    )
    aili = ApplyIlastik_LargeImage(ilastik_path="", ilastik_project="", ncpu=1, data_file=data_file)
    # Sample data is there but file path in data json is specific to thomastathey
