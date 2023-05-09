import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from brainlit.BrainLine import util
from cloudvolume import CloudVolume
from pathlib import Path
import os


@pytest.fixture(scope="session")
def make_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    return data_dir


############################
### functionality checks ###
############################


def test_download_subvolumes(make_data_dir):
    data_dir = make_data_dir  # tmp_path_factory.mktemp("data")
    layer_names = ["average_10um"] * 3
    data_file = (
        Path(os.path.abspath(__file__)).parents[3]
        / "docs"
        / "notebooks"
        / "pipelines"
        / "BrainLine"
        / "axon_data.json"
    )
    util.download_subvolumes(
        data_dir=data_dir,
        brain_id="pytest",
        layer_names=layer_names,
        dataset_to_save="val",
        data_file=data_file,
    )
    output_dir = data_dir / "brainpytest" / "val"
    files = os.listdir(output_dir)
    assert len(files) == 2


def test_json_to_points():
    url = "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=ki9d3Hsk5jcsJg"
    point_layers = util.json_to_points(url)
    keys = point_layers.keys()
    assert len(keys) == 2
    assert "points_1" in keys
    assert "points_2" in keys
    assert_array_almost_equal(point_layers["points_1"][0], [370, 235, 512], decimal=0)

    point_layers = util.json_to_points(url, round=True)
    keys = point_layers.keys()
    assert len(keys) == 2
    assert "points_1" in keys
    assert "points_2" in keys
    assert_array_equal(point_layers["points_1"][0], [371, 235, 512])


def test_find_sample_names(tmp_path):
    fname = "val_#_#_#"
    fnames = [
        f"{fname}.h5",
        f"{fname}_Labels.h5",
        f"{fname}_Probabilities.h5",
        fname,
        "train_#_#_#.h5",
        "train_#_#",
    ]

    # create files
    for fname in fnames:
        with open(tmp_path / fname, "w") as fp:
            pass

    test_fnames = util._find_sample_names(tmp_path)
    assert len(test_fnames) == 2

    test_fnames = util._find_sample_names(tmp_path, dset="train")
    assert len(test_fnames) == 1
    assert test_fnames[0] == "train_#_#_#.h5"

    test_fnames = util._find_sample_names(tmp_path, add_dir=True)
    assert len(test_fnames) == 2
    true_fname = str(tmp_path) + "/val_#_#_#.h5"
    assert test_fnames[0] == true_fname or test_fnames[1] == true_fname


def test_setup_atlas_graph():
    G = util._setup_atlas_graph()

    assert G.nodes[997]["name"] == "root"
    assert G.nodes[997]["st_level"] == 0
    assert len(list(G.predecessors(997))) == 0
    assert len(list(G.successors(997))) > 0

    assert G.nodes[872]["name"] == "Dorsal nucleus raphe"
    assert G.nodes[872]["st_level"] == 8
    assert G.nodes[872]["acronym"] == "DR"
    assert len(list(G.predecessors(872))) > 0
    assert len(list(G.successors(872))) == 0


def test_get_atlas_level_nodes():
    G = util._setup_atlas_graph()
    atlas_level_nodes_test = util._get_atlas_level_nodes(0, G)

    assert len(atlas_level_nodes_test) == 1
    assert atlas_level_nodes_test[0] == 997

    atlas_level_nodes_test = util._get_atlas_level_nodes(1, G)

    assert len(atlas_level_nodes_test) == 5
    for idx, region in enumerate([8, 1009, 73, 1024, 304325711]):
        assert atlas_level_nodes_test[idx] == region


def test_find_atlas_level_label():
    atlas_level = 1

    G = util._setup_atlas_graph()
    atlas_level_nodes = util._get_atlas_level_nodes(atlas_level, G)

    atlas_level_label_test = util._find_atlas_level_label(
        997, atlas_level_nodes, atlas_level, G
    )
    assert atlas_level_label_test == 997

    atlas_level_label_test = util._find_atlas_level_label(
        872, atlas_level_nodes, atlas_level, G
    )
    assert atlas_level_label_test == 8


def test_fold():
    img = np.eye(6)
    true_fold = np.zeros((6, 3))
    true_fold[:3, :3] = np.eye(3)
    true_fold[3, 2] = 1
    true_fold[4, 1] = 1
    true_fold[5, 0] = 1

    test_fold = util._fold(img)

    assert_array_equal(true_fold, test_fold)
