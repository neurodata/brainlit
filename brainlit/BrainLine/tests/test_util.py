import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from brainlit.BrainLine import util
from cloudvolume import CloudVolume
from pathlib import Path
import os
import json
from skimage import io


@pytest.fixture(scope="session")
def make_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    return data_dir


# Makes a data file with invalid object_type, and a sample where info files can be written
@pytest.fixture(scope="session")
def make_bad_datafile(make_data_dir):
    data_dir = make_data_dir
    bad_data_file = data_dir / "bad_data.json"
    base_s3 = f"precomputed://file://{str(data_dir)}"
    bad_type = {
        "object_type": "invalid",
        "brain2paths": {"write_info": {"base_s3": base_s3}},
    }
    with open(bad_data_file, "w") as f:
        json.dump(bad_type, f)

    return bad_data_file


@pytest.fixture(scope="session")
def ontology_path():
    ontology_json_path = (
        Path(os.path.abspath(__file__)).parents[1]
        / "data"
        / "ara_structure_ontology.json"
    )
    return ontology_json_path


############################
### functionality checks ###
############################


def test_get_corners():
    corners = util._get_corners([100, 100, 100], [50, 50, 50], max_coords=[50, 50, 50])
    assert corners == [[[0, 0, 0], [50, 50, 50]]]
    corners = util._get_corners([100, 100, 100], [50, 50, 50], min_coords=[50, 50, 50])
    assert corners == [[[50, 50, 50], [100, 100, 100]]]


def test_find_atlas_level_label(ontology_path):
    region_graph = util._setup_atlas_graph(ontology_json_path=ontology_path)
    new_label = util._find_atlas_level_label(
        label=567,
        atlas_level_nodes=[1009, 73, 1024, 304325711],
        atlas_level=1,
        G=region_graph,
    )
    assert new_label == 8


def test_download_subvolumes(make_data_dir, make_bad_datafile):
    data_dir = make_data_dir  # tmp_path_factory.mktemp("data")
    layer_names = ["average_10um", "average_10um", "zero"]

    # Data file with bad object_fype
    bad_data_file = make_bad_datafile
    with pytest.raises(ValueError) as e_info:
        util.download_subvolumes(
            data_dir=data_dir,
            brain_id="pytest",
            layer_names=layer_names,
            dataset_to_save="val",
            data_file=bad_data_file,
        )
    assert e_info.value.args[0] == f"object_type must be soma or axon, not invalid"

    # Sample with no brain_s3 path
    data_file = (
        Path(os.path.abspath(__file__)).parents[3]
        / "docs"
        / "notebooks"
        / "pipelines"
        / "BrainLine"
        / "axon_data.json"
    )
    with pytest.raises(ValueError) as e_info:
        util.download_subvolumes(
            data_dir=data_dir,
            brain_id="pytest_nobases3",
            layer_names=layer_names,
            dataset_to_save="val",
            data_file=data_file,
        )
    assert (
        e_info.value.args[0]
        == f"base_s3 not an entry in brain2paths for brain pytest_nobases3"
    )

    # Axon
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

    # Soma
    # data_dir is string and data folder has already been made

    output_dir = data_dir / "brainpytest_download"
    output_dir.mkdir()
    output_dir = output_dir / "val"
    output_dir.mkdir()

    data_file = (
        Path(os.path.abspath(__file__)).parents[3]
        / "docs"
        / "notebooks"
        / "pipelines"
        / "BrainLine"
        / "soma_data.json"
    )
    util.download_subvolumes(
        data_dir=str(data_dir),
        brain_id="pytest_download",
        layer_names=layer_names,
        dataset_to_save="val",
        data_file=data_file,
    )
    files = os.listdir(output_dir)
    assert len(files) == 2


def test_json_to_points(make_data_dir):
    data_dir = make_data_dir
    json_data = {
        "layers": [
            {
                "type": "annotation",
                "name": "points",
                "annotations": [{"point": [0, 1, 2]}],
            }
        ]
    }
    json_path = data_dir / "json_file.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f)
    point_layers = util.json_to_points(str(json_path))

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


def test_setup_atlas_graph(ontology_path):
    G = util._setup_atlas_graph(ontology_json_path=ontology_path)

    assert G.nodes[997]["name"] == "root"
    assert G.nodes[997]["st_level"] == 0
    assert len(list(G.predecessors(997))) == 0
    assert len(list(G.successors(997))) > 0

    assert G.nodes[872]["name"] == "Dorsal nucleus raphe"
    assert G.nodes[872]["st_level"] == 8
    assert G.nodes[872]["acronym"] == "DR"
    assert len(list(G.predecessors(872))) > 0
    assert len(list(G.successors(872))) == 0


def test_get_atlas_level_nodes(ontology_path):
    G = util._setup_atlas_graph(ontology_json_path=ontology_path)
    atlas_level_nodes_test = util._get_atlas_level_nodes(0, G)

    assert len(atlas_level_nodes_test) == 1
    assert atlas_level_nodes_test[0] == 997

    atlas_level_nodes_test = util._get_atlas_level_nodes(1, G)

    assert len(atlas_level_nodes_test) == 5
    for idx, region in enumerate([8, 1009, 73, 1024, 304325711]):
        assert atlas_level_nodes_test[idx] == region


def test_find_atlas_level_label(ontology_path):
    atlas_level = 1

    G = util._setup_atlas_graph(ontology_json_path=ontology_path)
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


def test_create_transformed_mask_info(make_bad_datafile):
    bad_data_file = make_bad_datafile
    util.create_transformed_mask_info(brain="write_info", data_file=bad_data_file)


def test_dir_to_atlas_pts(tmp_path):
    json_dir = tmp_path / "json_data"
    json_dir.mkdir()

    json_data = [{"point": [10, 0, 0]}, {"point": [1, 1, 1]}, {"point": [2, 1, 1]}]
    with open(json_dir / "json1.json", "w") as f:
        json.dump(json_data, f)

    json_data = [{"point": [0, 0, 0]}, {"point": [1, 2, 1]}]
    with open(json_dir / "json2.json", "w") as f:
        json.dump(json_data, f)

    atlas_file = tmp_path / "atlas.tif"
    atlas_im = np.zeros((3, 3, 3), dtype="uint16")
    atlas_im[1:, 1:, 1:] = 1
    io.imsave(atlas_file, atlas_im)

    outname = tmp_path / "filtered.txt"
    util.dir_to_atlas_pts(dir=json_dir, outname=outname, atlas_file=atlas_file)

    with open(outname, "r") as f:
        for count, _ in enumerate(f):
            pass

    assert count + 1 == 3
