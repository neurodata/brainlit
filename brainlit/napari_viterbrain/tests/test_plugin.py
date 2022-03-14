import numpy as np
from brainlit.napari_viterbrain.viterbrain_plugin import napari_get_reader, comp_trace
from brainlit.algorithms.connect_fragments.tests.test_viterbrain import create_vb
import pickle

vb = create_vb()


def test_reader(tmp_path):
    my_test_file = str(tmp_path / "viterbrain.pickle")
    with open(my_test_file, "wb") as handle:
        pickle.dump(vb, handle)

    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    reader = napari_get_reader("fake.file")
    assert reader is None


def test_example_magic_widget(make_napari_viewer, tmp_path):
    my_test_file = str(tmp_path / "viterbrain.pickle")
    with open(my_test_file, "wb") as handle:
        pickle.dump(vb, handle)

    viewer = make_napari_viewer()

    start_layernum = len(list(viewer.layers))

    comp_trace(v=viewer, start_comp=1, end_comp=5, filename=my_test_file)

    # test that a layer was added
    assert len(list(viewer.layers)) == start_layernum + 1
