import numpy as np
from brainlit.napari_viterbrain import viterbrain_reader
from brainlit.algorithms.connect_fragments.tests.test_viterbrain import create_vb
import pickle

vb = create_vb()


def test_reader(tmp_path):
    my_test_file = str(tmp_path / "viterbrain.pickle")
    with open(my_test_file, "rb") as handle:
        pickle.dump(vb, handle)

    # try to read it back in
    reader = viterbrain_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    reader = viterbrain_reader("fake.file")
    assert reader is None
