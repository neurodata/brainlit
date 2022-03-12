import pickle
import zarr
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory

def viterbrain_reader(path):
    
    with open(path, "rb") as handle:
        viterbi = pickle.load(handle)

    layer_labels = zarr.open(viterbi.fragment_path)
    image_path = viterbi.fragment_path[:-12] + ".zarr"
    layer_image = zarr.open(image_path)

    scale = viterbi.resolution

    meta_labels = {'name': 'fragments', 'scale': scale}
    meta_image = {'name': 'image', 'scale': scale}

    return [(layer_image,meta_image,'image'),(layer_labels,meta_labels,'labels')]

def napari_get_reader(path):
    parts = path.split('.')
    if parts[-1] == "pickle" or parts[-1] == "pkl":
        return viterbrain_reader
    else:
        return None

class NapariViterBrain(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer


        btn_start = QPushButton("Select Start")
        btn_start.clicked.connect(self._click_start)

        btn_end = QPushButton("Select End")
        btn_end.clicked.connect(self._click_end)

        btn_trace = QPushButton("Trace")
        btn_trace.clicked.connect(self._click_trace)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn_start)
        self.layout().addWidget(btn_end)
        self.layout().addWidget(btn_trace)

    def _click_start(self,):
        print("start")

    def _click_end(self,):
        print("end")

    def _click_trace(self,):
        print("trace")

@magic_factory
def ViterBrain(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return ViterBrain