import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import napari


def napari_viewer(
    img, labels=None, shapes=None, points=None, label_name="Segmentation"
):
    with napari.gui_qt():
        viewer = napari.view_image(np.squeeze(np.array(img)))
        if labels is not None:
            if isinstance(labels, list):
                for l, label in enumerate(labels):
                    name = label_name + "_" + str(l)
                    viewer.add_labels(label, name=name)
            else:
                viewer.add_labels(labels, name=label_name)
        if shapes is not None:
            if isinstance(labels, list):
                for l, shape in enumerate(shapes):
                    name = label_name + "_" + str(l)
                    viewer.add_shapes(data=shape, shape_type="path", edge_color="blue")
            else:
                viewer.add_shapes(
                    data=shapes, shape_type="path", edge_color="blue", name="Skeleton"
                )
        if points is not None:
            viewer.add_points(points, size=2, face_color="red")
        return viewer
