import zarr
import numpy as np
import h5py
from joblib import Parallel, delayed
import os
from skimage import measure
from brainlit.preprocessing import image_process
from tqdm import tqdm

import subprocess


class state_generation:
    def __init__(
        self,
        image_path,
        ilastik_program_path,
        ilastik_project_path,
        chunk_size=[375, 375, 125],
        soma_coords=[],
        resolution=[],
        parallel=1,
    ):
        self.image_path = image_path
        image = zarr.open(image_path, mode="r")
        self.image_shape = image.shape
        self.ilastik_program_path = ilastik_program_path
        self.ilastik_project_path = ilastik_project_path
        self.chunk_size = chunk_size
        self.soma_coords = soma_coords
        self.resolution = resolution
        self.parallel = parallel

        self.prob_path = None

    def predict_thread(self, corner1, corner2, data_bin):
        print(f"{corner1}, {corner2}, {data_bin}")
        image = zarr.open(self.image_path, mode="r")
        image_chunk = np.squeeze(
            image[
                corner1[0] : corner2[0],
                corner1[1] : corner2[1],
                corner1[2] : corner2[2],
            ]
        )
        fname = data_bin + "image_" + str(corner1[2]) + ".h5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("image_chunk", data=image_chunk)
        subprocess.run(
            [
                self.ilastik_program_path,
                "--headless",
                f"--project={self.ilastik_project_path}",
                fname,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def predict(self, data_bin):
        image = zarr.open(self.image_path, mode="r")
        probabilities = zarr.zeros(
            np.squeeze(image.shape), chunks=image.chunks, dtype="float"
        )
        chunk_size = self.chunk_size
        items = self.image_path.split(".")
        prob_fname = items[0] + "_probs.zarr"

        print(f"Constructing probability  image {prob_fname} of shape {probabilities.shape}")

        for x in tqdm(np.arange(0, image.shape[0], chunk_size[0]), desc="Computing Ilastik Predictions"):
            x2 = np.amin([x + chunk_size[0], image.shape[0]])
            for y in tqdm(np.arange(0, image.shape[1], chunk_size[1]), leave=False):
                y2 = np.amin([y + chunk_size[1], image.shape[1]])
                Parallel(n_jobs=self.parallel)(
                    delayed(self.predict_thread)(
                        [x, y, z],
                        [x2, y2, np.amin([z + chunk_size[2], image.shape[2]])],
                        data_bin,
                    )
                    for z in np.arange(0, image.shape[2], chunk_size[2])
                )

                for f in os.listdir(data_bin):
                    fname = os.path.join(data_bin, f)
                    if "Probabilities" in f:
                        items = f.split("_")
                        z = int(items[1])
                        z2 = np.amin([z + chunk_size[2], image.shape[2]])
                        f = h5py.File(fname, "r")
                        pred = f.get("exported_data")
                        pred = pred[:, :, :, 0]

                        probabilities[x:x2, y:y2, z:z2] = pred
                    os.remove(fname)

        zarr.save(prob_fname, probabilities)
        self.prob_path = fname

    def _get_frag_specifications(self):
        image = zarr.open(self.image_path, mode="r")
        chunk_size = self.chunk_size
        soma_coords = self.soma_coords

        specifications = []

        for x in np.arange(0, image.shape[0], chunk_size[0]):
            x2 = np.amin([x + chunk_size[0], image.shape[0]])
            for y in np.arange(0, image.shape[1], chunk_size[1]):
                y2 = np.amin([y + chunk_size[1], image.shape[1]])
                for z in np.arange(0, image.shape[2], chunk_size[2]):
                    z2 = np.amin([z + chunk_size[2], image.shape[2]])
                    soma_coords_new = []
                    for soma_coord in soma_coords:
                        if (
                            np.less_equal([x, y, z], soma_coord).all()
                            and np.less_equal(
                                soma_coord,
                                [x2, y2, z2],
                            ).all()
                        ):
                            soma_coords_new.append(np.subtract(soma_coord, [x, y, z]))

                    specifications.append(
                        {
                            "corner1": [x, y, z],
                            "corner2": [x2, y2, z2],
                            "soma_coords": soma_coords_new,
                        }
                    )

        return specifications

    def split_frags_thread(self, corner1, corner2, soma_coords=[]):
        threshold = 0.9

        prob = zarr.open(self.prob_path, mode="r")

        im_processed = prob[
            corner1[0] : corner2[0], corner1[1] : corner2[1], corner1[2] : corner2[2]
        ]
        labels = measure.label(im_processed > threshold)

        radius_states = 7

        (
            image_iterative,
            states,
            comp_to_states,
            new_soma_masks,
        ) = image_process.remove_somas(
            soma_coords, labels, im_processed, res=self.resolution
        )
        mask = labels > 0
        mask2 = image_process.removeSmallCCs(mask, 25)
        image_iterative[mask & (~mask2)] = 0

        states, comp_to_states = image_process.split_frags_place_points(
            image_iterative,
            labels,
            radius_states,
            self.resolution,
            threshold,
            states,
            comp_to_states,
        )

        new_labels = image_process.split_frags_split_comps(
            labels, new_soma_masks, states, comp_to_states
        )

        new_labels = image_process.split_frags_split_fractured_components(new_labels)

        props = measure.regionprops(new_labels)
        for label, prop in enumerate(tqdm(props, desc="remove small fragments")):
            if prop.area < 15:
                new_labels[new_labels == prop.label] = 0

        new_labels = image_process.rename_states_consecutively(new_labels)

        return (corner1, corner2, new_labels)

    def compute_frags(self):
        image = zarr.open(self.image_path, mode="r")
        fragments = zarr.zeros(
            np.squeeze(image.shape), chunks=image.chunks, dtype="uint16"
        )
        items = self.image_path.split(".")
        frag_fname = items[0] + "_labels.zarr"

        print(f"Constructing fragment image {frag_fname} of shape {fragments.shape}")


        specifications = self._get_frag_specifications()

        results = Parallel(n_jobs=self.parallel)(
            delayed(self.split_frags_thread)(
                specification["corner1"],
                specification["corner2"],
                specification["soma_coords"],
            )
            for specification in specifications
        )

        max_label = 0
        for result in results:
            corner1, corner2, labels = result
            labels[labels >= 0] += max_label
            max_label = np.amax([max_label, np.amax(labels)])
            fragments[corner1[0]:corner2[0], corner1[1]:corner2[1], corner1[2]:corner2[2]] = labels


        
        zarr.save(frag_fname, fragments)
        self.fragment_name = frag_fname


