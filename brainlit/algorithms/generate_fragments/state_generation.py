import zarr
import numpy as np
import h5py
from joblib import Parallel, delayed
import os
from skimage import measure
from brainlit.preprocessing import image_process
from tqdm import tqdm
from skimage import morphology
from sklearn.neighbors import radius_neighbors_graph, KernelDensity
from brainlit.viz.swc2voxel import Bresenham3D
import math
import warnings
import subprocess
import random
import pickle


class state_generation:
    def __init__(
        self,
        image_path,
        ilastik_program_path,
        ilastik_project_path,
        chunk_size=[375, 375, 125],
        soma_coords=[],
        resolution=[0.3, 0.3, 1],
        parallel=1,
        prob_path=None,
        fragment_path=None,
        tiered_path=None,
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

        for other_im, name in zip(
            [prob_path, fragment_path, tiered_path], ["prob", "frag", "tiered"]
        ):
            if other_im is not None:
                other_image = zarr.open(other_im, mode="r")
                if other_image.shape != self.image_shape:
                    raise ValueError(f"{name} image has different shape than image")

        self.prob_path = prob_path
        self.fragment_path = fragment_path
        self.tiered_path = tiered_path

    def predict_thread(self, corner1, corner2, data_bin):
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

        print(
            f"Constructing probability  image {prob_fname} of shape {probabilities.shape}"
        )

        for x in tqdm(
            np.arange(0, image.shape[0], chunk_size[0]),
            desc="Computing Ilastik Predictions",
        ):
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
                        pred = pred[:, :, :, 1]

                        probabilities[x:x2, y:y2, z:z2] = pred
                    os.remove(fname)

        zarr.save(prob_fname, probabilities)
        self.prob_path = prob_fname

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
        print(f"Processing @corner: {corner1}")
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
            image_iterative=image_iterative,
            labels=labels,
            radius_states=radius_states,
            res=self.resolution,
            threshold=threshold,
            states=states,
            comp_to_states=comp_to_states,
        )

        new_labels = image_process.split_frags_split_comps(
            labels, new_soma_masks, states, comp_to_states
        )

        new_labels = image_process.split_frags_split_fractured_components(new_labels)

        props = measure.regionprops(new_labels)
        for _, prop in enumerate(tqdm(props, desc="remove small fragments")):
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
            fragments[
                corner1[0] : corner2[0],
                corner1[1] : corner2[1],
                corner1[2] : corner2[2],
            ] = labels
        print(f"*****************Number of components: {max_label}*******************")
        zarr.save(frag_fname, fragments)

        self.fragment_path = frag_fname

    def compute_soma_lbls(self):
        fragments = zarr.open(self.fragment_path, mode="r")

        soma_lbls = []
        for soma_coord in self.soma_coords:
            local_labels = fragments[
                soma_coord[0] - 20 : soma_coord[0] - 20,
                soma_coord[1] - 20 : soma_coord[1] - 20,
                soma_coord[2] - 20 : soma_coord[2] - 20,
            ]
            soma_label = image_process.label_points(
                local_labels, [soma_coord], res=self.resolution
            )[0]
            soma_lbls.append(soma_label)

        self.soma_lbls = soma_lbls




    def compute_image_tiered_thread(self, corner1, corner2):
        kde = self.kde
        image = zarr.open(self.image_path, mode="r")

        image = image[
            corner1[0] : corner2[0], corner1[1] : corner2[1], corner1[2] : corner2[2]
        ]

        vals = np.array([np.unique(image)]).T
        scores_neg = -1 * kde.score_samples(vals)
        vals = np.squeeze(vals)

        data = np.reshape(np.copy(image), (image.size,))
        sort_idx = np.argsort(vals)
        idx = np.searchsorted(vals, data, sorter=sort_idx)
        out = scores_neg[sort_idx][idx]
        image_tiered = np.reshape(out, image.shape)

        return (corner1, corner2, image_tiered)

    def compute_image_tiered(self):
        image = zarr.open(self.image_path, mode="r")
        fragments = zarr.open(self.fragment_path, mode="r")
        tiered = zarr.zeros(
            np.squeeze(image.shape), chunks=image.chunks, dtype="uint16"
        )
        items = self.image_path.split(".")
        tiered_fname = items[0] + "_tiered.zarr"
        print(f"Constructing tiered image {tiered_fname} of shape {tiered.shape}")

        image_chunk = image[:300, :300, :300]
        fragments_chunk = fragments[:300, :300, :300]
        data_fg = image_chunk[fragments_chunk > 0]
        if len(data_fg.flatten()) > 10000:
            data_sample = random.sample(list(data_fg), k=10000)
        else:
            data_sample = data_fg
        data_2d = np.expand_dims(np.sort(np.array(data_sample)), axis=1)
        kde = KernelDensity(kernel="gaussian", bandwidth=100).fit(data_2d)
        self.kde = kde

        specifications = self._get_frag_specifications()

        results = Parallel(n_jobs=self.parallel)(
            delayed(self.compute_image_tiered_thread)(
                specification["corner1"],
                specification["corner2"],
            )
            for specification in specifications
        )

        for result in results:
            corner1, corner2, image_tiered = result
            tiered[
                corner1[0] : corner2[0],
                corner1[1] : corner2[1],
                corner1[2] : corner2[2],
            ] = image_tiered

        zarr.save(tiered_fname, tiered)
        self.tiered_path = tiered_fname

    def compute_bounds(self, label, pad):
        """compute coordinates of bounding box around a masked object, with given padding

        Args:
            label (np.array): mask of the object
            pad (float): padding around object in um

        Returns:
            ints: integer coordinates of bounding box
        """
        image_shape = self.image_shape
        res = self.resolution

        r = np.any(label, axis=(1, 2))
        c = np.any(label, axis=(0, 2))
        z = np.any(label, axis=(0, 1))
        rmin, rmax = np.where(r)[0][[0, -1]]
        rmin = np.amax((0, math.floor(rmin - pad / res[0])))
        rmax = np.amin((image_shape[0], math.ceil(rmax + (pad + 1) / res[0])))
        cmin, cmax = np.where(c)[0][[0, -1]]
        cmin = np.amax((0, math.floor(cmin - (pad) / res[1])))
        cmax = np.amin((image_shape[1], math.ceil(cmax + (pad + 1) / res[1])))
        zmin, zmax = np.where(z)[0][[0, -1]]
        zmin = np.amax((0, math.floor(zmin - (pad) / res[2])))
        zmax = np.amin((image_shape[2], math.ceil(zmax + (pad + 1) / res[2])))
        return int(rmin), int(rmax), int(cmin), int(cmax), int(zmin), int(zmax)

    def endpoints_from_coords_neighbors(self, coords):
        """Compute endpoints of fragment.

        Args:
            coords (np.array): coordinates of voxels in the fragment

        Returns:
            list: endpoints of the fragment
        """
        res = self.resolution

        dims = np.multiply(np.amax(coords, axis=0) - np.amin(coords, axis=0), res)
        max_length = np.sqrt(np.sum([dim ** 2 for dim in dims]))

        r = 15
        if max_length < r:
            radius = max_length / 2
            close_enough = radius
        else:
            radius = r
            close_enough = 9

        A = radius_neighbors_graph(
            coords, radius=radius, metric="wminkowski", metric_params={"w": res}
        )
        degrees = np.squeeze(np.array(np.sum(A, axis=1).T, dtype=int))
        indices = np.argsort(degrees)
        sorted = [degrees[i] for i in indices]

        # point with fewest neighbors
        ends = [coords[indices[0], :]]
        # second endpoint is point with fewest neighbors that is not within "close_enough" of the first endpoint
        # close_enough gets smaller until a second point is found
        while len(ends) < 2:
            for coord_idx, degree in zip(indices, sorted):
                coord = coords[coord_idx, :]
                dists = np.array(
                    [np.linalg.norm(np.multiply(coord - end, res)) for end in ends]
                )
                if not any(dists < close_enough):
                    ends.append(coord)
                    break
            close_enough = close_enough / 2

        return ends

    def compute_states_thread(self, corner1, corner2):
        print(f"Computing state representations @corner {corner1}")
        fragments = zarr.open(self.fragment_path, mode="r")
        image_tiered = zarr.open(self.tiered_path, mode="r")
        labels = fragments[
            corner1[0] : corner2[0], corner1[1] : corner2[1], corner1[2] : corner2[2]
        ]

        unq = np.unique(labels)
        components = unq[unq != 0]

        results = []
        for component in tqdm(
            components, desc=f"Computing state representations @corner {corner1}"
        ):
            if component in self.soma_lbls:
                results.append((component, None, None, None, None, None))

            mask = labels == component
            rmin, rmax, cmin, cmax, zmin, zmax = self.compute_bounds(mask, pad=1)
            mask = mask[rmin:rmax, cmin:cmax, zmin:zmax]

            skel = morphology.skeletonize_3d(mask)

            coords_mask = np.argwhere(mask)
            coords_skel = np.argwhere(skel)
            if len(coords_skel) < 4:
                coords = coords_mask
            else:
                coords = coords_skel

            endpoints = self.endpoints_from_coords_neighbors(coords)
            a = endpoints[0]
            try:
                b = endpoints[1]
            except:
                print(f"only 1 endpoint for component {component}")
                raise ValueError

            a = np.add(a, [rmin, cmin, zmin])
            b = np.add(b, [rmin, cmin, zmin])
            dif = b - a
            dif = dif / np.linalg.norm(dif)

            a = [int(x) for x in a]
            b = [int(x) for x in b]

            xlist, ylist, zlist = Bresenham3D(a[0], a[1], a[2], b[0], b[1], b[2])
            sum = np.sum(image_tiered.vindex[xlist, ylist, zlist])
            if sum < 0:
                warnings.warn(f"Negative int cost for comp {component}: {sum}")

            results.append((component, a, b, -dif, dif, sum))
        return results

    class state:
        def __init__(
            self, id, fragment, point1, point2, orientation1, orientation2, image_cost
        ):
            self.id = id
            self.fragment = fragment
            self.point1 = point1
            self.point2 = point2
            self.orientation1 = orientation1
            self.orientation2 = orientation2
            self.image_cost = image_cost

            self.soma_connection_point = None

    def compute_states(self):
        print(f"Computing states")
        items = self.image_path.split(".")
        states_fname = items[0] + "_states.pickle"

        specifications = self._get_frag_specifications()

        results = Parallel(n_jobs=self.parallel)(
            delayed(self.compute_states_thread)(
                specification["corner1"],
                specification["corner2"],
            )
            for specification in specifications
        )

        state_num = 0
        states = []
        for result in results:
            component, a, b, oa, ob, sum = result
            state = self.state(
                id=state_num,
                fragment=component,
                point1=a,
                point2=b,
                orientation1=-oa,
                orientation2=ob,
                image_cost=sum,
            )
            states.append(state)
            state_num += 1
            state = self.state(
                id=state_num,
                fragment=component,
                point1=a,
                point2=b,
                orientation1=-oa,
                orientation2=ob,
                image_cost=sum,
            )
            state_num += 1
            states.append(state)
        print(f"*****************Number of states: {len(states)}*******************")

        with open(states_fname, "wb") as handle:
            pickle.dump(states, handle)
