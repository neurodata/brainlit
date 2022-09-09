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
from scipy.stats import gaussian_kde
from brainlit.viz.swc2voxel import Bresenham3D
from brainlit.algorithms.connect_fragments import ViterBrain
import math
import warnings
import subprocess
import random
import pickle
import networkx as nx
from typing import List, Tuple

import pcurve.pcurve as pcurve


class state_generation:
    def __init__(
        self,
        image_path: str,
        ilastik_program_path: str,
        ilastik_project_path: str,
        fg_channel: int,
        chunk_size: List[float] = [375, 375, 125],
        soma_coords: List[list] = [],
        resolution: List[float] = [0.3, 0.3, 1],
        parallel: int = 1,
        prob_path: str = None,
        fragment_path: str = None,
        tiered_path: str = None,
        states_path: str = None,
    ) -> None:
        """This class encapsulates the processing that turns an image into a set of fragments with endpoints etc. needed to perform viterbrain tracing.

        Args:
            image_path (str): Path to image zarr.
            ilastik_program_path (str): Path to ilastik program.
            ilastik_project_path (str): Path to ilastik project for segmentation of image.
            fg_channel (int): Channel of image taken to be foreground.
            chunk_size (List[float], optional): Chunk size too be used in parallel processing. Defaults to [375, 375, 125].
            soma_coords (List[list], optional): List of coordinates of soma centers. Defaults to [].
            resolution (List[float], optional): Resolution of image in microns. Defaults to [0.3, 0.3, 1].
            parallel (int, optional): Number of threads to use for parallel processing. Defaults to 1.
            prob_path (str, optional): Path to alrerady computed probability image (ilastik output). Defaults to None.
            fragment_path (str, optional): Path to alrerady computed fragment image. Defaults to None.
            tiered_path (str, optional): Path to alrerady computed tiered image. Defaults to None.
            states_path (str, optional): Path to alrerady computed states file. Defaults to None.

        Raises:
            ValueError: Image must be four dimensional (cxyz)
            ValueError: Chunks must include all channels and be 4D.
            ValueError: Already computed images must match image in spatial dimensions.
        """

        self.image_path = image_path
        image = zarr.open(image_path, mode="r")
        if len(image.shape) != 4:
            raise ValueError(f"Image must be 4D (cxyz), rather than shape: {image.shape}")

        self.fg_channel = fg_channel
        self.image_shape = image.shape
        self.ilastik_program_path = ilastik_program_path
        self.ilastik_project_path = ilastik_project_path

        if len(chunk_size) != 4 or chunk_size[0] != self.image_shape[0]:
            raise ValueError(f"Chunk size must include all channels and be 4D (cxyz), not {chunk_size}")

        self.chunk_size = chunk_size
        self.soma_coords = soma_coords
        self.resolution = resolution
        self.parallel = parallel

        for other_im, name in zip(
            [prob_path, fragment_path, tiered_path], ["prob", "frag", "tiered"]
        ):
            if other_im is not None:
                other_image = zarr.open(other_im, mode="r")
                if other_image.shape != self.image_shape[1:]:
                    raise ValueError(f"{name} image has different shape {other_image.shape} than image {self.image_shape}")


        self.prob_path = prob_path
        self.fragment_path = fragment_path
        self.tiered_path = tiered_path
        self.states_path = states_path

    def _predict_thread(
        self, corner1: List[int], corner2: List[int], data_bin: str
    ) -> None:
        """Execute ilastik on an image chunk

        Args:
            corner1 (list of ints): first corner of image chunk
            corner2 (list of ints): second corner of image chunk
            data_bin (str): path to directory to store intermediate files
        """
        image = zarr.open(self.image_path, mode="r")
        image_chunk = np.squeeze(
            image[
                :,
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

    def predict(self, data_bin: str) -> None:
        """Run ilastik on zarr image

        Args:
            data_bin (str): path to directory to store intermediate files
        """
        image = zarr.open(self.image_path, mode="r")
        items = self.image_path.split(".")
        prob_fname = items[0] + "_probs.zarr"

        probabilities = zarr.open(
            prob_fname, mode='w',
            shape = np.squeeze(image.shape[1:]), chunks=image.chunks[1:], dtype="float"
        )
        chunk_size = self.chunk_size

        print(
            f"Processing image of shape {image.shape} with chunks {image.chunks} into probability image {prob_fname} of shape {probabilities.shape}"
        )

        for x in tqdm(
            np.arange(0, image.shape[1], chunk_size[1]),
            desc="Computing Ilastik Predictions",
        ):
            x2 = np.amin([x + chunk_size[1], image.shape[1]])
            for y in tqdm(np.arange(0, image.shape[2], chunk_size[2]), leave=False):
                y2 = np.amin([y + chunk_size[2], image.shape[2]])
                Parallel(n_jobs=self.parallel)(
                    delayed(self._predict_thread)(
                        [x, y, z],
                        [x2, y2, np.amin([z + chunk_size[3], image.shape[3]])],
                        data_bin,
                    )
                    for z in np.arange(0, image.shape[3], chunk_size[3])
                )

                for f in os.listdir(data_bin):
                    fname = os.path.join(data_bin, f)
                    if "Probabilities" in f:
                        items = f.split("_")
                        z = int(items[1])
                        z2 = np.amin([z + chunk_size[3], image.shape[3]])
                        f = h5py.File(fname, "r")
                        pred = f.get("exported_data")
                        pred = np.squeeze(pred[1, :, :, :])

                        probabilities[x:x2, y:y2, z:z2] = pred

                    if "image" in f:
                        os.remove(fname)

        self.prob_path = prob_fname

    def _get_frag_specifications(self) -> list:
        image = zarr.open(self.image_path, mode="r")
        chunk_size = self.chunk_size
        soma_coords = self.soma_coords

        specifications = []

        for x in np.arange(0, image.shape[1], chunk_size[1]):
            x2 = np.amin([x + chunk_size[1], image.shape[1]])
            for y in np.arange(0, image.shape[2], chunk_size[2]):
                y2 = np.amin([y + chunk_size[2], image.shape[2]])
                for z in np.arange(0, image.shape[3], chunk_size[3]):
                    z2 = np.amin([z + chunk_size[3], image.shape[3]])
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

    def _split_frags_thread(
        self, corner1: List[int], corner2: List[int], soma_coords: List[list] = []
    ) -> Tuple[List[int], List[int], np.ndarray]:
        """Compute fragments of image chunk

        Args:
            corner1 (list of ints): first corner of image chunk
            corner2 (list of ints): second corner of image chunk
            soma_coords (list, optional): list of soma centerpoint coordinates. Defaults to [].

        Returns:
            tuple: tuple containing corner coordinates and fragment image
        """
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
            soma_coords, labels, im_processed, res=self.resolution, verbose=False
        )
        mask = labels > 0
        mask2 = image_process.removeSmallCCs(mask, 25, verbose=False)
        image_iterative[mask & (~mask2)] = 0

        states, comp_to_states = image_process.split_frags_place_points(
            image_iterative=image_iterative,
            labels=labels,
            radius_states=radius_states,
            res=self.resolution,
            threshold=threshold,
            states=states,
            comp_to_states=comp_to_states,
            verbose=False,
        )

        new_labels = image_process.split_frags_split_comps(
            labels, new_soma_masks, states, comp_to_states, verbose=False
        )

        new_labels = image_process.split_frags_split_fractured_components(
            new_labels, verbose=False
        )

        props = measure.regionprops(new_labels)
        for _, prop in enumerate(
            tqdm(props, desc="remove small fragments", disable=True)
        ):
            if prop.area < 15:
                new_labels[new_labels == prop.label] = 0

        new_labels = image_process.rename_states_consecutively(new_labels)

        return (corner1, corner2, new_labels)

    def compute_frags(self) -> None:
        """Compute all fragments for image"""
        probs = zarr.open(self.prob_path, mode="r")
        items = self.image_path.split(".")
        frag_fname = items[0] + "_labels.zarr"
        fragments = zarr.open(
            frag_fname, mode='w',
            shape = np.squeeze(probs.shape), chunks=probs.chunks, dtype="uint16"
        )

        print(f"Constructing fragment image {frag_fname} of shape {fragments.shape}")

        specifications = self._get_frag_specifications()

        results = Parallel(n_jobs=self.parallel)(
            delayed(self._split_frags_thread)(
                specification["corner1"],
                specification["corner2"],
                specification["soma_coords"],
            )
            for specification in specifications
        )

        max_label = 0
        for result in results:
            corner1, corner2, labels = result
            labels[labels > 0] += max_label
            max_label = np.amax([max_label, np.amax(labels)])
            fragments[
                corner1[0] : corner2[0],
                corner1[1] : corner2[1],
                corner1[2] : corner2[2],
            ] = labels
        print(f"*****************Number of components: {max_label}*******************")

        self.fragment_path = frag_fname

    def compute_soma_lbls(self) -> None:
        """Compute fragment ids of soma coordinates."""
        fragments = zarr.open(self.fragment_path, mode="r")

        soma_lbls = []
        radius = 20
        for soma_coord in self.soma_coords:
            local_labels = fragments[
                np.amax([soma_coord[0] - radius, 0]) : soma_coord[0] + radius,
                np.amax([soma_coord[1] - radius, 0]) : soma_coord[1] + radius,
                np.amax([soma_coord[2] - radius, 0]) : soma_coord[2] + radius,
            ]
            soma_label = image_process.label_points(
                local_labels, [[radius, radius, radius]], res=self.resolution
            )[1][0]
            soma_lbls.append(soma_label)

        self.soma_lbls = soma_lbls

    def _compute_image_tiered_thread(
        self, corner1: List[int], corner2: List[int]
    ) -> Tuple[List[int], List[int], np.ndarray]:
        """Compute tiered image (image likelihood costs)

        Args:
            corner1 (list of ints): first corner of image chunk
            corner2 (list of ints): second corner of image chunk

        Returns:
            tuple: tuple containing corner coordinates and tiered image
        """
        print(f"Processing @corner: {corner1}-{corner2}")
        kde = self.kde
        image = zarr.open(self.image_path, mode="r")

        image = np.squeeze(image[
            self.fg_channel, corner1[0] : corner2[0], corner1[1] : corner2[1], corner1[2] : corner2[2]
        ])

        vals = np.unique(image)
        scores_neg = -1 * kde.logpdf(vals)

        data = np.reshape(np.copy(image), (image.size,))
        sort_idx = np.argsort(vals)
        idx = np.searchsorted(vals, data, sorter=sort_idx)
        out = scores_neg[sort_idx][idx]
        image_tiered = np.reshape(out, image.shape)

        return (corner1, corner2, image_tiered)

    def compute_image_tiered(self) -> None:
        """Compute entire tiered image then reassemble and save as zarr"""
        image = zarr.open(self.image_path, mode="r")
        fragments = zarr.open(self.fragment_path, mode="r")
        items = self.image_path.split(".")
        tiered_fname = items[0] + "_tiered.zarr"
        tiered = zarr.open(
            tiered_fname, mode='w',
            shape = np.squeeze(fragments.shape), chunks=fragments.chunks, dtype="uint16"
        )

        print(f"Constructing tiered image {tiered_fname} of shape {tiered.shape}")

        shp = np.array(np.array(image.shape) / 2).astype(int)
        image_chunk = np.squeeze(image[self.fg_channel, shp[1]:shp[1]+300, shp[2]:shp[2]+300, shp[3]:shp[3]+300])
        fragments_chunk = fragments[shp[1]:shp[1]+300, shp[2]:shp[2]+300, shp[3]:shp[2]+300]
        data_fg = image_chunk[fragments_chunk > 0]
        if len(data_fg.flatten()) > 10000:
            data_sample = random.sample(list(data_fg), k=10000)
        else:
            data_sample = data_fg

        kde = gaussian_kde(data_sample)

        self.kde = kde

        specifications = self._get_frag_specifications()

        results = Parallel(n_jobs=self.parallel)(
            delayed(self._compute_image_tiered_thread)(
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

        self.tiered_path = tiered_fname

    def _compute_bounds(
        self, label: np.ndarray, pad: float
    ) -> Tuple[int, int, int, int, int, int]:
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

    def _endpoints_from_coords_neighbors(self, coords: np.ndarray) -> List[list]:
        """Compute endpoints of fragment.

        Args:
            coords (np.array): coordinates of voxels in the fragment

        Returns:
            list: endpoints of the fragment
        """
        res = self.resolution

        dims = np.multiply(np.amax(coords, axis=0) - np.amin(coords, axis=0), res)
        max_length = np.sqrt(np.sum([dim**2 for dim in dims]))

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

    def _pc_endpoints_from_coords_neighbors(self, coords: np.ndarray) -> List[list]:
        """Compute endpoints of fragment with Principal Curves.

        Args:
            coords (np.array): coordinates of voxels in the fragment

        Returns:
            list: endpoints of the fragment

        References
        ----------
        .. [1] Hastie, Trevor, and Werner Stuetzle. “Principal Curves.”
        Journal of the American Statistical Association, vol. 84, no. 406,
        [American Statistical Association, Taylor & Francis, Ltd.], 1989,
        pp. 502–16, https://doi.org/10.2307/2289936.
        .. [2] Principal Curves Code written by zsteve,
        https://github.com/zsteve, https://github.com/zsteve/pcurvepy
        """

        ends = []

        p_curve = pcurve.PrincipalCurve(k=1, s_factor=5)
        p_curve.fit(coords, max_iter=50)
        pc = p_curve.p

        pc = np.asarray(np.floor(pc + 0.5), dtype=np.int64)

        ends.append(pc[0])
        ends.append(pc[-1])

        return ends

    def _compute_states_thread(
        self, corner1: List[int], corner2: List[int], alg: str = "nb"
    ) -> List[tuple]:
        """Compute states of fragments within image chunk

        Args:
            corner1 (list of ints): first corner of image chunk
            corner2 (list of ints): second corner of image chunk
            alg (string): algorithm to use for endpoint estimation. "nb" for neighborhood method, "pc" for principal curves method.

        Raises:
            ValueError: only one endpoint found for fragment

        Returns:
            [list]: list of tuples containing fragment and state information
        """
        fragments_zarr = zarr.open(self.fragment_path, mode="r")
        tiered_zarr = zarr.open(self.tiered_path, mode="r")
        labels = fragments_zarr[
            corner1[0] : corner2[0], corner1[1] : corner2[1], corner1[2] : corner2[2]
        ]
        image_tiered = tiered_zarr[
            corner1[0] : corner2[0], corner1[1] : corner2[1], corner1[2] : corner2[2]
        ]
        unq = np.unique(labels)
        components = unq[unq != 0]

        results = []
        for component in tqdm(
            components,
            desc=f"Computing state representations @corner {corner1}-{corner2}",
        ):
            mask = labels == component

            if component in self.soma_lbls:
                results.append(
                    (
                        component,
                        np.add(np.argwhere(mask), corner1),
                        None,
                        None,
                        None,
                        None,
                    )
                )
                continue


            rmin, rmax, cmin, cmax, zmin, zmax = self._compute_bounds(mask, pad=1)


            if component == 3:
                print(f"{np.argwhere(mask).shape} coords with bounds: {(rmin, rmax, cmin, cmax, zmin, zmax)}")
            # now in bounding box coordinates
            mask = mask[rmin:rmax, cmin:cmax, zmin:zmax]


            if component == 3:
                print(f"after crop: {np.argwhere(mask).shape}")

            skel = morphology.skeletonize_3d(mask)

            coords_mask = np.argwhere(mask)
            coords_skel = np.argwhere(skel)


            if component == 3:
                print(f"after skeletonize: {coords_mask.shape} vs {coords_skel.shape}")
            if len(coords_skel) < 4:
                coords = coords_mask
            else:
                coords = coords_skel

            if alg == "pc":
                endpoints_initial = self._pc_endpoints_from_coords_neighbors(coords)
            elif alg == "nb":
                try:
                    endpoints_initial = self._endpoints_from_coords_neighbors(coords)
                except ValueError:
                    raise ValueError(f"Component: {component} in corner {corner1}-{corner2} coords: {coords}")
            endpoints = endpoints_initial.copy()
            used_eps = np.zeros((len(endpoints), 3)) - 1
            for i, endpoint in enumerate(endpoints_initial):
                difs = np.multiply(np.subtract(coords_mask, endpoint), self.resolution)
                dists = np.linalg.norm(difs, axis=1)
                argmin = np.argmin(dists)

                while (coords_mask[argmin, :] == used_eps).all(1).any():
                    dists[argmin] = np.infty
                    argmin = np.argmin(dists)

                endpoints[i] = coords_mask[argmin, :]
                used_eps[i, :] = endpoints[i]
            a = endpoints[0]
            try:
                b = endpoints[1]
            except:
                print(f"only 1 endpoint for component {component}")
                raise ValueError

            # now in chunk coordinates
            a = np.add(a, [rmin, cmin, zmin])
            b = np.add(b, [rmin, cmin, zmin])
            dif = b - a
            dif = dif / np.linalg.norm(dif)

            a = [int(x) for x in a]
            b = [int(x) for x in b]

            xlist, ylist, zlist = Bresenham3D(a[0], a[1], a[2], b[0], b[1], b[2])
            sum = np.sum(image_tiered[xlist, ylist, zlist])
            if sum < 0:
                warnings.warn(f"Negative int cost for comp {component}: {sum}")

            # now in full image coordinates
            a = np.add(a, corner1)
            b = np.add(b, corner1)

            results.append((component, a, b, -dif, dif, sum))
        return results

    def compute_states(self, alg: str = "nb") -> None:
        """Compute entire collection of states

        Args:
            alg (string, optional): algorithm to use for endpoint estimation.
                "nb" for neighborhood method, "pc" for principal curves method. Defaults to "nb"

        Raises:
            ValueError: erroneously computed endpoints of soma state
        """
        print(f"Computing states")
        items = self.image_path.split(".")
        states_fname = items[0] + "_nx.pickle"

        specifications = self._get_frag_specifications()

        results_tuple = Parallel(n_jobs=self.parallel)(
            delayed(self._compute_states_thread)(
                specification["corner1"],
                specification["corner2"],
                alg,
            )
            for specification in specifications
        )
        results = [item for result in results_tuple for item in result]

        state_num = 0
        G = nx.DiGraph()
        soma_comp2state = {}
        for result in results:
            component, a, b, oa, ob, sum = result
            if component in self.soma_lbls:
                if b is not None:
                    raise ValueError(
                        f"Component {component} is a soma component but the state is not a soma: {result}"
                    )
                if component in soma_comp2state.keys():
                    coords1 = G.nodes[soma_comp2state[component]]["soma_coords"]
                    coords2 = a
                    coords = np.concatenate((coords1, coords2))
                    G.nodes[soma_comp2state[component]]["soma_coords"] = coords
                else:
                    G.add_node(
                        state_num, type="soma", fragment=component, soma_coords=a
                    )
                    soma_comp2state[component] = state_num
            else:
                G.add_node(
                    state_num,
                    type="fragment",
                    fragment=component,
                    point1=a,
                    point2=b,
                    orientation1=-oa,
                    orientation2=ob,
                    image_cost=sum,
                    twin=state_num + 1,
                )

                state_num += 1
                G.add_node(
                    state_num,
                    type="fragment",
                    fragment=component,
                    point1=b,
                    point2=a,
                    orientation1=-ob,
                    orientation2=oa,
                    image_cost=sum,
                    twin=state_num - 1,
                )

            state_num += 1
        print(
            f"*****************Number of states: {G.number_of_nodes()}*******************"
        )

        with open(states_fname, "wb") as handle:
            pickle.dump(G, handle)

        self.states_path = states_fname

    def compute_edge_weights(self) -> None:
        """Create viterbrain object and compute edge weights"""
        items = self.image_path.split(".")
        viterbrain_fname = items[0] + "_viterbrain.pickle"

        with open(self.states_path, "rb") as handle:
            G = pickle.load(handle)

        viterbrain = ViterBrain(
            G,
            self.tiered_path,
            fragment_path=self.fragment_path,
            resolution=self.resolution,
            coef_curv=1000,
            coef_dist=10,
            coef_int=1,
            parallel=self.parallel,
        )

        viterbrain.compute_all_costs_dist(
            frag_frag_func=viterbrain.frag_frag_dist,
            frag_soma_func=viterbrain.frag_soma_dist,
        )

        viterbrain.compute_all_costs_int()

        print(f"# Edges: {viterbrain.nxGraph.number_of_edges()}")

        with open(viterbrain_fname, "wb") as handle:
            pickle.dump(viterbrain, handle)

        self.viterbrain = viterbrain

    def compute_bfs(self) -> None:
        """Compute bfs from highest degree node"""
        nodes_sorted = sorted(
            self.viterbrain.nxGraph.degree, key=lambda x: x[1], reverse=True
        )

        print(
            f"bfs tree: {nx.bfs_tree(self.viterbrain.nxGraph, source=nodes_sorted[0][0])}"
        )
