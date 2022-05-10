from brainlit.utils.Neuron_trace import NeuronTrace
import numpy as np
from skimage import io
import os
from scipy.ndimage.morphology import distance_transform_edt
from pathlib import Path
from brainlit.viz.swc2voxel import Bresenham3D
from brainlit.utils.benchmarking_params import (
    brain_offsets,
    vol_offsets,
    scales,
    type_to_date,
)


def make_masks(data_dir):
    """Swc to numpy mask
        Args:
            data_dir: direction to base data folder that download_benchmarking points to.
            Should contain sample-tif-location and sample-swc-location
        Returns:
            Saved numpy masks in data-dir/mask-location for each image in sample-tif-location
    """
    im_dir = Path(os.path.join(data_dir, "sample-tif-location"))
    swc_dir = Path(os.path.join(data_dir, "sample-swc-location"))
    mask_dir = os.path.join(data_dir, "mask-location")
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # loading all the benchmarking images from local paths

    # mask_dir = data_dir / "benchmarking_masks"
    gfp_files = list(im_dir.glob("**/*.tif"))
    # swc_base_path = data_dir / "Manual-GT"
    save = True

    for im_num, im_path in enumerate(gfp_files):
        # loading one gfp image
        im = io.imread(im_path, plugin="tifffile")
        im = np.swapaxes(im, 0, 2)

        file_name = im_path.parts[-1][:-8]

        f = im_path.parts[-1][:-8].split("_")
        image = f[0]
        date = type_to_date[image]
        num = int(f[1])

        scale = scales[date]
        brain_offset = brain_offsets[date]
        vol_offset = vol_offsets[date][num]
        im_offset = np.add(brain_offset, vol_offset)

        # loading all the .swc files corresponding to the image
        # all the paths of .swc files are saved in variable swc_files
        lower = int(np.floor((num - 1) / 5) * 5 + 1)
        upper = int(np.floor((num - 1) / 5) * 5 + 5)
        dir1 = date + "_" + image + "_" + str(lower) + "-" + str(upper)
        dir2 = date + "_" + image + "_" + str(num)
        swc_path = swc_dir / "Manual-GT" / dir1 / dir2
        swc_files = list(swc_path.glob("**/*.swc"))

        paths_total = []
        labels_total = np.zeros(im.shape)

        # generate paths and save them into paths_total
        for swc_num, swc in enumerate(swc_files):
            if "cube" in swc.parts[-1]:
                # skip the bounding box swc
                continue
            swc = str(swc)
            swc_trace = NeuronTrace(path=swc)
            paths = swc_trace.get_paths()
            swc_offset, _, _, _ = swc_trace.get_df_arguments()
            offset_diff = np.subtract(swc_offset, im_offset)

            # for every path in that swc
            for path_num, p in enumerate(paths):
                pvox = (p + offset_diff) / (scale) * 1000
                paths_total.append(pvox)

        # generate labels by using paths
        for path_voxel in paths_total:
            for voxel_num, voxel in enumerate(path_voxel):
                if voxel_num == 0:
                    continue
                voxel_prev = path_voxel[voxel_num - 1, :]
                xs, ys, zs = Bresenham3D(
                    int(voxel_prev[0]),
                    int(voxel_prev[1]),
                    int(voxel_prev[2]),
                    int(voxel[0]),
                    int(voxel[1]),
                    int(voxel[2]),
                )
                for x, y, z in zip(xs, ys, zs):
                    vox = np.array((x, y, z))
                    if (vox >= 0).all() and (vox < im.shape).all():
                        labels_total[x, y, z] = 1

        label_flipped = labels_total * 0
        label_flipped[labels_total == 0] = 1
        dists = distance_transform_edt(label_flipped, sampling=scale)
        labels_total[dists <= 1000] = 1

        if save:
            im_file_name = file_name + "_mask.npy"
            out_file = mask_dir + "/" + im_file_name
            np.save(out_file, labels_total)
