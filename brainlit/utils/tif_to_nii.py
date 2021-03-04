#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tif_to_nii
command line executable to convert a directory of tif images
(from one image) to a nifti image stacked along a user-specified axis
call as: python tif_to_nii.py /path/to/tif/ /path/to/nifti
ex. python tif_to_nii.py C:/Users/shrey/Downloads/benchmarking_tif C:/Users/shrey/Downloads/benchmarking_nifti
(append optional arguments to the call as desired)
"""

import argparse
from glob import glob
import os
from pathlib import Path
import sys

from PIL import Image
import nibabel as nib
import numpy as np

from skimage import io
from brainlit.utils.benchmarking_params import brain_offsets, vol_offsets, scales, type_to_date
from brainlit.utils.Neuron_trace import NeuronTrace




def arg_parser():
    parser = argparse.ArgumentParser(description='merge 2d tif images into a 3d image')
    parser.add_argument('img_dir', type=str,
                        help='path to tiff image directory')
    parser.add_argument('out_dir', type=str,
                        help='path to output the corresponding tif image slices')
    parser.add_argument('-a', '--axis', type=int, default=2,
                        help='axis on which to stack the 2d images')
    return parser


def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def main():
    try:
        args = arg_parser().parse_args()
        print(args)
        img_dir = Path(args.img_dir)
        swc_base_path = img_dir / "benchmarking_swc"
        fns = sorted([str(fn) for fn in img_dir.glob('*.tif*')])

        if not fns:
            raise ValueError(f'img_dir ({args.img_dir}) does not contain any .tif or .tiff images.')

        for fn in fns:
            _, base, ext = split_filename(fn)

            #converting tif to nifti (nii.gz)
            img = io.imread(fn, plugin="tifffile")
            #img = io.imread(fn, plugin="tifffile").astype(np.single)


            #makes directories of the format test_1, test_2 etc.
            #if not os.path.exists(os.path.join(args.out_dir, str(base)[:-4])):
                #os.makedirs(os.path.join(args.out_dir, str(base)[:-4]))
            #out_dir = os.path.join(args.out_dir, str(base)[:-4])

            #makes directories numbered 1-50, with validation starting at 26
            #this is bc innereye requires folders to be unique integers
            indx = base.find('_') + 1
            if base[0] == 'v':
                if not os.path.exists(os.path.join(args.out_dir, str(int(str(base)[indx:-4])+25))):
                    os.makedirs(os.path.join(args.out_dir, str(int(str(base)[indx:-4])+25)))
                out_dir = os.path.join(args.out_dir, str(int(str(base)[indx:-4])+25))

            elif base[0] == 't':
                if not os.path.exists(os.path.join(args.out_dir, str(base)[indx:-4])):
                    os.makedirs(os.path.join(args.out_dir, str(base)[indx:-4]))
                out_dir = os.path.join(args.out_dir, str(base)[indx:-4])

            nib.Nifti1Image(img, None).to_filename(os.path.join(out_dir, f'{base}.nii.gz'))


            #converting swc to npy
            f = Path(fn).parts[-1][:-8].split("_")
            print(f)
            image = f[0]
            date = type_to_date[image]
            num = int(f[1])
            scale = scales[date]
            brain_offset = brain_offsets[date]
            vol_offset = vol_offsets[date][num]
            im_offset = np.add(brain_offset, vol_offset)
            lower = int(np.floor((num - 1) / 5) * 5 + 1)
            upper = int(np.floor((num - 1) / 5) * 5 + 5)
            dir1 = date + "_" + image + "_" + str(lower) + "-" + str(upper)
            dir2 = date + "_" + image + "_" + str(num)
            swc_path = swc_base_path / dir1 / dir2
            swc_files = list(swc_path.glob("**/*.swc"))

            paths_total = []
            for swc_num, swc in enumerate(swc_files):
                if "0" in swc.parts[-1]:
                    # skip the bounding box swc
                    continue
                swc_trace = NeuronTrace(path=str(swc))
                paths = swc_trace.get_paths()
                swc_offset, _, _, _ = swc_trace.get_df_arguments()
                offset_diff = np.subtract(swc_offset, im_offset)

                for path_num, p in enumerate(paths):
                    pvox = (p + offset_diff) / (scale) * 1000
                    paths_total.append(pvox)

            np_paths_total = np.asarray(paths_total, dtype=object)
            np.save((os.path.join(out_dir, f'{base[:-4]}.npy')), np_paths_total)

        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())