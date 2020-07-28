from tqdm import tqdm
from glob import glob
import argparse
import numpy as np
from cloudvolume import CloudVolume, Skeleton, storage
import pandas as pd
from pathlib import Path
import tifffile as tf


def swc2skeleton(swc_file, origin=None):
    """Converts swc file into Skeleton object

    Arguments:
        swc_file {str} -- path to SWC file
    Keyword Arguments:
        origin {numpy array with shape (3,1)} -- origin of coordinate frame in microns, (default: None assumes (0,0,0) origin)
    Returns:
        skel {cloudvolume.Skeleton} -- Skeleton object of given SWC file
    """
    with open(swc_file, "r") as f:
        contents = f.read()
    # get every line that starts with a hashtag
    comments = [i.split(" ") for i in contents.split("\n") if i.startswith("#")]
    offset = np.array([float(j) for i in comments for j in i[2:] if "OFFSET" in i])
    color = [float(j) for i in comments for j in i[2].split(",") if "COLOR" in i]
    # set alpha to 0.0 so skeleton  is opaque
    color.append(0.0)
    color = np.array(color, dtype="float32")
    skel = Skeleton.from_swc(contents)
    # physical units
    # space can be 'physical' or 'voxel'
    skel.space = "physical"
    # hard coding parsing the id from the filename
    idx = swc_file.find("G")

    skel.id = int(swc_file[idx + 2 : idx + 5])

    # hard coding changing  data type of vertex_types
    skel.extra_attributes[-1]["data_type"] = "float32"
    skel.extra_attributes.append(
        {"id": "vertex_color", "data_type": "float32", "num_components": 4}
    )
    # add offset to vertices
    # and shift by origin
    skel.vertices += offset
    if origin is not None:
        skel.vertices -= origin
    # convert from microns to nanometers
    skel.vertices *= 1000
    skel.vertex_color = np.zeros((skel.vertices.shape[0], 4), dtype="float32")
    skel.vertex_color[:, :] = color

    return skel


def create_skeleton_layer(s3_bucket, skel_res, img_dims, num_res=7):
    """Creates segmentation layer for skeletons

    Arguments:
        s3_bucket {str} -- path to precomputed skeleton destination
        skel_res {list} -- x,y,z dimensions of highest res voxel size (nm)
        img_dims {list} -- x,y,z voxel dimensions of tiff images

    Keyword Arguments:
        num_res {int} -- number of image resolutions to be downsampled

    Returns:
        vol {cloudvolume.CloudVolume} -- CloudVolume to upload skeletons to
    """
    # create cloudvolume info
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint64",  # Channel images might be 'uint8'
        encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
        # Voxel scaling, units are in nanometers
        resolution=skel_res,
        voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size=[128, 128, 64],  # units are voxels
        volume_size=[i * 2 ** (num_res - 1) for i in img_dims],  # units are voxels
        skeletons="skeletons",
    )
    skel_info = {
        "@type": "neuroglancer_skeletons",
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "vertex_attributes": [
            {"id": "radius", "data_type": "float32", "num_components": 1},
            {"id": "vertex_types", "data_type": "float32", "num_components": 1},
            {"id": "vertex_color", "data_type": "float32", "num_components": 4},
        ],
    }
    # get cloudvolume info
    vol = CloudVolume(s3_bucket, info=info, parallel=True)
    [vol.add_scale((2 ** i, 2 ** i, 2 ** i)) for i in range(num_res - 1)]
    # vol.commit_info() COMMENT OUT IF LOCAL DIRECTORY

    # upload skeleton info to /skeletons/ dir
    with storage.SimpleStorage(vol.cloudpath) as stor:
        stor.put_json(str(Path("skeletons") / "info"), skel_info)

    return vol


def get_volume_info(brain_dir, num_resolutions=7):
    """Get the volume info from transform.txt file in directory and tiff volume

    Arguments:
        brain_dir {str} -- path to top level of brain directory(ex:~/2018-08-01)
        num_resolutions {int} -- num of resolutions of downsampling

    Returns:
        origin {list} -- x,y,z coordinate of coordinate frame in space in mircons
        vox_size {list} -- x,y,z resoltions of highest res brain images
        tiff_dims {list} --  x,y,z size of tiff images
    """
    tiff_dims = np.squeeze(tf.imread(str(Path(brain_dir) / "default.0.tif"))).T.shape
    transform = open(str(Path(brain_dir) / "transform.txt"), "r")
    vox_size = [
        float(s[4:].rstrip("\n")) * (0.5 ** (num_resolutions - 1))
        for s in transform.readlines()
        if "s" in s
    ]
    transform.seek(0)
    # origin from nm to mircons
    origin = [int(o[4:].rstrip("\n")) / 1000 for o in transform.readlines() if "o" in o]
    return origin, vox_size, tiff_dims


def create_skel_segids(swc_dir, origin):
    """ Create skeletons to be uploaded as precomputed format

    Arguments:
        swc_dir {str} -- path to consensus swc files
        origin {list} -- x,y,z coordinate of coordinate frame in space in mircons

    Returns:
        skeletons {list} -- swc skeletons to be pushed to bucket
        segids {list} --  list of ints for each swc's label
    """
    # if colors is None:
    p = Path(swc_dir)
    files = [str(i) for i in p.rglob(f"*.swc")]
    skeletons = []
    segids = []
    for i in tqdm(files, desc="converting swcs to neuroglancer format..."):
        skeletons.append(swc2skeleton(i, origin=origin))
        segids.append(skeletons[-1].id)
    return skeletons, segids


def main():
    """Runs the script to upload SWC files to S3 in neuroglancer format.

    Example:
    >> python swc2neuroglancer.py s3://mouse-light-viz/precomputed_volumes/brain1_segments ~/Downloads/swcs/17_9_19_consensus/2018-08-01-consensus-swcs
    """
    parser = argparse.ArgumentParser(
        "Convert a folder of SWC files to neuroglancer format and upload them to the given S3 bucket location."
    )
    parser.add_argument(
        "s3_bucket",
        help="S3 bucket path of the form s3://<bucket-name>/<path-to-layer>",
    )
    parser.add_argument(
        "brain_dir", help="Path to local directory where SWC files are located."
    )

    parser.add_argument(
        "--colors",
        help="File with colors for a single swc",
        nargs=1,
        type=str,
        default=None,
    )

    args = parser.parse_args()
    origin, vox_size, tiff_dims = get_volume_info(args.brain_dir)
    swc_dir = glob(f"{args.brain_dir}/*consensus-swcs")[0]

    skeletons, segids = create_skel_segids(swc_dir, origin)

    vol = create_skeleton_layer(args.s3_bucket, vox_size, tiff_dims)

    for skel in tqdm(skeletons, desc="uploading skeletons to S3.."):
        vol.skeleton.upload(skel)

    print("Uploaded segment ids: " + str(segids))


if __name__ == "__main__":
    main()
