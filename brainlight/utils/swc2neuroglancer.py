from tqdm import tqdm
from glob import glob
import argparse
import numpy as np
from cloudvolume import CloudVolume, Skeleton, storage
import pandas as pd


def swc2skeleton(swc_file, colors=None, origin=None, segid=None):
    """Converts swc file into Skeleton object
    
    Arguments:
        swc_file {str} -- path to SWC file
    Keyword Arguments:
        color {iterable with same length as number of vertices} - values to be used as vertex colors.
        origin {numpy array with shape (3,1)} -- origin of coordinate frame in microns, (default: None assumes (0,0,0) origin)
        segid {int} -- id associated with this skeleton. Default is None and is pulled from filename
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
    if segid == None:
        idx = swc_file.find("G")
        if idx == -1:
            skel.id = hash(swc_file)
        else:
            skel.id = int(swc_file[idx + 2 : idx + 5])
    else:
        skel.id = segid
    # hard coding changing  data type of vertex_types
    skel.extra_attributes[-1]["data_type"] = "float32"
    skel.extra_attributes.append(
        {"id": "vertex_color", "data_type": "float32", "num_components": 4}
    )
    # add offset to vertices
    # and shift by origin
    # print(offset)
    skel.vertices += offset
    if origin is not None:
        skel.vertices -= origin
    # convert from microns to nanometers
    skel.vertices *= 1000
    # print(skel.vertices)
    skel.vertex_color = np.zeros((skel.vertices.shape[0], 4), dtype="float32")
    if colors is None:
        skel.vertex_color[:, :] = color
    else:
        colors_normalized = (colors - np.min(colors)) / np.max(colors)
        print(colors_normalized)
        r = colors_normalized < 0.33
        g = (0.33 <= colors_normalized) & (colors_normalized < 0.66)
        b = colors_normalized >= 0.66
        skel.vertex_color[r, 0] = 1
        skel.vertex_color[g, 1] = 1
        skel.vertex_color[b, 2] = 1
        skel.vertex_color[:, 0] = colors_normalized

        skel.id = 1000  # new seg id to show that its colored
    # print(color)
    # print(skel.color)
    # print(skel.extra_attributes)
    return skel


def create_skeleton_layer(s3_bucket):
    """Creates segmentation layer for skeletons
    
    Arguments:
        s3_bucket {str} -- path to SWC file
    Returns:
        vol {cloudvolume.CloudVolume} -- CloudVolume to upload skeletons to
    """
    # create cloudvolume info
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint64",  # Channel images might be 'uint8'
        encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
        resolution=[299, 304, 988],  # Voxel scaling, units are in nanometers
        voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size=[128, 128, 64],  # units are voxels
        volume_size=[33792, 25600, 13312],  # units are voxels
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
    [vol.add_scale((2 ** i, 2 ** i, 2 ** i)) for i in range(7)]
    vol.commit_info()

    # upload skeleton info to /skeletons/ dir
    with storage.SimpleStorage(vol.cloudpath) as stor:
        stor.put_json("skeletons/info", skel_info)

    return vol


def main():
    """Runs the script to upload SWC files to S3 in neuroglancer format.
    
    """
    parser = argparse.ArgumentParser(
        "Convert a folder of SWC files to neuroglancer format and upload them to the given S3 bucket location."
    )
    parser.add_argument(
        "s3_bucket",
        help="S3 bucket path of the form s3://<bucket-name>/<path-to-layer>",
    )
    parser.add_argument(
        "swc_dir", help="Path to local directory where SWC files are located."
    )
    parser.add_argument(
        "--origin",
        help="Origin of the SWC files in microns. Default is 0 0 0",
        nargs=3,
        type=float,
        default=None,
    )
    parser.add_argument(
        "--colors",
        help="File with colors for a single swc",
        nargs=1,
        type=str,
        default=None,
    )

    args = parser.parse_args()
    if args.colors is None:
        files = glob(f"{args.swc_dir}/*.swc")
        skeletons = []
        segids = []
        for i in tqdm(files, desc="converting swcs to neuroglancer format..."):
            skeletons.append(swc2skeleton(i, origin=args.origin))
            segids.append(skeletons[-1].id)
    else:
        files = glob(args.swc_dir)  # a single file
        colors = pd.read_csv(
            args.colors[0], header=None, names=["idx", "SNR0"], index_col=0
        )
        skeletons = []
        segids = []
        for i in tqdm(files, desc="converting swcs to neuroglancer format..."):
            skeletons.append(
                swc2skeleton(i, colors=colors.SNR0.values, origin=args.origin)
            )
            segids.append(skeletons[-1].id)

    vol = create_skeleton_layer(args.s3_bucket)

    for i in tqdm(skeletons, desc="uploading skeletons to S3.."):
        vol.skeleton.upload(i)
    print(segids)


if __name__ == "__main__":
    main()
