from brainlit.preprocessing import removeSmallCCs
from brainlit.BrainLine.data.soma_data import brain2paths, brain2centers
from brainlit.BrainLine.util import (
    json_to_points,
    find_atlas_level_label,
    fold,
    setup_atlas_graph,
    get_atlas_level_nodes,
    download_subvolumes
)
from brainlit.BrainLine.apply_ilastik import ApplyIlastik, plot_results
from brainlit.BrainLine.parse_ara import *
import xml.etree.ElementTree as ET
from cloudreg.scripts.transform_points import NGLink
from brainlit.BrainLine.imports import *
from brainlit.BrainLine.util import find_sample_names

'''
Inputs
'''
data_dir = str(Path.cwd().parents[0])  # path to directory where training/validation data should be stored
brain = "test"
antibody_layer = "antibody"
background_layer = "background"
endogenous_layer = "endogenous"

dataset_to_save = "val"  # train or val

ilastik_path = "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh"
ilastik_project = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/matt_soma_rabies_pix_3ch.ilp"

'''
Setup
'''
brainlit_path = Path.cwd()
print(f"Path to brainlit: {brainlit_path}")
layer_names = [antibody_layer, background_layer, endogenous_layer]

if brain not in brain2paths.keys():
    raise ValueError(f"brain {brain} not an entry in brain2paths in axon_data.py file")

if f"{dataset_to_save}_info" not in brain2paths[
    brain
].keys() or dataset_to_save not in ["train", "val"]:
    raise ValueError(f"{dataset_to_save}_info not in brain2paths[{brain}].keys()")

'''
Download data
'''

download_subvolumes(data_dir, brain_id = brain, layer_names = layer_names, dataset_to_save = dataset_to_save)
doubles = []

'''
Apply ilastik
'''
brains = [
    brain
]  # sample IDs to be processed
applyilastik = ApplyIlastik(ilastk_path = ilastik_path, project_path = ilastik_project, brains_path = f"{data_dir}/", brains = brains)
applyilastik.process_somas()

'''
Check results
'''
plot_results(data_dir, brain_id = brain, doubles=doubles)

'''
Making info files for transformed images
'''
make_trans_layers = input(f"Will you be transforming image data into atlas space? (should relevant info files be made) (y/n)")
if make_trans_layers == "y":
    atlas_vol = CloudVolume(
        "precomputed://https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_10um/annotation_10um_2017"
    )
    for layer in [
        antibody_layer,
        background_layer,
    ]:  # axon_mask is transformed into an image because nearest interpolation doesnt work well after downsampling
        layer_path = brain2paths[brain]["base"] + layer + "_transformed"
        print(f"Writing info file at {layer_path}")
        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type="image",
            data_type="uint16",  # Channel images might be 'uint8'
            encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
            resolution=atlas_vol.resolution,  # Voxel scaling, units are in nanometers
            voxel_offset=atlas_vol.voxel_offset,
            chunk_size=[32, 32, 32],  # units are voxels
            volume_size=atlas_vol.volume_size,  # e.g. a cubic millimeter dataset
        )
        vol_mask = CloudVolume(layer_path, info=info)
        vol_mask.commit_info()