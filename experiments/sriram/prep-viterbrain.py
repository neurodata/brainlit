from brainlit.utils.write import czi_to_zarr, zarr_to_omezarr, write_trace_layer
import zarr
from cloudvolume import CloudVolume
import json
from pathlib import Path
import time
from brainlit.algorithms.generate_fragments.state_generation import state_generation
import numpy as np

project_path = "/cis/project/sriram/ng_data/sriram-adipo-brain1-im3-timing" 
project_path = Path(project_path)
czi_path = project_path.parents[1] / "Sriram" / "SS IUE 175 SNOVA RFP single channel AdipoClear Brain 1 full Image 3.czi"  # path to czi image
fg_channel = 0
ilastik_program_path = "/cis/home/tathey/ilastik-1.4.0-Linux/run_ilastik.sh"   # path to ilastik executable e.g. for windows something like "\Program Files\ilastik-1.3.2\ilastik.exe"
parallel = 10

# /cis/home/tathey/ilastik-1.4.0-Linux/run_ilastik.sh  --headless --project=/cis/project/sriram/ilastik_dataaxon_segmentation.ilp /cis/project/sriram/ng_data/sriram-adipo-brain1-im3-timing/data_bin/image_0-20_0-400_0-400.h5 

times = {}

# # Convert to zarr

# start = time.time()
# zarr_paths = czi_to_zarr(
#     czi_path=czi_path, out_dir=project_path, fg_channel=fg_channel, parallel=2
# )
# times["Convert CZI to ZARR"]= time.time() - start
# print([f"{key}:{times[key]}" for key in times.keys()])

## Convert to OME zarr
ome_path = project_path / "fg_ome.zarr"  # path of ome zarr to be made
resolution = [510, 510, 3000]  # xyz resolution in nm

# start = time.time()
# zarr_to_omezarr(zarr_path=zarr_paths[0], out_path=ome_path, res=resolution)
# times["Convert ZARR to OME-ZARR"]= time.time() - start
# print([f"{key}:{times[key]}" for key in times.keys()])

# start = time.time()
# write_trace_layer(parent_dir=project_path, res=resolution)
# times["Make trace layer"]= time.time() - start
# print([f"{key}:{times[key]}" for key in times.keys()])



image_path = project_path / "fg_ome.zarr" / "0"
z_im = zarr.open_array(image_path)
ilastik_project_path = project_path.parents[1] / "ilastik_data" / "axon_segmentation.ilp"
chunk_size = [c * np.amax([1, int(np.ceil(100/c))]) for c in z_im.chunks]
data_bin = project_path / "data_bin/"

prob_path = project_path / "probs.zarr"
fragment_path = project_path / "labels.zarr"
ome_path_lbl = project_path / "labels_ome.zarr"
tiered_path = project_path / "tiered.zarr"
states_path = project_path / "nx.pickle"

sg = state_generation(
    image_path=image_path,
    new_layers_dir=project_path,
    ilastik_program_path=ilastik_program_path,
    ilastik_project_path=ilastik_project_path,
    chunk_size=chunk_size,
    parallel=parallel,
    resolution=[
        resolution[2] / 1000,
        resolution[0] / 1000,
        resolution[1] / 1000,
    ],  # ome zarr is zxy, in microns
    prob_path=prob_path,
    # fragment_path=fragment_path,
    # tiered_path=tiered_path,
    # states_path=states_path,
)


# start = time.time()
# sg.predict(data_bin=data_bin)
# times["Run ilastik"]= time.time() - start
# print([f"{key}:{times[key]}" for key in times.keys()])

start = time.time()
sg.compute_frags()
times["Compute Fragments"]= time.time() - start
print([f"{key}:{times[key]}" for key in times.keys()])


start = time.time()
zarr_to_omezarr(zarr_path=fragment_path, out_path=ome_path_lbl, res=resolution)
times["Convert fragments to ZARR"]= time.time() - start
print([f"{key}:{times[key]}" for key in times.keys()])


sg.compute_soma_lbls()
start = time.time()
sg.compute_image_tiered()
times["Compute tiered image"]= time.time() - start
print([f"{key}:{times[key]}" for key in times.keys()])

start = time.time()
sg.compute_states()
times["Compute states"]= time.time() - start
print([f"{key}:{times[key]}" for key in times.keys()])

start = time.time()
sg.compute_edge_weights(str(ome_path_lbl / "0"))
times["Compute edge weights"]= time.time() - start
print([f"{key}:{times[key]}" for key in times.keys()])

with open(project_path / 'times.pickle', 'wb') as handle:
    pickle.dump(times, handle)