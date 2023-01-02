'''
Inputs
'''
brain_ids = ["8650", "8649", "8788", "8589", "8590", "8613"]
brain_ids = ["8650", "8788", "8613", "8589"]
path_key = "transformed_mask"

'''
Script
'''
import random
import numpy as np

from brainrender import Scene, settings
from brainrender.actors import Volume
settings.SHOW_AXES = False

from rich import print
from myterial import orange
from pathlib import Path

from axon_data import brain2paths, brain2centers
from tqdm import tqdm
from cloudvolume import CloudVolume
from skimage import io
from skimage.transform import rescale
import scipy.ndimage as ndi
from bg_atlasapi.bg_atlas import BrainGlobeAtlas

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


## Plotting them overlaid doesnt work bc one of the vglut3 brains covers everything


print(f"[{orange}]Running example: {Path(__file__).name}")

scene = Scene(atlas_name="allen_mouse_50um",title="Output Axons")
atlas = BrainGlobeAtlas("allen_mouse_50um")
atlas_mask = atlas.annotation
atlas_mask = rescale(atlas_mask, 5/2)
atlas_mask = atlas_mask > 0
print(f"Atlas mask shape: {atlas_mask.shape}")

dr = scene.add_brain_region("DR", alpha=0.15)

type2id = {}
for brain_id in brain_ids:
    genotype = brain2paths[brain_id]["genotype"]
    if genotype not in type2id.keys():
        type2id[genotype] = [brain_id]
    else:
        new_val = type2id[genotype] + [brain_id]
        type2id[genotype] = new_val

vols_transformed_gad = [CloudVolume(brain2paths[id][path_key]) for id in type2id["tph2 gad2"]]
vols_transformed_vglut = [CloudVolume(brain2paths[id][path_key]) for id in type2id["tph2 vglut3"]]

for vols, color in zip([vols_transformed_gad, vols_transformed_vglut], ["Reds", "Greens"]):
    for i, vol in enumerate(tqdm(vols, desc="Processing volumes...")):
        if i == 0:
            im_total = np.zeros(vol.shape)
        im = np.zeros(vol.shape)
        im[:,:,:,:] = vol[:,:,:,:]
        # path = f"/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_axon/npy-files/{color}-{i}.tif"
        # io.imsave(path, im)
        # im = io.imread(path)
        im_total += im

    im_total = np.squeeze(im_total)
    im_total /= len(vols)

    # Process
    # Remove small components
    small_comp_mask = remove_small_objects(im_total>0,100)
    im_total[small_comp_mask == False] = 0

    im_total = rescale(im_total, 0.5)
    im_total[atlas_mask == 0] = 0

    im_total = np.swapaxes(im_total, 0, 2)

    print(im_total.shape)
    print(np.sum(im_total))
    print(f"{np.amin(im_total)} to {np.amax(im_total)}")

    # make a volume actor and add
    actor = Volume(
        im_total,
        voxel_size=20,  # size of a voxel's edge in microns
        as_surface=False,  # if true a surface mesh is rendered instead of a volume
        c=color,  # use matplotlib colormaps to color the volume
    )

    scene.add(actor)

# render
scene.content
scene.render()

# coords: A->P, D->V, L->R