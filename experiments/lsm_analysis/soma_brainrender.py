import random
import numpy as np

from brainrender import Scene
from brainrender.actors import Points

from rich import print
from myterial import orange
from pathlib import Path

from soma_rabies_somadetector_data import brain2paths
from cloudreg.scripts.transform_points import NGLink
from tqdm import tqdm

print(f"[{orange}]Running example: {Path(__file__).name}")


scene = Scene(atlas_name="allen_mouse_50um",title="Input Somas")

# Get a numpy array with (fake) coordinates of some labelled cells
dr = scene.add_brain_region("DR", alpha=0.15)

# Add to scene
colors = {"tph2 vglut3": "blue", "tph2 gad2": "red", "gad2 vgat": "green"}

for key in tqdm(brain2paths.keys(), desc="Going through samples"):
    if "atlas_viz" in brain2paths[key].keys():
        viz_link = brain2paths[key]["atlas_viz"]
        viz_link = NGLink(viz_link.split("json_url=")[-1])
        ngl_json = viz_link._json
        for layer in ngl_json['layers']:
            if layer['type'] == 'annotation':
                points = []
                for annot in tqdm(layer['annotations'], desc="Going thruogh points...", leave=False):
                    struct_coord = np.array(annot['point'])/5
                    try:
                        region = scene.atlas.structure_from_coords(struct_coord)
                        if region != 0 and np.any(struct_coord < 0) == False:
                            points.append(annot['point'])
                    except IndexError:
                        continue
                points = np.array(points)*10

        scene.add(Points(points, name="CELLS", colors=colors[brain2paths[key]["genotype"]]))

# render
scene.content
scene.render()

# coords: A->P, D->V, L->R