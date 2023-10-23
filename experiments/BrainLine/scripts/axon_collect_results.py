from brainlit.BrainLine.analyze_results import collect_regional_segmentation
from pathlib import Path
import os

"""
Inputs
"""
brain_ids = [
    "MS32"
]  # list of sample IDs to be shown


brainline_exp_dir = Path(os.getcwd()) / Path(__file__).parents[1]
data_file = brainline_exp_dir / "data" / "axon_data.json"


min_coords = [
    -1,
    1025,
    -1,
]  # max coords or -1 if you want to process everything along that dimension
max_coords = [
    6036,
    -1,
    -1,
]  # max coords or -1 if you want to process everything along that dimension
ncpu = 11  # number of cores to use for collection
s3_reg = True

for brain in brain_ids:
    data_dir = (
        brainline_exp_dir / "data" / f"brain_temp_{brain}"
    )  # data_dir = "/data/tathey1/matt_wright/brain_temp/"  # directory to store temporary subvolumes for segmentation

    """
    Collect results
    """

    collect_regional_segmentation(
        brain_id=brain,
        data_file=data_file,
        outdir=data_dir,
        ncpu=ncpu,
        min_coords=min_coords,
        max_coords=max_coords,
        s3_reg=s3_reg,
    )

