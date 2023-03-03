"""
This file stores information on how to access neuroglancer data.

Data should be stored in the brain2paths dictionary, with entries like:

    "<sample ID>": {
        "base": "<Path to directory with layers with CloudVolume prependings>",
        "transformed_mask": "<axon mask layer that was transformed to atlas space with CloudVolume prependings"
        "val_info": {
            "url": "<neuroglancer URL>",
            "layer": "<name of layer with points for subvolumes>",
        },
        "subtype": "<subtype>"
        #Optional:
        "train_info": {
            "url": "<neuroglancer URL>",
            "layer": "<name of layer with points for subvolumes>",
        },
    },

    e.g.

    "test": {
        "base": "precomputed://file:///Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/brainlit/BrainLine/data/example/",
        "transformed_mask": "precomputed://file:///Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/brainlit/BrainLine/data/example/axon_mask_transformed/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=rywXvJ3kVOV71w",
            "layer": "val",
        },
        "subtype": "test_type",
    }
"""

brain2paths = {
    "atlas": {
        "url": "precomputed://https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_10um/annotation_10um_2017",
        "filepath": "/Users/thomasathey/Documents/mimlab/mouselight/ailey/ara/ara_10um.tif",  # atlas can be downloaded from here: https://neurodata.io/data/allen_atlas/
    },
    "test": {
        "base": "precomputed://file:///Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/brainlit/BrainLine/data/example/",
        "transformed_mask": "precomputed://file:///Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/brainlit/BrainLine/data/example/axon_mask_transformed/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=rywXvJ3kVOV71w",
            "layer": "val",
        },
        "subtype": "test_type",
    },
    "pytest": {
        "subtype": "test_type",
    },
}
