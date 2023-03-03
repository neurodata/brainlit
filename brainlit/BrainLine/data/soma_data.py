"""
This file stores information on how to access neuroglancer data.

Data should be stored in the brain2paths dictionary, with entries like:

    "<sample ID>": {
        "base": "<Path to directory with layers with CloudVolume prependings (ending with forward slash)>",
        "val_info": {
            "url": "<neuroglancer URL>",
            "somas_layer": "<name of layer with coordinates on somas>",
            "nonsomas_layer": "<name of layer with coordinates on non-somas>",
        },
        "somas_atlas_url": "<neuroglancer URL with a single annotation layer which contains points of soma detections>",
        "subtype": "<subtype>"
        #Optional:
        "train_info": {
            "url": "<neuroglancer URL>",
            "somas_layer": "<name of layer with coordinates on somas>",
            "nonsomas_layer": "<name of layer with coordinates on non-somas>",
        },
    },

    e.g.
    
    "test": {
        "base": "precomputed://file:///Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/brainlit/BrainLine/data/example/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=15e9owS_Hr51fg",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=LTWdmg7lYf1nbA",
        "subtype": "test"
    },
"""


brain2paths = {
    "atlas": {
        "url": "precomputed://https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_10um/annotation_10um_2017",
        # "filepath": "/Users/thomasathey/Documents/mimlab/mouselight/ailey/ara/ara_10um.tif",  # atlas can be downloaded from here: https://neurodata.io/data/allen_atlas/
    },
    "test": {
        "base": "precomputed://file:///Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/brainlit/BrainLine/data/example/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=15e9owS_Hr51fg",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=LTWdmg7lYf1nbA",
        "subtype": "test_type",
    },
    "pytest": {
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=f9iYKQbYLFRMbA",
        "subtype": "test_type",
    },
}
