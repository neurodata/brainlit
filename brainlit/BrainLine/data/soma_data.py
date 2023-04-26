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
    "biccn": {
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=8VHD-seixZg9pg",
        "subtype": "test"
    },
    "8557": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2021_10_06/8557/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=3p4kSvgMMFl61Q",
            "somas_layer": "<unknown>",
            "nonsomas_layer": "<unknown>",
        },
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=4M6ppFjZAUijkg",
        "subtype": "tph2 vglut3",
    },
    "8555": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2021_12_01/8555/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=xs4d74cYlU9JOg",
            "somas_layer": "<unknown>",
            "nonsomas_layer": "<unknown>",
        },
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=qmoYCOl5qUBzkQ",
        "subtype": "tph2 vglut3",
    },
    "8607": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2021_12_02/8607/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=ygWt5ejevaABZQ",
            "somas_layer": "<unknown>",
            "nonsomas_layer": "<unknown>",
        },
        "somas_atlas_path": "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/wholebrain_results/atlas_somas_8607/",
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=UtK0gLiq_8WlQA",
        "subtype": "tph2 gad2",
    },
    "8468": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_01_19/8468/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=KUfdi6YMYWLbsg",
            "somas_layer": "<unknown>",
            "nonsomas_layer": "<unknown>",
        },
    },
    "8606": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_03_15/8606/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=18lUZ-z6xx15Rg",
            "somas_layer": "<unknown>",
            "nonsomas_layer": "<unknown>",
        },
        "somas_atlas_path": "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/wholebrain_results/atlas_somas_8606/",
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=guSqN8FCetYXGA",
        "subtype": "tph2 gad2",
    },
    "8529": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_03_02/8529/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=miguDvzAX3SlXQ",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "vizlink": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=kLWMLcZ066yYpQ",
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=eeigeCQSqG8znw",
        "subtype": "gad2 vgat",
    },
    "8477": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_03_14/8477/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=1NvqIFK_NyImiQ",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "vizlink": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=UgEYYM3ycYGC7A",
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=qACBTohd8X4DcA",
        "subtype": "gad2 vgat",
    },
    "8531": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_03_10/8531/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=4Qfka7AySR8k5A",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "vizlink": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=42tANDn1cjREEA",
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=5Kow5I-W6eVGUw",
        "subtype": "gad2 vgat",
    },
    "8608": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_04_13/8608/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=NRFI2aWmv3d0Ww",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=sAjrqYlYDLkF_A",
        "subtype": "tph2 gad2",
    },
    "8446": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_03_25/8446/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=8okqpTPUtDXJLw",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "train_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=8okqpTPUtDXJLw",
            "somas_layer": "soma_train",
            "nonsomas_layer": "nonsoma_train",
        },
        "vizlink": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=C4DKojFJSgEo-A",
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=HXtu5wki04Y0yw",
        "subtype": "gad2 vgat",
    },
    "8454": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_03_09/8454/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=7g3CvrESEx1TsA",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "vizlink": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=7ugLt3twr6RBIA",
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=LgrOarq-AeFUPQ",
        "subtype": "gad2 vgat",
    },
    "887": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_09_20/887/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=9fbb_i2khFWqEA",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "vizlink": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=9fbb_i2khFWqEA",
        "subtype": "tph2 vglut3",
    },
    "878": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_09_20/878/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=d1TTGG-eVOQ2lA",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "vizlink": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=d1TTGG-eVOQ2lA",
        "subtype": "tph2 vglut3",
    },
    "MPRRabies": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2023_01_20/MPRRabies/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=2U2yHGe-1YFPHA",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "train_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=2U2yHGe-1YFPHA",
            "somas_layer": "soma_train",
            "nonsomas_layer": "nonsoma_train",
        },
        "subtype": "tph2 gad2",
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=6O1ufCnHBBeK8Q",
        "somas_atlas_url_partial": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=iiX4cb9vLbHLXQ",
    },
    "969": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2023_03_15/969/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=kSzIi9RUiPLr9Q",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val"
        },
        "subtype": "tph2 gad2",
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=tYjwxkvRGgMd5g"
    },
    "910": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2023_03_16/910/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=YzUVCI2qZiqp3w",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val"
        },
        "train_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=QlJARbMY_pyoDQ",
            "somas_layer": "soma_train",
            "nonsomas_layer": "nonsoma_train"
        },
        "subtype": "tph2 gad2"
    },
    "892": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2023_03_31/892/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=H7h_cTXW4gLgWA",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "subtype": "tph2 vglut3",
        "somas_atlas_path": "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/wholebrain_results/atlas_somas_892/",
        "somas_atlas_url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=A4qnN-6JzqWCcw"
    },
    "MS37": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2023_04_12/MS37/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=NTaOrY8WpOLyKA",
            "somas_layer": "soma_val",
            "nonsomas_layer": "nonsoma_val",
        },
        "subtype": "gad2 vgat",
    },
    "pytest": {
        "somas_atlas_url": "https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=f9iYKQbYLFRMbA",
        "subtype": "test_type",
    },
}
