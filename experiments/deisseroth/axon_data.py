'''
This file stores information on how to access neuroglancer data.

Data should be stored in the brain2paths variable, where the key is the brain sample ID as a string, and the value is a dictionary with the following information:
- "base" : path to directory 
'''

brain2paths = {
    "2": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2021_06_02_Sert_Cre/Ch_647"
    },
    "1": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2021_04_08/gad2cre_tph2flp_con_fon_8291/642",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2021_04_08/gad2cre_tph2flp_con_fon_8291/561",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2021_04_08/gad2cre_tph2flp_con_fon_8291/axon_mask",
    },
    "3": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2021_07_01_Sert_Cre_B/Ch_647",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2021_07_01_Sert_Cre_B/Ch_561",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2021_07_01_Sert_Cre_B/Ch_488",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2021_07_01_Sert_Cre_B/axon_mask",
        "atlas": "precomputed://s3://smartspim-precomputed-volumes/2021_07_01_Sert_Cre_B/atlas_to_target",
    },
    "4": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2021_07_15_Sert_Cre_R/Ch_647",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2021_07_15_Sert_Cre_R/Ch_561",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2021_07_15_Sert_Cre_R/Ch_488",
        "mask": "s3://smartspim-precomputed-volumes/2021_07_01_Sert_Cre_B/axon_mask",
    },
    "8613": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_01_14/8613/Ch_647",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_01_14/8613/Ch_561",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_01_14/8613/Ch_488",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_01_14/8613/axon_mask",
        "transformed_mask": "precomputed://s3://smartspim-precomputed-volumes/2022_01_14/8613/axon_mask_transformed",
    },
    "8604": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_02_02/8604/Ch_647",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_02_02/8604/Ch_561",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_02_02/8604/Ch_488",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_02_02/8604/axon_mask",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=K1JfdZmA4pqr1Q",
            "layer": "axon_val",
        },
    },
    "8650": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_01_21/8650/Ch_647",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_01_21/8650/Ch_561",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_01_21/8650/Ch_488",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_01_21/8650/axon_mask",
        "transformed_mask": "precomputed://s3://smartspim-precomputed-volumes/2022_01_21/8650/axon_mask_transformed",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=tnXt-hF7Uyuq-w",
            "layer": "val",
        },
    },
    "8589": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_11_03/8589/Ch_647",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_11_03/8589/Ch_561",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_11_03/8589/Ch_488",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_11_03/8589/axon_mask",
        "transformed_mask": "precomputed://s3://smartspim-precomputed-volumes/2022_11_03/8589/axon_mask_transformed",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=sCrJq3bVmN5N4Q",
            "layer": "val",
        },
    },
    "8590": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_03_04/8590/Ch_647_iso",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_03_04/8590/Ch_561_iso",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_03_04/8590/Ch_488_iso",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_03_04/8590/axon_mask",
        "transformed_mask": "precomputed://s3://smartspim-precomputed-volumes/2022_03_04/8590/axon_mask_transformed",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=gq91htCi35XdPw",
            "layer": "val",
        },
    },
    "8649": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_03_28/8649/Ch_647_iso",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_03_28/8649/Ch_561_iso",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_03_28/8649/Ch_488_iso",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_03_28/8649/axon_mask",
        "transformed_mask": "precomputed://s3://smartspim-precomputed-volumes/2022_03_28/8649/axon_mask_transformed",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=b0unpdsrz-bO6A",
            "layer": "val",
        },
        "train_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=b0unpdsrz-bO6A",
            "layer": "train",
        },
    },
    "8590_v2": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_07_29/8590/Ch_647_iso",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_07_29/8590/Ch_561_iso",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_07_29/8590/Ch_488_iso",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_07_29/8590/axon_mask",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=iDjM_B1mVW1drA",
            "layer": "val",
        },
    },
    "8612": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_04_12/8612/Ch_647_iso",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_04_12/8612/Ch_561_iso",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_04_12/8612/Ch_488_iso",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_04_12/8612/axon_mask",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=U0_OeyX-pY606w",
            "layer": "val",
        },
    },
    "8788": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_10_24/8788/Ch_647",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_10_24/8788/Ch_561",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_10_24/8788/Ch_488",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_10_24/8788/axon_mask",
        "transformed_mask": "precomputed://s3://smartspim-precomputed-volumes/2022_10_24/8788/axon_mask_transformed",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=a7nPluS2FyMT0g",
            "layer": "val",
        },
        "train_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=a7nPluS2FyMT0g",
            "layer": "train",
        },
    },
    "11537": {
        "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_10_26/11537/Ch_647",
        "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_10_26/11537/Ch_561",
        "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_10_26/11537/Ch_488",
        "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_10_26/11537/axon_mask",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=9MCP5DRs2D32Bg",
            "layer": "val",
        },
    },
    "8786": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_11_02/8786/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=YRB3G4Hn19TjIA",
            "layer": "val",
        },
    },
    "8790": {
        "base": "precomputed://s3://smartspim-precomputed-volumes/2022_11_01/8790/",
        "val_info": {
            "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=hhlEcKDIfqrY-w",
            "layer": "val",
        },
    },


    # apparently not an axon sample!
    # "887": {
    #     "ab": "precomputed://s3://smartspim-precomputed-volumes/2022_09_20/887/Ch_647_iso",
    #     "bg": "precomputed://s3://smartspim-precomputed-volumes/2022_09_20/887/Ch_561_iso",
    #     "endo": "precomputed://s3://smartspim-precomputed-volumes/2022_09_20/887/Ch_488_iso",
    #     "mask": "precomputed://s3://smartspim-precomputed-volumes/2022_09_20/887/axon_mask",
    #     "val_info": {
    #         "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=eBYBAm6K90IWdw",
    #         "layer": "val",
    #     },
    #     "train_info": {
    #         "url": "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=eBYBAm6K90IWdw",
    #         "layer": "train",
    #     },
    # },
}


'''
Deprecated method of storing subvolumes for training/validation
'''
brain2centers = {
    "2": [
        [
            [2626, 3837, 2366],
            [799, 3315, 2366],
            [3851, 5443, 2346],
            [3260, 5490, 2351],
            [3198, 7658, 2351],
            [2998, 4502, 1215],
            [4269, 1657, 1215],
            [3071, 1153, 3412],
            [3672, 1792, 409],
            [2235, 4195, 409],
        ]
    ],
    "1": [
        [  # train
            [2477, 3638, 2409],
            [3605, 2873, 2405],
            [4939, 5186, 2398],
            [4538, 5148, 2398],
            [4618, 3225, 2388],
            [3223, 5206, 3550],
            [1953, 2102, 3577],
            [2395, 5004, 3584],
            [941, 3711, 708],
            [2030, 2164, 701],
            [3283, 3406, 1255],
            [1531, 2220, 1242],
            [2569, 6420, 2924],
            [2282, 8206, 2924],
            [4424, 5689, 2896],
            [3269, 3987, 2896],
            [2817, 6831, 4565],
            [3308, 3276, 4124],
            [4560, 6354, 4133],
            [4293, 2411, 2297],
            [5254, 5429, 2877],
            [3851, 2185, 2877],
            [2261, 8272, 2877],
            [3970, 3496, 2877],
            [1277, 2284, 2877],
        ],
        [  # val
            [5701, 3357, 4137],
            [3610, 2346, 4137],
            [2564, 4086, 2829],
            [1282, 2182, 2829],
            [3960, 2836, 1546],
            [2347, 4866, 1545],
            [1680, 4284, 806],
            [3223, 3294, 2514],
            [3564, 3847, 2516],
            [2620, 7237, 4610],
        ],
    ],
    "3": [
        [  # train
            [3007, 4546, 2697],
            [4466, 4467, 2697],
            [2670, 606, 2700],
            [1135, 751, 2700],
            [2354, 793, 1686],
            [327, 3241, 1686],
            [3961, 5206, 1686],
            [2852, 5223, 1686],
            [3574, 5145, 3261],
            [3566, 1245, 1399],
            [836, 2652, 1399],
            [2232, 6470, 1399],
            [2274, 3788, 2248],
            [2286, 1678, 2248],
            [4215, 1992, 2248],
            [2974, 2556, 2711],
            [2133, 2004, 2711],
            [3942, 5743, 2711],
            [2600, 5259, 485],
            [3090, 3676, 485],
            [1341, 5648, 2227],
            [1013, 4294, 2227],
            [3341, 3992, 2227],
            [1772, 5628, 3011],
            [1039, 5685, 1038],
            [2170, 3928, 1861],
            [3040, 3047, 1861],
            [2638, 1807, 676],
            [2153, 1863, 676],
            [2043, 2587, 676],
            [2558, 1950, 1965],
            [4310, 1665, 1202],
        ],
        [  # val
            [2639, 513, 2432],
            [4592, 4209, 2432],
            [4234, 4105, 3331],
            [2067, 4007, 1683],
            [4764, 2569, 1683],
            [2579, 2759, 583],
            [3244, 3921, 575],
            [2298, 1555, 575],
            [3081, 4587, 2971],
            [3402, 6232, 2697],
            [4426, 4334, 1038],
            [857, 2867, 1038],
            [1665, 5523, 2391],
            [2027, 4258, 1861],
            [2028, 4259, 1861],
            [3396, 2053, 1002],
            [2245, 1169, 1002],
            [1880, 2255, 695],
            [3184, 2487, 695],
            [3826, 1842, 696],
            [2843, 1240, 2198],
        ],
    ],
    "4": [
        [  # train
            [2835, 1768, 1906],
            [4284, 2059, 998],
            [4805, 3306, 998],
            [1961, 2160, 574],
            [4095, 3402, 574],
            [3097, 5111, 573],
            [2628, 3998, 2479],
            [2757, 7391, 1988],
            [332, 3555, 1988],
            [4227, 3671, 2608],
        ],
        [  # val
            [4781, 3581, 1861],
            [2982, 482, 1841],
            [2258, 6504, 1392],
            [2884, 2434, 1392],
            [1659, 4108, 1392],
            [3457, 5082, 1392],
            [2850, 5163, 3116],
            [2091, 3731, 2608],
            [3026, 6285, 2606],
            [678, 5501, 1388],
        ],
    ],
    "8613": [
        [],  # train
        [  # val
            [3058.365478515625, 4692.974609375, 1025.5],
            [649.9442749023438, 2539.990966796875, 1031.5],
            [3439.125732421875, 1684.6531982421875, 1031.5],
            [1713.416259765625, 2330.409912109375, 1591.4998779296875],
            [3269.08984375, 5614.2373046875, 1590.5001220703125],
            [3188.93701171875, 5547.95703125, 1590.5001220703125],
            [1201.314697265625, 5000.23828125, 1590.4998779296875],
            [2525.391845703125, 4362.392578125, 471.5],
            [3439.92431640625, 5827.6337890625, 1310.5],
            [2789.37060546875, 3125.13525390625, 2005.5],
        ],
    ],
}
