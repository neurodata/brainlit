import numpy as np
import brainlit
from brainlit.utils import read_octree as octree, image_getters
from brainlit.utils import combine_swc_img
from brainlit.utils import read_swc
from brainlit.utils import swc2voxel
from brainlit.preprocessing import preprocess, image_process
from scipy import ndimage as ndi
from pathlib import Path
import pandas as pd
from itertools import product

kernels, _ = image_process.getKernels()

num_kernels = len(kernels)

# Data Formalities
im_path = "/cis/net/io50/local/jacs/data/jacsstorage/samples/2018-08-01"
tree = octree.octree(im_path)

output_feats = "../large_files/features_01_22_all.csv"

swc_dir_path = Path(
    "/cis/net/io50/local/jacs/data/jacsstorage/samples/2018-08-01/swcs/17_9_19_consensus/2018-08-01-consensus-swcs/"
)
files = list(swc_dir_path.glob("**/*.swc"))[1:]

save_feats = True

q_neuron = 0.0
q_point = 0.0


for j, file in enumerate(files):
    if np.random.uniform() < q_neuron:
        continue

    training = pd.DataFrame(columns=["File No.", "Point", "Label"])
    training_features = None
    df, _, _, _ = read_swc.read_swc_offset(file)

    num_pts = df.shape[0]

    print(j)
    print("Num points:" + str(num_pts))
    for i in range(0, num_pts, 1):
        if np.random.uniform() < q_point:
            continue

        # unprocessed
        img0, start = combine_swc_img.points2img(
            tree, df.iloc[i : i + 1].reset_index(), pad=[31, 31, 31]
        )

        # processed
        img0 = preprocess.center(img0)
        # img0 = ndi.gaussian_filter(img0,sigma=[3,3,0.9])
        points = combine_swc_img.points2voxel(
            tree, df.iloc[i : i + 1].reset_index(), start
        )
        voxels = points[["xvox", "yvox", "zvox"]].values
        center = voxels[0]

        training = training.append(
            {"File No.": j, "Point": i, "Label": 0}, ignore_index=True
        )
        features0 = np.zeros((1, num_kernels))
        training = training.append(
            {"File No.": j, "Point": i, "Label": 1}, ignore_index=True
        )
        features1 = np.zeros((1, num_kernels))

        for k, kernel in enumerate(kernels):
            s = np.array(kernel.shape)
            s = (s - 1) / 2

            start0 = [15 - int(s[i]) for i in range(len(s))]
            end0 = [15 + int(s[i]) + 1 for i in range(len(s))]

            start1 = [center[i] - int(s[i]) for i in range(len(s))]
            end1 = [center[i] + int(s[i]) + 1 for i in range(len(s))]

            features0[0, k] = np.dot(
                img0[
                    start0[0] : end0[0], start0[1] : end0[1], start0[2] : end0[2]
                ].flatten(),
                kernel.flatten(),
            )
            features1[0, k] = np.dot(
                img0[
                    start1[0] : end1[0], start1[1] : end1[1], start1[2] : end1[2]
                ].flatten(),
                kernel.flatten(),
            )

            # im_c = ndi.convolve(img0,kernel,mode='reflect')
            # features0[0,k] = im_c[15,15,15]
            # features1[0,k] = im_c[center[0],center[1],center[2]]

        if training_features is None:
            training_features = np.concatenate((features0, features1), axis=0)
        else:
            training_features = np.concatenate(
                (training_features, features0, features1), axis=0
            )
    features = pd.DataFrame(training_features)

    training = pd.concat((training, features), axis=1)

    if save_feats:
        if j == 0:
            flag = "w"
        else:
            flag = "a"
        with open(output_feats, flag) as f:
            training.to_csv(f, header=False)
