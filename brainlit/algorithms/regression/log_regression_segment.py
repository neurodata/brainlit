import numpy as np
import brainlit
from brainlit.utils import read_octree as octree
from brainlit.utils import combine_swc_img
from brainlit.utils import read_swc
from brainlit.plot import visualize
from brainlit.utils import swc2voxel
from brainlit.preprocessing import image_process
from brainlit.preprocessing import preprocess
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from itertools import product
from statsmodels.discrete.discrete_model import Logit
from statsmodels.api import add_constant
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn import preprocessing

stats = False
read_whole = True
append = "feat"

kernels, names = image_process.getKernels(neighborhood=None)  # [2,2,2])

num_kernels = len(kernels)

# ****************Read data**********
print("Loading Training Data...")
if read_whole:
    df = pd.read_csv("../large_files/features_01_22_all.csv", header=None, index_col=0)
else:
    df_iter = pd.read_csv(
        "../large_files/neighborhood_01_22_25259.csv",
        header=None,
        index_col=0,
        chunksize=100000,
        iterator=True,
    )
    for iter_num, df in enumerate(df_iter, 1):
        break


vars = np.arange(40)  # [0,3,7,12,13,19,25,29,35,38]

kernels = [kernels[v] for v in vars]

dfn = df.to_numpy()
X = dfn[:, 3:].astype(float)
print(X.shape)
X = X[:, vars]
# names = list(names[v] for v in vars)
# names.insert(0,'constant')
y = dfn[:, 2].astype(int)

print("Preprocessing Data...")
preprocesser = preprocessing.StandardScaler().fit(X)
X = preprocesser.transform(X)

# ***********************************************************
print("Training full model...")
if stats:
    logr_model = Logit(y, X)
    logr = logr_model.fit_regularized(method="l1", max_iter=100, alpha=1.0)
else:
    logr = LogisticRegression(max_iter=1000).fit(X, y)
    # print(logr.coef_.reshape((3,3,3)))

# Data formalities

im_path = "/cis/net/io50/local/jacs/data/jacsstorage/samples/2018-08-01"
tree = octree.octree(im_path)


swc_dir_path = Path(
    "/cis/net/io50/local/jacs/data/jacsstorage/samples/2018-08-01/swcs/17_9_19_consensus/2018-08-01-consensus-swcs/"
)
files = list(swc_dir_path.glob("**/*.swc"))[1:]

save_fig = True

q_neuron = 0.9
q_point = 0.99

r = 3

j_s = [21, 45]
i_s = [1123, 1101]
# 21_1123, 45_1101
print("Starting Segmentation...")
for j, file in enumerate(files):
    # if np.random.uniform() < q_neuron:
    if j not in j_s:
        continue

    df, _, _, _ = read_swc.read_swc_offset(file)

    num_pts = df.shape[0]

    count_snr2 = 0

    print(j)
    print("Num points:" + str(num_pts))
    for i in range(0, num_pts, 1):
        # if np.random.uniform() < q_point:
        if i not in i_s:
            continue
        print("new point")
        # unprocessed
        img0, start = combine_swc_img.points2img(
            tree, df.iloc[i : i + 1].reset_index(), pad=[31, 31, 31]
        )

        # processed
        img0 = preprocess.center(img0)
        shp = img0.shape

        X = np.zeros((img0.reshape((-1, 1)).shape[0], num_kernels))
        for k, kernel in enumerate(kernels):
            print(k)
            # kernel = np.flip(kernel)
            im_c = ndi.convolve(img0, kernel)
            X[:, k] = np.squeeze(im_c.reshape((-1, 1)))

        X = preprocesser.transform(X)

        if stats:
            X = add_constant(X)
            lp = logr.predict(X)
        else:
            lp = logr.predict_proba(X)
            lp = lp[:, 1]
        img0_pred = lp.reshape(shp)
        # post-process
        img_proc = img0_pred.copy()
        img_proc[img_proc > 0.95] = 1
        img_proc[img_proc <= 0.95] = 0

        img_proc = image_process.removeSmallCCs(img_proc, size=25)

        if save_fig:
            f, a = visualize.plot_image_mip(
                [img0, img0_pred, img_proc], titles=["Image", "Prob", "Threshold"]
            )
            fname = output_fig + str(j) + "_" + str(i) + "_" + append + ".png"
            plt.savefig(fname)
            plt.close()

# %%
