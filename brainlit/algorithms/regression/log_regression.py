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


examine_failure = False
stats = False
append = "nei"
output_fail = "/cis/home/tathey/projects/mouselight/images/01_22_failure/"

kernels, names = image_process.getKernels(neighborhood=[12, 12, 4])

num_kernels = len(kernels)

# ****************Read data**********
print("Loading Training Data...")
df_iter = pd.read_csv(
    "../large_files/neighborhood_01_24_555.csv", header=None, index_col=0
)  # , chunksize=100000, iterator=True)

df_iter = [df_iter]

for iter_num, df in enumerate(df_iter, 1):
    #%%
    print("Iteration: " + str(iter_num))
    if iter_num >= 5:
        break
    vars = np.arange(125)  # [0,3,7,12,13,19,25,29,35,38]

    kernels = [kernels[v] for v in vars]

    dfn = df.to_numpy()
    X = dfn[:, 3:].astype(float)
    print(X.shape)
    X = X[:, vars]
    # names = list(names[v] for v in vars)
    # names.insert(0,'constant')
    y = dfn[:, 2].astype(int)
    locs = dfn[:, :2].astype(int)

    """
    #**************read manual data*****
    df = pd.read_csv('./mouselight_code/experiment_output/features_01_14_manual.csv',header=None,
        index_col=0)

    #%%

    dfn = df.to_numpy()
    X2 = dfn[:,4:].astype(float)
    X2 = X2[:,vars]
    print(X2.shape)
    X = np.concatenate((X,X2), axis=0)
    y2 = dfn[:,3].astype(int)
    y = np.concatenate((y,y2), axis=0)
    locs2 = -1* np.ones((X2.shape[0],2),dtype=int)
    locs = np.concatenate((locs,locs2), axis=0)
    """

    print("Preprocessing Data...")
    preprocesser = preprocessing.StandardScaler().fit(X)
    X = preprocesser.transform(X)

    if stats:
        X = add_constant(X)
    else:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()

    print(X.shape)
    print(np.sum(y))
    # *******************************************8
    print("Training and Evaluating...")
    kf = KFold(n_splits=5, shuffle=True)
    acc = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        locs_train, locs_test = locs[train_index, :], locs[test_index, :]

        if stats:
            # *************************statsmodels

            logr_model = Logit(y_train, X_train)

            logr = logr_model.fit_regularized(method="l1", max_iter=500, alpha=1.0)

            print(logr.summary(xname=names))

            y_test_predict = logr.predict(X_test)
            y_test_predict = np.rint(y_test_predict).astype(int)
            acc.append(accuracy_score(y_test, y_test_predict))
        else:
            print("split")
            logr = LogisticRegression(max_iter=2000).fit(X_train, y_train)

            viz = plot_roc_curve(
                logr,
                X_test,
                y_test,
                name="ROC fold {}".format(i),
                alpha=0.3,
                lw=1,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            y_test_predict = logr.predict(X_test)
            acc.append(accuracy_score(y_test, y_test_predict))

            if examine_failure:
                examine_failures(y_test, y_test_predict, locs_test)

    if not stats:
        ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.4f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        print("AUC:")
        print(mean_auc)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="ROC for Feature-based Classifier",
        )
        ax.legend(loc="lower right")
        plt.show()
    print(np.mean(acc))

# *******************************************8
def examine_failures(y_true, y_guess, locs):
    im_path = "/cis/net/io50/local/jacs/data/jacsstorage/samples/2018-08-01"
    tree = octree.octree(im_path)
    swc_dir_path = Path(
        "/cis/net/io50/local/jacs/data/jacsstorage/samples/2018-08-01/swcs/17_9_19_consensus/2018-08-01-consensus-swcs/"
    )
    files = list(swc_dir_path.glob("**/*.swc"))[1:]

    len_swcpts = locs.shape[0]

    wrong_idxs = np.argwhere(y_true != y_guess).squeeze()

    for wrong_idx in wrong_idxs:
        if np.random.rand() < 0.995:
            continue
        true_class = y_true[wrong_idx]
        loc = locs[wrong_idx, :]
        print(loc)
        # sample from manual data
        if loc[0] == -1:
            continue

        file = loc[0]
        pt = loc[1]

        df, _, _, _ = read_swc.read_swc_offset(files[file])
        img0, start = combine_swc_img.points2img(
            tree, df.iloc[pt : pt + 1].reset_index(), pad=[31, 31, 31]
        )
        img0 = preprocess.center(img0)

        if true_class == 1:
            print(1)
            points = combine_swc_img.points2voxel(
                tree, df.iloc[pt : pt + 1].reset_index(), start
            )
            voxels = points[["xvox", "yvox", "zvox"]].values
            ttl = "Incorrectly Labeled as Background"
        else:
            voxels = np.array([[15, 15, 15]])
            print(0)
            ttl = "Incorrectly Labeled as Axon"
        voxels = [voxels]

        f, a = visualize.plot_image_pts([img0], voxels, titles=[ttl])

        fname = (
            output_fail
            + str(file)
            + "_"
            + str(pt)
            + "_"
            + append
            + "_"
            + str(true_class)
            + ".png"
        )
        plt.savefig(fname)
        plt.close()
