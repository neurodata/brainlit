#%%
from ...utils.swc import read_swc_offset
from .spline_fxns import curvature as curv, torsion as tors
from pathlib import Path
from .make_connections import GeometricGraph
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import interpolate
from scipy.integrate import odeint
from scipy.interpolate import splev
import pickle
from scipy.stats import pearsonr
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

if __name__ == "__main__":
    #%%
    plot = False
    log = False
    #%%
    swc_dir_path = Path("/Users/bijanvarjavand/Documents/spring20/ndd/brainlit/tests")
    files = list(swc_dir_path.glob("**/*.swc"))
    print(files)

    n_neurons = 200
    if plot:
        fig = plt.figure()
    # fig, axs = plt.subplots(n_neurons, 5, projection="3d")

    total_curvatures = []

    total_mean_curvs = []
    total_lens = []

    total_mean_tors = []
    print(len(files))
    for i, file in enumerate(files):
        print(i)
        print(file)

        if i >= n_neurons:
            break

        neuron = GeometricGraph()
        df, _, _, _ = read_swc_offset(file)
        neuron.fit_dataframe(df)

        spline_tree = neuron.fit_spline_tree_invariant()
        if plot:
            ax = fig.add_subplot(n_neurons, 5, i * 5 + 1, projection="3d")
            for node in spline_tree.nodes:
                spline = spline_tree.nodes[node]["spline"]
                u = spline[1]
                tck = spline[0]
                pts = splev(u, tck)
                ax.plot(pts[0], pts[1], pts[2], "red")
            ax.scatter(soma[0], soma[1], soma[2], "blue")
            if i == 0:
                ax.set_title("Neuron")

        curvatures = []
        lens = []
        n_splines = len(spline_tree.nodes)
        if plot:
            ax = fig.add_subplot(n_neurons, 5, i * 5 + 2)
        for n, node in enumerate(spline_tree.nodes):
            u0 = spline_tree.nodes[node]["starting_length"]
            spline = spline_tree.nodes[node]["spline"]
            u = spline[1]
            u = np.arange(u[0], u[-1], 1)
            tck = spline[0]

            # try:
            curvature = curv(u, tck)
            curvatures.append(curvature)
            ends = splev([spline[1][0], spline[1][-1]], tck)
            ends = np.stack(ends, axis=0)
            traversal = np.linalg.norm(ends[:, 0] - ends[:, 1])
            arclength = u[-1]
            lens.append(traversal)
            if plot:
                ax.plot(u + u0, curvature)
            # except:
            #    print(f"Undefined curvature for {n} ({n_splines} total)")

        if i == 0 and plot:
            ax.set_title("Curvature vs Length from Soma")

        total_curvatures.append(np.concatenate(curvatures))

        mean_curvs = []

        for ln, curvature in zip(lens, curvatures):
            mean_curvs.append(np.mean(curvature))

        total_mean_curvs.append(np.array(mean_curvs))
        total_lens.append(np.array(lens))

        if plot:
            ax = fig.add_subplot(n_neurons, 5, i * 5 + 3)
            ax.scatter(lens, mean_curvs)
            if i == 0:
                ax.set_title("Average Curvature vs Segment Length")
            ax = fig.add_subplot(n_neurons, 5, i * 5 + 4)

        torsions = []
        lens = []
        for n, node in enumerate(spline_tree.nodes):
            u0 = spline_tree.nodes[node]["starting_length"]
            spline = spline_tree.nodes[node]["spline"]
            u = spline[1]
            u = np.arange(u[0], u[-1], 1)
            tck = spline[0]

            # try:
            torsion = tors(u, tck)
            torsions.append(torsion)
            ends = splev([spline[1][0], spline[1][-1]], tck)
            ends = np.stack(ends, axis=0)
            traversal = np.linalg.norm(ends[:, 0] - ends[:, 1])
            arclength = u[-1]
            lens.append(traversal)
            if plot:
                ax.plot(u + u0, torsion)
            # except:
            #    print(f"Undefined torsion for {n}")

        if i == 0 and plot:
            ax.set_title("Torsion vs Length from Soma")
        mean_tors = []
        for ln, torsion in zip(lens, torsions):
            mean_tors.append(np.mean(torsion))

        total_mean_tors.append(np.array(mean_tors))
        if plot:
            ax = fig.add_subplot(n_neurons, 5, i * 5 + 5)
            ax.scatter(lens, mean_tors)

            if i == 0:
                ax.set_title("Average Torsion vs Segment Length")
    """
    axs[0,0].set_title('Neuron and Soma')
    axs[0,1].set_title('Curvature vs. Length from Soma')
    axs[0,2].set_title('Mean Curvature vs. Segment Length')
    axs[0,3].set_title('Curvature vs. Length from Soma')
    axs[0,4].set_title('Mean Curvature vs. Segment Length')
    """
    if plot:
        plt.show()

    if log:
        total_lens = np.log(np.concatenate(total_lens, axis=0))

        total_mean_curvs = np.log(np.concatenate(total_mean_curvs, axis=0))
        curv_idxs = np.isfinite(total_mean_curvs)
        total_mean_curvs = total_mean_curvs[curv_idxs]
        total_lens_curv = np.expand_dims(total_lens[curv_idxs], axis=1)

        # Create linear regression object
        regr = linear_model.LinearRegression()

        print(f"{len(total_lens_curv)} segments for curvature")

        # Train the model using the training sets
        print(pearsonr(total_lens_curv[:, 0], total_mean_curvs))
        regr.fit(total_lens_curv, total_mean_curvs)

        # Make predictions using the testing set
        total_mean_curvs_pred = regr.predict(total_lens_curv)

        # The coefficients
        print("Coefficients: \n", regr.coef_)
        # The mean squared error
        print(
            "Mean squared error: %.2f"
            % mean_squared_error(total_mean_curvs, total_mean_curvs_pred)
        )
        # The coefficient of determination: 1 is perfect prediction
        r2 = r2_score(total_mean_curvs, total_mean_curvs_pred)
        print("Coefficient of determination: %.2f" % r2)

        line_lbl = f"y={regr.coef_[0]:.2f}x+{regr.intercept_:.2f} with r^2={r2:.2f}"
        plt.scatter(
            total_lens_curv,
            total_mean_curvs,
            label="Segment Mean Curvature",
            alpha=0.01,
            s=4,
        )
        plt.plot(
            total_lens_curv,
            total_mean_curvs_pred,
            label=line_lbl,
            color="red",
            linewidth=3,
        )
        plt.xlabel("Log Segment Length (um)", fontsize="x-large")
        plt.xticks(fontsize=12)
        plt.ylabel("Log Mean Curvature", fontsize="x-large")
        plt.yticks(fontsize=12)
        plt.legend(loc="lower right", fontsize="large")
        plt.show()

        # Torsion

        total_mean_tors = np.log(np.abs(np.concatenate(total_mean_tors, axis=0)))
        tor_idxs = np.isfinite(total_mean_tors)
        total_mean_tors = total_mean_tors[tor_idxs]
        total_lens_tor = np.expand_dims(total_lens[tor_idxs], axis=1)

        # Create linear regression object
        regr = linear_model.LinearRegression()

        print(f"{len(total_lens_tor)} segments for curvature")
        # Train the model using the training sets
        print(pearsonr(total_lens_tor[:, 0], total_mean_tors))
        regr.fit(total_lens_tor, total_mean_tors)

        # Make predictions using the testing set
        total_mean_tors_pred = regr.predict(total_lens_tor)

        # The coefficients
        print("Coefficients: \n", regr.coef_)
        # The mean squared error
        print(
            "Mean squared error: %.2f"
            % mean_squared_error(total_mean_tors, total_mean_tors_pred)
        )
        # The coefficient of determination: 1 is perfect prediction
        r2 = r2_score(total_mean_tors, total_mean_tors_pred)
        print("Coefficient of determination: %.2f" % r2)

        line_lbl = f"y={regr.coef_[0]:.2f}x+{regr.intercept_:.2f} with r^2={r2:.2f}"

        plt.scatter(
            total_lens_tor,
            total_mean_tors,
            label="Segment Mean Torsion",
            alpha=0.01,
            s=4,
        )
        plt.plot(
            total_lens_tor,
            total_mean_tors_pred,
            label=line_lbl,
            color="red",
            linewidth=3,
        )
        plt.xlabel("Log Segment Length (um)", fontsize="x-large")
        plt.xticks(fontsize=12)
        plt.ylabel("Log Absolute Value Mean Torsion", fontsize="x-large")
        plt.yticks(fontsize=12)
        plt.legend(loc="lower right", fontsize="large")
        plt.show()
    else:

        def inv_power_law(x, c, k):
            y = c * (x ** k)
            return y

        total_lens = np.concatenate(total_lens, axis=0)
        total_mean_curvs = np.concatenate(total_mean_curvs, axis=0)
        total_mean_tors = np.abs(np.concatenate(total_mean_tors, axis=0))
        print(total_lens.shape)
        print(total_mean_curvs.shape)
        print(total_mean_tors.shape)

        popt = curve_fit(inv_power_law, total_lens, total_mean_curvs)
        print(popt)
        print(pearsonr(total_lens, total_mean_curvs))
        params = popt[0]
        c = params[0]
        k = params[1]

        min = np.amin(total_lens)
        max = np.amax(total_lens)
        ls = np.arange(min, max, (max - min) / 100)
        est = inv_power_law(ls, c, k)

        plt.scatter(
            total_lens, total_mean_curvs, label="Segment Mean Curvature", alpha=1, s=4
        )
        plt.plot(ls, est, "r", label="Estimate")
        plt.xlabel("Segment Length (um)", fontsize="x-large")
        plt.xticks(fontsize=12)
        plt.ylabel("Mean Curvature", fontsize="x-large")
        plt.yticks(fontsize=12)
        plt.legend(loc="lower right", fontsize="large")
        plt.show()

        popt = curve_fit(inv_power_law, total_lens, total_mean_tors)
        print(popt)
        print(pearsonr(total_lens, total_mean_tors))
        params = popt[0]
        c = params[0]
        k = params[1]
        est = inv_power_law(ls, c, k)

        plt.scatter(
            total_lens, total_mean_tors, label="Segment Mean Torsion", alpha=1, s=4
        )
        plt.plot(ls, est, "r", label="Estimate")
        plt.xlabel("Segment Length (um)", fontsize="x-large")
        plt.xticks(fontsize=12)
        plt.ylabel("Mean Absolute Value Torsion", fontsize="x-large")
        plt.yticks(fontsize=12)
        plt.legend(loc="lower right", fontsize="large")
        plt.show()
