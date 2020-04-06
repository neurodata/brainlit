import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import plot_roc_curve
from sklearn import datasets, metrics, model_selection

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import L1L2
from keras.utils import plot_model

from sklearn.preprocessing import LabelEncoder
from scipy import stats
import seaborn as sns


def MLP_LR_NN(X_train, y_train, n_features):
    """
    Keras model for nonlinear feature activation regression.
    Running this method defines and trains the model.

    Parameters
    ----------
    X_train : np.array
        The training data points
    y_train : np.array
        The training data labels
    n_features: int
        Number of input features (X_train)

    Returns
    -------
    model : Model object
        The model after it has been trained
    history : list
        Contains accuracy and score over epochs.
        Used to compare results with validation.
    """
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    model = Sequential()

    model.add(
        Dense(
            n_features,
            activation="relu",
            kernel_regularizer=L1L2(l1=0.0, l2=0.1),
            input_dim=len(X_train[0]),
        )
    )
    model.add(
        Dense(
            1,  # output dim is 2, one score per each class
            activation="sigmoid",
            kernel_regularizer=L1L2(l1=0.0, l2=0.1),
            input_dim=20,
        )
    )  # input dimension = number of features your data has
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, encoded_Y, epochs=70, batch_size=20, verbose=0)
    return model, history


def run_classifiers(
    X_sel,
    y_sel,
    X_test,
    y_test,
    classifiers,
    names,
    filename="Non_linear_classification.csv",
    num_runs=2,
):
    """
    Partially complete.
    Trains a list of classifiers and evaluates them.

    Paramters
    ---------
    X_sel : np.array
        training data points
    y_sel : np.array
        training data labels
    X_test : np.array
        testing data points
    y_test : np.array
        testing data labels
    classifiers : list
        a list of classifier objects
    names : dict
        dictionary of names for each classifier along with color of line
    filename : str, optional (default = "Non_linear_classification.csv")
        the filename to save results to, as a csv
    num_runs : int, optional (default = 2)
        the number of runs to average across

    Returns
    -------
    f : file
        the csv file containing names, accuracies, train and test time
    """
    size = len(y_sel)
    num_features = X_sel.shape[1] - 1

    f = open(filename, "w+")
    f.write("classifier,n,Accuracy,trainTime,testTime,iterate\n")
    f.flush()

    ns = np.logspace(1, np.log10(size), base=10, num=4).astype(int)
    runList = [(clf) for clf in zip(classifiers, [key for key in names])]
    for n in tqdm(ns):
        for iteration in tqdm(range(num_runs)):
            X_train = X_sel[:n]
            y_train = np.array(y_sel[:n]).ravel()

            for clf in tqdm(runList):
                if clf[1] == "MLP-relu-LR":  # check if from Keras
                    trainStartTime = time.time()
                    cls, his = MLP_LR_NN(X_train, y_train, num_features)
                    trainEndTime = time.time()
                    trainTime = trainEndTime - trainStartTime
                    encoder = LabelEncoder()
                    encoder.fit(y_test)
                    en_y_test = encoder.transform(y_test)

                    testStartTime = time.time()
                    score = cls.evaluate(X_test, en_y_test, batch_size=20)
                    testEndTime = time.time()
                    testTime = testEndTime - testStartTime
                    acc = score[1]
                else:
                    # training
                    trainStartTime = time.time()
                    clf[0].fit(X_train, y_train)
                    trainEndTime = time.time()
                    trainTime = trainEndTime - trainStartTime
                    # prediction
                    testStartTime = time.time()
                    out = clf[0].predict(X_test)
                    testEndTime = time.time()
                    testTime = testEndTime - testStartTime
                    # accuracy
                    acc = accuracy_score(y_test, out)
                # writing to file
                f.write(
                    f"{clf[1]}, {n}, {acc:2.9f}, {trainTime:2.9f}, {testTime:2.9f}, {iteration}\n"
                )
                f.flush()
    f.close()
    return filename


def plot_data(filepath, names, plotWhat, y_label, title):
    """
    Plots data stored in the file generated by run_classifiers.

    Parameters
    ----------
    filepath : string
        Path to data file.
    names : dict
        dictionary of names for each classifier along with color of line
    plotWhat : string
        Parameter to plot.
    y_label : string
        The y axis label.
    title : string
        The title of the plot.

    Returns
    -------
    fig : Matplotlib figure object
    ax : Matplotlib axis object
    """
    dat = pd.read_csv(filepath)
    d1 = pd.DataFrame(columns=["classifier", "n", plotWhat, "color"])

    k = 0
    for ni in np.unique(dat["n"]):
        for cl in np.unique(dat["classifier"]):

            tmp = dat[np.logical_and(dat["classifier"] == cl, dat["n"] == ni)][
                ["n", plotWhat]
            ]
            se = stats.sem(tmp[plotWhat].astype(float))
            list(tmp.mean())
            d1.loc[k] = [cl] + list(tmp.mean()) + [names[cl]]
            k += 1

    sns.set(style="darkgrid", rc={"figure.figsize": [12, 8], "figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(8, 6))

    for key in names.keys():
        grp = d1[d1["classifier"] == key]
        ax = grp.plot(
            ax=ax, kind="line", x="n", y=plotWhat, label=key, c=names[key], alpha=0.65
        )
        # ax.set_xscale('log')

    plt.legend(loc="upper left", title="Algorithm")
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Number of Training Samples")
    fig.savefig("classification_accuracy_plot.png")
    plt.close()
    return fig, ax
