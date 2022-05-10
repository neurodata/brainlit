# functions for model training and performance evaluation

import numpy as np
from sklearn.metrics import roc_curve, auc, jaccard_score
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)


def train_loop(dataloader, model, loss_fn, optimizer):
    """Pytorch model training loop

    Arguments:
        train_dataloader: torch object from getting_torch_objects function in preprocess.py
        model: pytorch model, defined locally
        loss_fn: loss_fn class name, ex: BCELoss, Dice
        optimizer: name of optimizer, ex. Adam, SGD, etc.
    """
    for batch, (X_all, y_all) in enumerate(dataloader):

        loss_list = []

        for image in range(X_all.shape[1]):
            X = np.reshape(X_all[0][image], (1, 1, 66, 66, 20))
            y = np.reshape(y_all[0][image], (1, 1, 66, 66, 20))

            # Compute prediction and loss
            optimizer.zero_grad()
            pred = model(X)
            pred = torch.squeeze(pred, 3).clone()
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            loss, current = loss.item(), batch * len(X)
            loss_list.append(loss)


def test_loop(dataloader, model, loss_fn):
    """Pytorch model testing loop

    Arguments:
        test_dataloader: torch object from getting_torch_objects function in preprocess.py
        model: pytorch model, defined locally
        loss_fn: loss_fn class name, ex: BCELoss, Dice

    Returns:
        x_list: list, true images
        y_pred: nested list, model predictions for each image at each epoch
        y_list: nested list, true masks for each image at each epoch
        avg_loss: list, average loss at each epoch
    """
    for batch, (X_all, y_all) in enumerate(dataloader):

        loss_list = []
        y_pred = []
        y_list = []
        x_list = []

        with torch.no_grad():
            for image in range(X_all.shape[1]):

                X = np.reshape(X_all[0][image], (1, 1, 330, 330, 100))
                y = np.reshape(y_all[0][image], (1, 1, 330, 330, 100))
                pred = model(X)
                pred = torch.squeeze(pred, 3)

                x_list.append(X)
                y_list.append(y)
                y_pred.append(pred)

                loss_list.append(loss_fn(pred, y).item())

        avg_loss = np.average(loss_list)
        print("Avg test loss:", avg_loss)

    return x_list, y_pred, y_list, avg_loss


# Dice loss class
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def get_metrics(pred_list, y_list):
    """Getting accuracy, precision, and recall at each epoch

    Arguments:
        pred_list: list of predictions for every image at every epoch, output of testing loop
        y_list: list of true y masks, output of testing loop

    Returns:
        acc_list: list of average accuracy for each epoch
        precision_list: list of average precision for each epoch
        recall_list: list of average recall for each epoch
        percent_nonzero: list of percent of nonzero predictions at each epoch
    """
    acc_list = []
    precision_list = []
    recall_list = []
    percent_nonzero = []

    for i in range(len(pred_list)):
        acc_list_t = []
        precision_list_t = []
        recall_list_t = []
        percent_nonzero_t = []

        for j in range(len(pred_list[0])):
            pred = pred_list[i][j].clone().numpy()[:, 0].round().astype(int).flatten()
            target = y_list[i][j][:, 0].clone().numpy().astype(int).flatten()

            acc = accuracy_score(target, pred) * 100
            acc_list_t.append(acc)

            pr = precision_score(target, pred) * 100
            precision_list_t.append(pr)

            rc = recall_score(target, pred) * 100
            recall_list_t.append(rc)

            nz = (np.count_nonzero(pred) / len(target)) * 100
            percent_nonzero_t.append(nz)

        mean_acc = np.mean(acc_list_t)
        mean_pr = np.mean(precision_list_t)
        mean_rc = np.mean(recall_list_t)
        mean_nz = np.mean(percent_nonzero_t)

        acc_list.append(mean_acc)
        precision_list.append(mean_pr)
        recall_list.append(mean_rc)
        percent_nonzero.append(mean_nz)

    return acc_list, precision_list, recall_list, percent_nonzero


def quick_stats(stat, epoch, acc_list, precision_list, recall_list, percent_nonzero):
    """Printing quick test stats at specified epoch

    Arguments:
        stat: str, "all" if you want to print all metrics (accuracy, precision, reacll, % nonzero)
        acc_list: list of average accuracy for each epoch, from get_metrics function
        precision_list: list of average precision for each epoch, from get_metrics function
        recall_list: list of average recall for each epoch, from get_metrics function
        percent_nonzero: list of percent of nonzero predictions at each epoch, from get_metrics function

    Returns:
        Printed metrics for specified epoch
    """
    if stat == "accuracy":
        print("Accuracy at epoch " + str(epoch) + " is " + str(acc_list[epoch - 1]))
    if stat == "all":
        print("Accuracy at epoch " + str(epoch) + " is " + str(acc_list[epoch - 1]))
        print(
            "Precision at epoch " + str(epoch) + " is " + str(precision_list[epoch - 1])
        )
        print("Recall at epoch " + str(epoch) + " is " + str(recall_list[epoch - 1]))
        print(
            "Percent nonzero at epoch "
            + str(epoch)
            + " is "
            + str(percent_nonzero[epoch - 1])
        )


def plot_metrics_over_epoch(
    loss_list, acc_list, precision_list, recall_list, percent_nonzero
):
    """Plotting all metrics over epoch

    Arguments:
        loss_list: list of test loss over epoch
        acc_list: list of average accuracy for each epoch, from get_metrics function
        precision_list: list of average precision for each epoch, from get_metrics function
        recall_list: list of average recall for each epoch, from get_metrics function
        percent_nonzero: list of percent of nonzero predictions at each epoch, from get_metrics function

    Returns:
        Plotted figures for accuracy, precision, recall, % nonzero, and loss over epoch
    """
    plt.figure()
    plt.title("Test loss over epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Test loss")
    plt.plot(loss_list)

    plt.figure()
    plt.title("Accuracy over epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Avg accuracy (%)")
    plt.plot(acc_list)

    plt.figure()
    plt.title("Precision over epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Avg precision (%)")
    plt.plot(precision_list)

    plt.figure()
    plt.title("Recall over epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Avg recall (%)")
    plt.plot(recall_list)

    plt.figure()
    plt.title("Percent_nonzero over epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Nonzeros (%)")
    plt.plot(percent_nonzero)


def plot_pr_histograms(pred_list, y_list):
    """Plotting histograms for precision and recall at final epoch

    Arguments:
        pred_list: list of predictions for all images at last epoch
        y_list: lost of true y masks for all images at last epoch

    Returns:
        Precision and recall plots for all images at last epoch
    """
    i = len(pred_list) - 1
    precision_list_t = []
    recall_list_t = []

    for j in tqdm(range(len(pred_list[0]))):
        pred = pred_list[i][j].clone().numpy()[:, 0].round().astype(int).flatten()
        target = y_list[i][j][:, 0].clone().numpy().astype(int).flatten()

        pr = precision_score(target, pred) * 100
        precision_list_t.append(pr)

        rc = recall_score(target, pred) * 100
        recall_list_t.append(rc)

    # Precision histogram on last epoch
    plt.figure()
    plt.title("Precision histogram for individual 11 images on last epoch")
    plt.ylabel("Individual Precision")
    plt.hist(precision_list_t, bins=20)

    # Recall histogram on last epoch
    plt.figure()
    plt.title("Recall histogram for individual 11 images on last epoch")
    plt.ylabel("Individual Recall")
    plt.hist(recall_list_t, bins=20)


def plot_with_napari(x_list, pred_list, y_list):
    """Plotting all test images at an epoch in napari

    Arguments:
        x_list: list of all x images from testing loop
        pred_list: list of all testing predictions at an epoch
        y_list: list of true ground truth masks at that same epoch

    Returns:
        Visualizations of napari image, ground truth mask, and thresholded prediction mask
    """
    for i in range(len(y_list[len(y_list) - 1])):
        x = x_list[i].clone()[:, 0].numpy()
        pred = pred_list[len(pred_list) - 1][i].clone()[:, 0].numpy()
        y = y_list[len(y_list) - 1][i].clone()[:, 0].numpy()

        fpr, tpr, thresholds = roc_curve(y.flatten(), pred.flatten())
        optimal_thresh = thresholds[np.argmax(tpr - fpr)]
        # print("Optimal Threshold for image " + str(i) + ": ", optimal_thresh)

        pred_thresh = pred

        for i in range(1):
            for a in range(330):
                for b in range(330):
                    for c in range(100):
                        if pred[i][a][b][c] > optimal_thresh:
                            pred_thresh[i][a][b][c] = 1
                        else:
                            pred_thresh[i][a][b][c] = 0

        import napari

        with napari.gui_qt():
            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(x[0])
            viewer.add_labels(y[0].astype(int))
            viewer.add_labels(pred_thresh[0].astype(int), num_colors=2)
