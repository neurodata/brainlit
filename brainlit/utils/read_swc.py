#%%
import numpy as np
from pathlib import Path
import pandas as pd
import re
from random import shuffle


def read_swc(path):
    """Read a single swc file

    Arguments:
        path {string} -- path to file

    Returns:
        df {pandas dataframe} -- indices, coordinates, and parents of each node
        offset {list of floats} -- offset value of fragment
        color {list of ints} -- color
        cc {int} -- cc value, from file name
        branch {int} -- branch number, from file name
    """

    # check input
    file = open(path, "r")
    in_header = True
    offset_found = False
    header_length = -1
    offset = np.nan
    color = np.nan
    cc = np.nan
    branch = np.nan
    while in_header:
        line = file.readline().split()
        if "OFFSET" in line:
            offset_found = True
            idx = line.index("OFFSET") + 1
            offset = [float(line[i]) for i in np.arange(idx, idx + 3)]
        elif "COLOR" in line:
            idx = line.index("COLOR") + 1
            line = line[idx]
            line = line.split(",")
            color = [float(line[i]) for i in np.arange(len(line))]
        elif "NAME" in line:
            idx = line.index("NAME") + 1
            name = line[idx]
            name = re.split("_|-|\.", name)
            idx = name.index("cc") + 1
            cc = int(name[idx])
            idx = name.index("branch") + 1
            branch = int(name[idx])
        elif line[0] != "#":
            in_header = False
        header_length += 1

    if not offset_found:
        raise IOError("No offset information found in: " + path)
    # read coordinates
    df = pd.read_table(
        path,
        names=["sample", "structure", "x", "y", "z", "r", "parent"],
        skiprows=header_length,
        delim_whitespace=True,
    )
    return df, offset, color, cc, branch


def read_swc_offset(path):
    df, offset, color, cc, branch = read_swc(path)
    df["x"] = df["x"] + offset[0]
    df["y"] = df["y"] + offset[1]
    df["z"] = df["z"] + offset[2]

    return df, color, cc, branch


def append_df(cumulative, new):
    """Append the dataframe of one fragment to the dataframe of one or more other fragments

    Arguments:
        cumulative {pandas dataframe} -- other fragments
        new {pandas dataframe} -- fragment to be appended


    Returns:
        datafranme -- appended result
    """
    # check cumulative df
    samples = cumulative["sample"].values
    unq = np.unique(samples)
    if len(unq) != cumulative.shape[0]:
        msg = (
            "cumulative df has "
            + str(cumulative.shape[0])
            + " rows but only "
            + str(len(unq))
            + " unique sample labels."
        )
        raise ValueError(msg)
    mx = np.amax(unq)
    sample_new = np.arange(mx + 1, mx + 1 + new.shape[0])
    sample_new = np.append(sample_new, -1)
    sample_old = new["sample"].values
    sample_old = np.append(sample_old, -1)
    parent_old = new["parent"].values

    idxs = [
        np.argwhere(sample_old == parent_old[i])[0, 0]
        for i in np.arange(len(parent_old))
    ]
    parent_new = [sample_new[idx] for idx in idxs]

    new["sample"] = sample_new[:-1]
    new["parent"] = parent_new
    cumulative = cumulative.append(new)
    return cumulative.reset_index(drop=True)


def read_swc_dir(path, nFiles=1):
    """Read all swc files in a directory and create a large dataframe of all the results

    Arguments:
        path {string} -- path to directory of swc files

    Keyword Arguments:
        nFiles {int} -- number of files to process in the directory (default: {1})

    Returns:
        pandas dataframe -- dataframe of all the fragment data
    """
    files = list(Path(path).glob("**/*.swc"))[0:nFiles]

    for i, file in enumerate(files):
        df, offset, color, cc, branch = read_swc(file)
        df["x"] = df["x"] + offset[0]
        df["y"] = df["y"] + offset[1]
        df["z"] = df["z"] + offset[2]

        df["color0"] = color[0]
        df["color1"] = color[1]
        df["color2"] = color[2]

        df["cc"] = cc
        df["branch"] = branch

        if i == 0:
            df_cumulative = df
        else:
            df_cumulative = append_df(df_cumulative, df)

    return df_cumulative


def bbox_vox(df):
    """Returns the coordinate of a box center and its dimensions that contain fragments

    Arguments:
        df {pandas dataframe} -- dataframe of points as outputted from read_swc_offset

    Returns:
        start {3 list} -- starting spatial coordinate
        end {3 list} -- ending spatial coordinate
    """
    min_x, max_x = np.min(df.x), np.max(df.x)
    min_y, max_y = np.min(df.y), np.max(df.y)
    min_z, max_z = np.min(df.z), np.max(df.z)

    start = [min_x, min_y, min_z]
    end = [max_x, max_y, max_z]
    return start, end
