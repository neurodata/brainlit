import urllib
import json
import numpy as np
import os


def json_to_points(url, round=False):
    pattern = "json_url="
    idx = url.find(pattern) + len(pattern)

    json_url = url[idx:]

    data = urllib.request.urlopen(json_url)

    string = data.readlines()[0].decode("utf-8")

    js = json.loads(string)

    point_layers = {}

    for layer in js["layers"]:
        if layer["type"] == "annotation":
            points = []
            for point in layer["annotations"]:
                coord = point["point"]
                if round:
                    coord = [int(np.round(c)) for c in coord]
                points.append(coord)
            point_layers[layer["name"]] = points

    for key in point_layers.keys():
        print(f"{len(point_layers[key])} points in {key} layer")
    return point_layers


def find_sample_names(dir, dset = "val", add_dir = False):
    items = os.listdir(dir)

    items = [item for item in items if ".h5" in item]
    items = [item for item in items if "Probabilities" not in item]
    items = [item for item in items if "Labels" not in item]
    items = [item for item in items if dset in item]

    if add_dir:
        items = [dir + item for item in items]

    return items