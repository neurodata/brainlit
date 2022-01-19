import numpy as np
from cloudvolume import CloudVolume
from tqdm import tqdm
import pickle

atlas_vol = CloudVolume("file:///mnt/data/Neuroglancer_Data/2021_10_06/8557/atlas_to_target/", parallel=1, mip=0, fill_missing=True)
somas = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/soma_detection/misc_results/somas_brainr1.txt"

file1 = open(somas, 'r')
lines = file1.readlines()

coords = []
for line in tqdm(lines):
    elements = line.split(",")
    coord = [elements[0][1:], elements[1], elements[2][:-1]]
    coord = [int(round(float(e))) for e in coord]
    coords.append(coord)

dict = {}
for coord in tqdm(coords):
    roi = atlas_vol[coord[0], coord[1], coord[2]]
    if roi not in dict.keys():
        dict[roi] = 1
    else:
        dict[roi] = dict[roi] + 1

print(dict)
with open('soma_counts.pickle', 'wb') as handle:
    pickle.dump(dict, handle)