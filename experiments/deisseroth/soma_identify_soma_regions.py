import numpy as np
from cloudvolume import CloudVolume
from tqdm import tqdm
import pickle

brain = "r3"

atlas_vol = CloudVolume("file:///mnt/data/Neuroglancer_Data/2021_12_02/8607/atlas_to_target/", parallel=1, mip=0, fill_missing=True)
somas = "/home/user/misc_tommy/somas_brain" + brain + ".txt"

file1 = open(somas, 'r')
lines = file1.readlines()

coords = []
for line in tqdm(lines):
    if line != '\n':
        elements = line.split(",")
        print(elements)
        coord = [elements[0][1:], elements[1], elements[2][:-1]]
        
        coord = [int(round(float(e.strip()))) for e in coord]
        coords.append(coord)

dict = {}
for coord in tqdm(coords):
    roi = atlas_vol[coord[0], coord[1], coord[2]]
    if roi not in dict.keys():
        dict[roi] = 1
    else:
        dict[roi] = dict[roi] + 1

print(dict)
with open('/home/user/misc_tommy/soma_counts_brain' + brain + '.pickle', 'wb') as handle:
    pickle.dump(dict, handle)