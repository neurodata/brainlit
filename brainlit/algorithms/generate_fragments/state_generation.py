import zarr
import numpy as np
import h5py
from joblib import Parallel, delayed
import os

import subprocess

class state_generation:
    def __init__(self, image_path, ilastik_program_path, ilastik_project_path, soma_coords=[], resolution=[], parallel=1):
        self.image_path = image_path
        self.ilastik_program_path = ilastik_program_path
        self.ilastik_project_path = ilastik_project_path
        self.soma_coords = soma_coords
        self.resolution = resolution
        self.parallel = parallel

    def predict_thread(self, corner1, corner2, data_bin):
        print(f"{corner1}, {corner2}, {data_bin}")
        image = zarr.open(self.image_path, mode='r')
        image_chunk = np.squeeze(image[corner1[0]:corner2[0], corner1[1]:corner2[1], corner1[2]:corner2[2]])
        fname = data_bin + "image_" + str(corner1[2]) + ".h5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("image_chunk", data=image_chunk)
        subprocess.run([self.ilastik_program_path, "--headless", f"--project={self.ilastik_project_path}", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    def predict(self, data_bin):
        image = zarr.open(self.image_path, mode='r')
        probabilities = zarr.zeros(np.squeeze(image.shape), chunks = image.chunks, dtype="float")
        chunk_size = [375, 375, 125] #image.chunks

        for x in np.arange(0, image.shape[0], chunk_size[0]):
            x2 = np.amin([x+chunk_size[0], image.shape[0]])
            for y in np.arange(0, image.shape[1], chunk_size[1]):
                y2 = np.amin([y+chunk_size[1], image.shape[1]])
                Parallel(n_jobs=self.parallel)(delayed(self.predict_thread)([x,y,z], [x2,y2,np.amin([z+chunk_size[2], image.shape[2]])], data_bin) for z in np.arange(0, image.shape[2], chunk_size[2]))
                
                for f in os.listdir(data_bin):
                    if "Probabilities" in f:
                        items = f.split("_")
                        z = int(items[1])
                        z2 = np.amin([z+chunk_size[2], image.shape[2]])
                        fname = os.path.join(data_bin, f)
                        f = h5py.File(fname, "r")
                        pred = f.get("exported_data")
                        pred = pred[:,:,:,0]

                        probabilities[x:x2,y:y2,z:z2] = pred
                    os.remove(os.path.join(data_bin, f))

