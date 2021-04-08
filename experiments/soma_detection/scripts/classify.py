import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
from pathlib import Path
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from scipy import ndimage as ndi
from tqdm import tqdm
from brainlit.utils.session import NeuroglancerSession

from skimage import (
    color, feature, filters, measure, morphology, segmentation, exposure, util
)

mip = 1
radius = 150
neigh_n = 6

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")
volumes_dir = os.path.join(data_dir, "volumes", str(radius))
somas_dir = os.path.join(data_dir, "somas")
fig_dir = os.path.join(data_dir, "figures")
# print(f"Volumes dir {volumes_dir}")

brains = [1]

s3 = boto3.resource("s3")
bucket = s3.Bucket("open-neurodata")

def contains_somas(img):
    proj = np.amax(img, axis=2)
    
    out = img.copy()
    
    t = filters.threshold_otsu(out)        
    out = out > 1.25*t
    
    clean_selem = morphology.octahedron(2)
    cclean_selem = morphology.octahedron(1)
    out = morphology.erosion(out, clean_selem)
    out = morphology.erosion(out, cclean_selem)
    
    out, labels = morphology.label(out, background=0, return_num=True)
    for label in np.arange(1, labels+1):
        A = np.sum(out.flatten() == label)
        if A < 50:
            out[out == label] = 0
    
    labels, m = morphology.label(out, background=0, return_num=True)
    label = 0 if (m==0 or m >= 12) else 1
    
    return proj, label, out

for brain in brains:
    brain_name = "brain%d" % brain

    brain_prefix = f"brainlit/{brain_name}"
    segments_prefix = f"brainlit/{brain_name}_segments"
    somas_prefix = f"brainlit/{brain_name}_somas"
    skeletons_prefix = f"{segments_prefix}/skeletons"

    brain_url = f"s3://open-neurodata/{brain_prefix}"
    segments_url = f"s3://open-neurodata/{segments_prefix}"
    
    ngl_sess = NeuroglancerSession(mip=1, url=brain_url, url_segments=segments_url)

    running_correct = 0
    running_seg_count = 0
    running_neigh_count = 0
    positive_neigh_count = 0
    negative_neigh_count = 0
    failed = []
    for i, seg_obj in tqdm(enumerate(bucket.objects.filter(Prefix=skeletons_prefix))):
        seg_id = os.path.basename(seg_obj.key)
        if seg_id != "info":
            
            volume_filepath = os.path.join(volumes_dir, brain_name, f"{seg_id}.npy")
            mask_filepath = os.path.join(volumes_dir, brain_name, f"{seg_id}_mask.npy")
            d = np.load(volume_filepath, allow_pickle=True).item()
            img = d["volume"]
            bbox = d["bbox"]
            
            try:
                proj, label, somas = contains_somas(img)
                out, labels = morphology.label(somas, background=0, return_num=True)
            except ValueError:
                failed.append(seg_id)
            else:
                for m in range(1, labels+1):
                    c = np.where(out == m)
                    
                    min_pt, max_pt = np.amin(c, axis=1), np.amax(c, axis=1)
                    
                    soma_bbox = bbox.copy()
                    soma_bbox[:3] += min_pt
                    soma_bbox[3:] += max_pt
                    soma_bbox_nm = np.multiply(soma_bbox, np.tile(ngl_sess.cv.scales[mip]["resolution"], 2))
                    
                    soma_filename = f"{soma_bbox_nm[0]}-{soma_bbox_nm[1]}_{soma_bbox_nm[2]}-{soma_bbox_nm[3]}_{soma_bbox_nm[4]}-{soma_bbox_nm[5]}"
                    
                    soma_s3key = f"{somas_prefix}/{soma_filename}"
                    soma_filepath = os.path.join(somas_dir, brain_name, f"{soma_filename}.npy")
                    
                    np.save(soma_filepath, soma_bbox_nm, allow_pickle=True)
                    bucket.upload_file(soma_filepath, soma_s3key)
                np.save(mask_filepath, somas)
                # Update stats
                running_seg_count += 1
                if label == 1:
                    running_correct += 1
                # Save figure
                fig = plt.figure(figsize=(12, 4))
                axes = fig.subplots(1, 2)
                ax = axes[0]
                ax.imshow(proj)
                ax.set_title(r"MIP")
                ax = axes[1]
                ax.imshow(np.amax(somas, axis=2))
                ax.set_title(r"label = $%d$" % label)
                
                fig_path = os.path.join(fig_dir, brain_name, "TP" if label == 1 else "FN", f"{seg_id}.eps")
                plt.savefig(fig_path)
                # plt.show()
                plt.close()
            
            for neigh_id in np.arange(neigh_n):
                neighvolume_filepath = os.path.join(volumes_dir, brain_name, f"{seg_id}_neigh{neigh_id}.npy")
                mask_filepath = os.path.join(volumes_dir, brain_name, f"{seg_id}_neigh{neigh_id}_mask.npy")

                neigh_d = np.load(neighvolume_filepath, allow_pickle=True).item()
                neigh_img = neigh_d["volume"]
                neigh_bbox = neigh_d["bbox"]
                
                try:
                    proj, label, somas = contains_somas(neigh_img)
                    out, labels = morphology.label(somas, background=0, return_num=True)
                except ValueError:
                    failed.append(seg_id)
                else:
                    for m in range(1, labels+1):
                        c = np.where(out == m)
                        
                        min_pt, max_pt = np.amin(c, axis=1), np.amax(c, axis=1)
                        
                        soma_bbox = neigh_bbox.copy()
                        soma_bbox[:3] += min_pt
                        soma_bbox[3:] += max_pt
                        soma_bbox_nm = np.multiply(soma_bbox, np.tile(ngl_sess.cv.scales[mip]["resolution"], 2))
                        
                        soma_filename = f"{soma_bbox_nm[0]}-{soma_bbox_nm[1]}_{soma_bbox_nm[2]}-{soma_bbox_nm[3]}_{soma_bbox_nm[4]}-{soma_bbox_nm[5]}"
                    
                        soma_s3key = f"{somas_prefix}/{soma_filename}"
                        soma_filepath = os.path.join(somas_dir, brain_name, f"{soma_filename}.npy")
                    
                        np.save(soma_filepath, soma_bbox_nm, allow_pickle=True)
                        bucket.upload_file(soma_filepath, soma_s3key)
                    np.save(mask_filepath, somas)
                    # Update stats
                    running_neigh_count += 1
                    if label == 1:
                        positive_neigh_count += 1
                    else:
                        negative_neigh_count += 1
                    # Save figure
                    fig = plt.figure(figsize=(12, 4))
                    axes = fig.subplots(1, 2)
                    ax = axes[0]
                    ax.imshow(proj)
                    ax.set_title(r"MIP")
                    ax = axes[1]
                    ax.imshow(np.amax(somas, axis=2))
                    ax.set_title(r"label = $%d$" % label)
                
                    fig_path = os.path.join(fig_dir, brain_name, "neighbors", str(label), f"{seg_id}_neigh{neigh_id}.eps")
                    plt.savefig(fig_path)
                    # plt.show()
                    plt.close()
                
    TPR = running_correct / running_seg_count
    print(f"{brain_name}: TPR = {TPR}")