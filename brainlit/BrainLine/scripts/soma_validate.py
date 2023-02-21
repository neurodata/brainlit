from brainlit.preprocessing import removeSmallCCs
from brainlit.BrainLine.data.soma_data import brain2paths, brain2centers
from brainlit.BrainLine.util import (
    json_to_points,
    find_atlas_level_label,
    fold,
    setup_atlas_graph,
    get_atlas_level_nodes,
)
from brainlit.BrainLine.apply_ilastik import ApplyIlastik
from brainlit.BrainLine.parse_ara import *
import xml.etree.ElementTree as ET
from cloudreg.scripts.transform_points import NGLink
from brainlit.BrainLine.imports import *
from brainlit.BrainLine.util import find_sample_names

'''
Inputs
'''
data_dir = str(Path.cwd().parents[0])  # path to directory where training/validation data should be stored
brain = "test"
antibody_layer = "antibody"
background_layer = "background"
endogenous_layer = "endogenous"

dataset_to_save = "val"  # train or val

ilastik_path = "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh"
ilastik_project = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/matt_soma_rabies_pix_3ch.ilp"

'''
Setup
'''
brainlit_path = Path.cwd()
print(f"Path to brainlit: {brainlit_path}")
base_dir = data_dir + f"/brain{brain}/{dataset_to_save}/"

if brain not in brain2paths.keys():
    raise ValueError(f"brain {brain} not an entry in brain2paths in axon_data.py file")

if f"{dataset_to_save}_info" not in brain2paths[
    brain
].keys() or dataset_to_save not in ["train", "val"]:
    raise ValueError(f"{dataset_to_save}_info not in brain2paths[{brain}].keys()")

if dataset_to_save == "val":
    url = brain2paths[brain]["val_info"]["url"]
    l_dict = json_to_points(url)
    soma_centers = l_dict[brain2paths[brain]["val_info"]["somas_layer"]]
    nonsoma_centers = l_dict[brain2paths[brain]["val_info"]["nonsomas_layer"]]
elif dataset_to_save == "train":
    url = brain2paths[brain]["train_info"]["url"]
    l_dict = json_to_points(url)
    soma_centers = l_dict[brain2paths[brain]["train_info"]["somas_layer"]]
    nonsoma_centers = l_dict[brain2paths[brain]["train_info"]["nonsomas_layer"]]
print(f"{len(soma_centers)} soma centers")
print(f"{len(nonsoma_centers)} nonsoma centers")

mip = 0
if "base" in brain2paths[brain].keys():
    base_dir_s3 = brain2paths[brain]["base"]
    dir = base_dir_s3 + antibody_layer
    vol_fg = CloudVolume(dir, parallel=1, mip=mip, fill_missing=True)
    print(f"fg shape: {vol_fg.shape} at {vol_fg.resolution}")

    dir = base_dir_s3 + background_layer
    vol_bg = CloudVolume(dir, parallel=1, mip=mip, fill_missing=True)
    print(f"bg shape: {vol_bg.shape} at {vol_bg.resolution}")

    dir = base_dir_s3 + endogenous_layer
    vol_endo = CloudVolume(dir, parallel=1, mip=mip, fill_missing=True)
    print(f"endo shape: {vol_endo.shape} at {vol_endo.resolution}")

'''
Download data
'''
isExist = os.path.exists(base_dir)
if not isExist:
    print(f"Creating directory: {base_dir}")
    os.makedirs(base_dir)
else:
    print(f"Downloaded data will be stored in {base_dir}")

for type, centers in zip(["pos", "neg"], [soma_centers, nonsoma_centers]):
    for i, center in enumerate(tqdm(centers, desc="Saving positive samples")):
        image_fg = vol_fg[
            center[0] - 24 : center[0] + 25,
            center[1] - 24 : center[1] + 25,
            center[2] - 24 : center[2] + 25,
        ]
        image_fg = image_fg[:, :, :, 0]
        image_bg = vol_bg[
            center[0] - 24 : center[0] + 25,
            center[1] - 24 : center[1] + 25,
            center[2] - 24 : center[2] + 25,
        ]
        image_bg = image_bg[:, :, :, 0]
        image_endo = vol_endo[
            center[0] - 24 : center[0] + 25,
            center[1] - 24 : center[1] + 25,
            center[2] - 24 : center[2] + 25,
        ]
        image_endo = image_endo[:, :, :, 0]

        image = np.squeeze(np.stack([image_fg, image_bg, image_endo], axis=0))

        fname = (
            base_dir + f"{int(center[0])}_{int(center[1])}_{int(center[2])}_{type}.h5"
        )
        with h5py.File(fname, "w") as f:
            dset = f.create_dataset("image_3channel", data=image)

doubles = []

'''
Apply ilastik
'''
brains = [
    'test'
]  # sample IDs to be processed
applyilastik = ApplyIlastik(ilastk_path = ilastik_path, project_path = ilastik_project, brains_path = f"{data_dir}/", brains = brains)
applyilastik.process_somas()

'''
Check results
'''
recalls = []
precisions = []

test_files = find_sample_names(base_dir, dset="", add_dir=True)
test_files = [file.split(".")[0] + "_Probabilities.h5" for file in test_files]
print(test_files)

size_thresh = 500

thresholds = list(np.arange(0.0, 1.0, 0.02))

for threshold in thresholds:
    tot_pos = 0
    tot_neg = 0
    true_pos = 0
    false_pos = 0
    for filename in tqdm(test_files, disable=True):
        if filename.split("/")[-1] in doubles:
            newpos = 2
        else:
            newpos = 1

        f = h5py.File(filename, "r")
        pred = f.get("exported_data")
        pred = pred[0, :, :, :]
        mask = pred > threshold
        labels = measure.label(mask)
        props = measure.regionprops(labels)

        if "pos" in filename:
            num_detected = 0
            tot_pos += newpos
            for prop in props:
                if prop["area"] > size_thresh:
                    if num_detected < newpos:
                        true_pos += 1
                        num_detected += 1
                    else:
                        false_pos += 1
        elif "neg" in filename:
            tot_neg += 1
            for prop in props:
                if prop["area"] > size_thresh:
                    false_pos += 1

    recall = true_pos / tot_pos
    recalls.append(recall)
    if true_pos + false_pos == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)
    precisions.append(precision)
    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    print(
        f"threshold: {threshold}: precision: {precision}, recall: {recall}, f-score: {fscore} for {tot_pos} positive samples in {len(test_files)} images"
    )

fscores = [
    2 * precision * recall / (precision + recall)
    if (precision != 0 and recall != 0)
    else 0
    for precision, recall in zip(precisions, recalls)
]
dict = {
    "Recall": recalls,
    "Precision": precisions,
    "F-score": fscores,
    "Threshold": thresholds,
}
df = pd.DataFrame(dict)
max_fscore = df["F-score"].max()
best_threshold = float(df.loc[df["F-score"] == max_fscore]["Threshold"].iloc[0])
best_rec = float(df.loc[df["F-score"] == max_fscore]["Recall"].iloc[0])
best_prec = float(df.loc[df["F-score"] == max_fscore]["Precision"].iloc[0])

print(f"Max f-score: {max_fscore:.2f} thresh:{best_threshold:.2f}")
print("If this performance is not adequate, improve model and try again")

figname = base_dir + "prec-recall.jpg"
savefig = input(f"Save Precision Recall Curve to: {figname} (y/n)?")
if savefig == "y":
    sns.set(font_scale=2)

    plt.figure(figsize=(8, 8))
    sns.lineplot(data=df, x="Recall", y="Precision", estimator=np.amax, ci=False)
    plt.scatter(
        best_rec,
        best_prec,
        c="r",
        label=f"Max f-score: {max_fscore:.2f} thresh:{best_threshold:.2f}",
    )
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.title(f"Brain {brain} Validation: {tot_pos}+ {tot_neg}-")
    plt.legend()
    plt.savefig(figname)

'''
Making info files for transformed images
'''
make_trans_layers = input(f"Will you be transforming image data into atlas space? (should relevant info files be made) (y/n)")
if make_trans_layers == "y":
    atlas_vol = CloudVolume(
        "precomputed://https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_10um/annotation_10um_2017"
    )
    for layer in [
        antibody_layer,
        background_layer,
    ]:  # axon_mask is transformed into an image because nearest interpolation doesnt work well after downsampling
        layer_path = brain2paths[brain]["base"] + layer + "_transformed"
        print(f"Writing info file at {layer_path}")
        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type="image",
            data_type="uint16",  # Channel images might be 'uint8'
            encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
            resolution=atlas_vol.resolution,  # Voxel scaling, units are in nanometers
            voxel_offset=atlas_vol.voxel_offset,
            chunk_size=[32, 32, 32],  # units are voxels
            volume_size=atlas_vol.volume_size,  # e.g. a cubic millimeter dataset
        )
        vol_mask = CloudVolume(layer_path, info=info)
        vol_mask.commit_info()