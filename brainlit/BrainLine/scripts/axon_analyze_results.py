from brainlit.BrainLine.analyze_results import collect_regional_segmentation, AxonDistribution
import warnings

'''
Inputs
'''
brain_ids = ["test"] # brain ids from soma_data file

regional_distribution_dir = '/Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/'
max_coords = [3072, 4352, 1792]

colors = {
    "test_type": "red",
}  # colors for different subtypes
fold_on = True # whether to fold

regions = [
    688,  # cerebral cortex
    698,  # olfactory areas
    1089,  # hippocampal formation
    # 583, # claustrum
    477,  # striatum
    # 803, # pallidum
    351,  # bed nuclei of stria terminalis
    # 703, #cortical subplate
    1097,  # hypothalamus
    549,  # thalamus
    186,  # lateral habenula
    519,  # cerebellar nuclei
    313,  # midbrain
    1065,  # hindbrain
]  # allen atlas region IDs to be shown in bar chart
# see: https://connectivity.brain-map.org/projection/experiment/480074702?imageId=480075280&initImage=TWO_PHOTON&x=17028&y=11704&z=3
composite_regions = {
    "Amygdalar Nuclei": [131, 295, 319, 780]
}  # Custom composite allen regions where key is region name and value is list of allen regions


warnings.filterwarnings("ignore")

'''
Collect regional data
'''
collect_ask = input(f"Do you want to collect regional segmentation data (i.e. are registration and segmentation complete)? (y/n)")
if collect_ask == "y":
    collect_regional_segmentation(brain_ids[0], regional_distribution_dir, ncpu = 1, max_coords=max_coords)

'''
Show coronal section
'''
ad = AxonDistribution(brain_ids = brain_ids, regional_distribution_dir=regional_distribution_dir)
ad.napari_coronal_section(z=1000, subtype_colors = colors, fold_on = fold_on)

'''
Make bar chart
'''
ad.region_barchart(regions, composite_regions=composite_regions, normalize_region=872)