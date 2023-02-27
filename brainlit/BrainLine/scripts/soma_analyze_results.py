from brainlit.BrainLine.analyze_results import SomaDistribution

"""
Inputs
"""
colors = {
    "test_type": "blue",
}  # colors for different subtypes
symbols = ["o", "+", "^", "vbar"]
brain_ids = ["test"]  # brain ids from soma_data file
fold_on = True  # whether to fold

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

"""
Show coronal section
"""
sd = SomaDistribution(brain_ids=brain_ids)
sd.napari_coronal_section(
    z=1000, subtype_colors=colors, symbols=symbols, fold_on=fold_on
)


"""
Make bar chart
"""
sd.region_barchart(regions, composite_regions=composite_regions, normalize_region=872)
