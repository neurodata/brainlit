from brainlit.BrainLine.analyze_results import SomaDistribution

colors = {
    "test": "blue",
}  # colors for different genotypes
symbols = ["o", "+", "^", "vbar"]
brain_ids = ["test"]
fold_on = True

sd = SomaDistribution(brain_ids = brain_ids)
sd.napari_coronal_section(z=1000, subtype_colors = colors, symbols = symbols, fold_on = fold_on)
