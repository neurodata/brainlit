from brainlit.BrainLine.analyze_results import SomaDistribution, AxonDistribution
import pickle
import pytest


def test_SomaDistribution():
    subtype_colors = {"test_type": "red"}

    sd = SomaDistribution(["pytest"], show_plots=False)
    # sd.brainrender_somas(subtype_colors=subtype_colors, brain_region="MOB")
    # sd.napari_coronal_section(z=1000, subtype_colors=subtype_colors, fold_on=True)
    sd.region_barchart(
        regions=[362, 795],
        composite_regions={"Olfactory Bulb": [507, 698, 1016]},
        normalize_region=507,
    )
    sd.region_barchart(
        regions=[362, 795], composite_regions={"Olfactory Bulb": [507, 698, 1016]}
    )


def test_collect_regional_segmentation():
    pass  # would need to create axon_mask and atlas_to_target layers


def test_AxonDistribution(tmp_path):
    subtype_colors = {"test_type": "red"}

    invalid_volumes = {698: [0, 11], 795: [20, 1]}
    outpath = tmp_path / "wholebrain_pytest.pkl"
    with open(outpath, "wb") as f:
        pickle.dump(invalid_volumes, f)
    ad = AxonDistribution(
        brain_ids=["pytest"],
        regional_distribution_dir=str(tmp_path) + "/",
        show_plots=False,
    )
    with pytest.raises(ValueError) as e_info:
        ad.region_barchart(
            regions=[362, 795], composite_regions={"Olfactory Bulb": [507, 698, 1016]}
        )

    invalid_volumes = {698: [11, 11], 795: [0, 1]}
    outpath = tmp_path / "wholebrain_pytest.pkl"
    with open(outpath, "wb") as f:
        pickle.dump(invalid_volumes, f)
    ad = AxonDistribution(
        brain_ids=["pytest"],
        regional_distribution_dir=str(tmp_path) + "/",
        show_plots=False,
    )
    with pytest.raises(ValueError) as e_info:
        ad.region_barchart(
            regions=[362, 795], composite_regions={"Olfactory Bulb": [507, 698, 1016]}
        )

    valid_volumes = {
        507: [10, 10],
        698: [10, 5],
        1016: [10, 0],
        362: [20, 0],
        795: [20, 1],
    }
    outpath = tmp_path / "wholebrain_pytest.pkl"
    with open(outpath, "wb") as f:
        pickle.dump(valid_volumes, f)

    ad = AxonDistribution(
        brain_ids=["pytest"],
        regional_distribution_dir=str(tmp_path) + "/",
        show_plots=False,
    )
    # testing napari and brainrender is expensive because it involves downloading segmentation data
    ad.region_barchart(
        regions=[362, 795],
        composite_regions={"Olfactory Bulb": [507, 698, 1016]},
        normalize_region=507,
    )
    ad.region_barchart(
        regions=[362, 795], composite_regions={"Olfactory Bulb": [507, 698, 1016]}
    )
