from brainlit.BrainLine.analyze_results import SomaDistribution, AxonDistribution
from brainlit.BrainLine.tests.test_util import ontology_path
import pickle
import pytest
from pathlib import Path
import os
import numpy as np
from numpy.testing import assert_array_equal

soma_data_file = (
    Path(os.path.abspath(__file__)).parents[3]
    / "docs"
    / "notebooks"
    / "pipelines"
    / "BrainLine"
    / "soma_data.json"
)


@pytest.fixture(scope="session")
def fixes_file():
    fixes_file = (
        Path(os.path.abspath(__file__)).parents[1] / "data" / "open-nd-ara-fixes.json"
    )
    return fixes_file


# BrainDistribution


def test_slicetolabels(ontology_path, fixes_file):
    sd = SomaDistribution(
        ["pytest"],
        data_file=soma_data_file,
        ontology_file=ontology_path,
        show_plots=False,
        fixes_file=fixes_file,
    )

    slice = np.array([[0, 997, 315], [184, 68, 667]], dtype="int")
    new_slice, _, half_width = sd._slicetolabels(slice, fold_on=False, atlas_level=5)
    assert half_width == -1
    assert_array_equal(
        new_slice, np.array([[0, 997, 315], [315, 315, 315]], dtype="int")
    )

    new_slice, _, half_width = sd._slicetolabels(slice, fold_on=True, atlas_level=5)
    assert half_width == 2
    assert_array_equal(new_slice, np.array([[0, 997], [315, 315]], dtype="int"))


# SomaDistribution


def test_SomaDistribution_no_somasatlasurl(ontology_path, fixes_file):
    with pytest.raises(ValueError):
        SomaDistribution(
            ["pytest_nosomasatlasurl"],
            data_file=soma_data_file,
            ontology_file=ontology_path,
            show_plots=False,
            fixes_file=fixes_file,
        )


def test_SomaDistribution(ontology_path):
    sd = SomaDistribution(
        ["pytest", "pytest2"],
        data_file=soma_data_file,
        ontology_file=ontology_path,
        show_plots=False,
    )
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


def test_SomaDistribution_shuffle_bootstrap(ontology_path):
    sd = SomaDistribution(
        ["pytest", "pytest2"],
        data_file=soma_data_file,
        ontology_file=ontology_path,
        show_plots=False,
        bootstrap=1,
        shuffle=True,
    )


# Other methods


def test_collect_regional_segmentation():
    pass  # would need to create axon_mask and atlas_to_target layers


# AxonDistribution


def test_AxonDistribution(tmp_path, ontology_path):
    subtype_colors = {"test_type": "red"}

    invalid_volumes = {698: [0, 11], 795: [20, 1]}
    outpath = tmp_path / "wholebrain_pytest.pkl"
    with open(outpath, "wb") as f:
        pickle.dump(invalid_volumes, f)
    data_file = (
        Path(os.path.abspath(__file__)).parents[3]
        / "docs"
        / "notebooks"
        / "pipelines"
        / "BrainLine"
        / "axon_data.json"
    )
    ad = AxonDistribution(
        brain_ids=["pytest"],
        data_file=data_file,
        ontology_file=ontology_path,
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
        data_file=data_file,
        ontology_file=ontology_path,
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
        data_file=data_file,
        ontology_file=ontology_path,
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
