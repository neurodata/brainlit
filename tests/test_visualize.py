from brainlit.utils.ngl_pipeline import NeuroglancerSession
from brainlit.algorithms.generate_fragments import adaptive_thresh
from brainlit.viz.swc import *
from brainlit.viz import visualize
import matplotlib
import SimpleITK as sitk
import napari
import numpy as np

ngl_sess = NeuroglancerSession(mip=1)
img, bbbox, vox = ngl_sess.pull_chunk(2, 300, 1, 1, 1)
img_slices = [img[:, 100, :], img[:, 99, :], img[:, 97, :], img[:, 96, :]]
img_list = [img, img]


def test_napari_viewer():
    """test if output is a napari viewer object"""

    seg_id = 11
    mip = 2
    url = "s3://mouse-light-viz/precomputed_volumes/brain1"
    df = read_s3(url + "_segments", seg_id, mip)
    subneuron_df = df[0:5]  # choose vertices to use for the subneuron
    vertex_list = subneuron_df["sample"].array
    buffer = [10, 10, 10]
    img, bounds, vox_in_img_list = ngl_sess.pull_vertex_list(
        seg_id, vertex_list, buffer=buffer, expand=True
    )
    seed = [adaptive_thresh.get_seed(sample)[1] for sample in vox_in_img_list]
    labels = adaptive_thresh.confidence_connected_threshold(
        img, seed, num_iter=1, multiplier=0.7
    )
    corrected_subneuron_df = generate_df_subset(subneuron_df, vox_in_img_list)
    G = df_to_graph(corrected_subneuron_df)
    paths = graph_to_paths(G)
    try:
        napari.gui_qt()
        n = visualize.napari_viewer(
            img,
            labels=labels,
            shapes=paths,
            label_name="Confidence-Connected Threshold",
        )
        assert isinstance(n, napari.viewer.Viewer)
        n.window.close()
    except RuntimeError:
        napari.gui_qt()


def test_fig_plot_2d_dim3():
    """test if 3 dim SITKImage output is correct type (matplotlib figure)"""

    test_img = np.zeros((100, 100, 100))
    test_img = sitk.GetImageFromArray(test_img)

    fig_2d, axis_2d = visualize.plot_2d(test_img, title="Test", show_plot=True)

    assert isinstance(fig_2d, matplotlib.figure.Figure)
    assert isinstance(axis_2d, matplotlib.axes._subplots.Subplot)


def test_fig_plot_2d_dim4():
    """test if 4 dim SITKImage output is correct type (matplotlib figure)"""

    test_img = np.zeros((100, 100, 100, 4))
    test_img = sitk.GetImageFromArray(test_img)

    fig_2d, axis_2d = visualize.plot_2d(test_img, title="Test", show_plot=False)

    assert isinstance(fig_2d, matplotlib.figure.Figure)
    assert isinstance(axis_2d, matplotlib.axes._subplots.Subplot)


def test_fig_plot_2d_dim4_exception():
    """test if 4 dim SITKImage output is correct type (matplotlib figure)"""

    test_img = np.zeros((100, 100, 100, 100))
    test_img = sitk.GetImageFromArray(test_img)

    try:
        visualize.plot_2d(test_img, title="Test", show_plot=False)
        assert False
    except RuntimeError:
        assert True


def test_fig_plot_3d():
    """test if fig output is correct type (matplotlib figure)"""
    url = "s3://mouse-light-viz/precomputed_volumes/brain1"
    seg_id = 11
    mip = 2
    df = read_s3(url + "_segments", seg_id, mip)

    df["sample"].size  # the number of vertex IDs [1, 2, ..., df['sample'].size]
    subneuron_df = df[0:5]  # choose vertices to use for the subneuron
    vertex_list = subneuron_df["sample"].array
    ngl = NeuroglancerSession(url, mip=mip)
    buffer = [10, 10, 10]
    img, bounds, vox_in_img_list = ngl.pull_vertex_list(
        seg_id, vertex_list, buffer=buffer, expand=True
    )

    fig_2d, axis_2d = visualize.plot_3d(
        sitk.GetImageFromArray(np.squeeze(img), isVector=False),
        xslices=range(48, 53),
        yslices=range(48, 53),
        zslices=range(48, 53),
        title="Downloaded Mouselight Volume",
        show_plot=False,
    )
    assert isinstance(fig_2d, matplotlib.figure.Figure)
    assert isinstance(axis_2d, matplotlib.axes._subplots.Subplot)


def test_fig_plot_image_2d_given_one():
    """test if fig output is correct type (matplotlib figure)"""
    fig_2d, axis_2d = visualize.plot_image_2d(
        img[:, 100, :], titles=["test"], colorbar=True
    )
    assert isinstance(fig_2d, matplotlib.figure.Figure)
    assert isinstance(axis_2d, matplotlib.axes._subplots.Subplot)


def test_fig_plot_image_2d_given_list():
    """test if fig output is correct type (matplotlib figure)"""
    fig_2d_list, ax_2d_list = visualize.plot_image_2d(
        img_slices, titles=["t1", "t2", "t3", "t4"]
    )
    assert isinstance(fig_2d_list, matplotlib.figure.Figure)
    for ax in ax_2d_list:
        assert isinstance(ax, matplotlib.axes._subplots.Subplot)


def test_fig_plot_mip_given_one():
    """test if output is correct type (matplotlib figure)"""
    fig_3d, ax_3d = visualize.plot_image_mip(img)
    assert isinstance(fig_3d, matplotlib.figure.Figure)
    assert isinstance(ax_3d, matplotlib.axes._subplots.Subplot)


def test_fig_plot_mip_given_list():
    """test if output is correct type (matplotlib figure)"""
    fig_3d_list, ax_3d_list = visualize.plot_image_mip(img_list)
    assert isinstance(fig_3d_list, matplotlib.figure.Figure)
    for ax in ax_3d_list:
        assert isinstance(ax, matplotlib.axes._subplots.Subplot)


def test_find_smalldim():
    """test if output is correct type (int)"""
    smallest_axis = visualize.find_smalldim(img)
    assert isinstance(smallest_axis, int)


def test_find_smalldim_list():
    """test if output is correct type (int)"""
    smallest_axis = visualize.find_smalldim(img_list)
    assert isinstance(smallest_axis, int)


def test_plot_image_hist():
    """test if output is correct type (tuple (n, bins, patches))"""
    hist_tuple = visualize.plot_image_hist(img, titles="histogram test")
    assert isinstance(hist_tuple, tuple)


def test_plot_image_hist_list():
    """test if output is correct type (tuple (n, bins, patches))"""
    hist_tuple = visualize.plot_image_hist(img_list)
    assert isinstance(hist_tuple, tuple)


def test_plot_img_pts_single_scatter():
    """test plot_image_pts method with voxels"""
    fig, axes = visualize.plot_image_pts(img[:, :, 100], vox, colorbar=True)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axes, matplotlib.axes._subplots.Subplot)


def test_plot_img_pts_multi_scatter():
    """test plot_image_pts method with voxels"""
    voxes = np.random.rand(3, 2)
    fig, axes = visualize.plot_image_pts(img[:, :, 100], voxes, colorbar=True)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axes, matplotlib.axes._subplots.Subplot)
