import brainlit
from brainlit.utils.ngl_pipeline import NeuroglancerSession
from brainlit.viz import visualize
import matplotlib

ngl_sess = NeuroglancerSession(mip=1)
img, bbbox, vox = ngl_sess.pull_chunk(2, 300, 1, 1, 1)

fig_2d, axis_2d = visualize.plot_image_2d(img[:, 100, :])
img_list = [img[:, 100, :], img[:, 99, :], img[:, 97, :], img[:, 96, :]]
fig_2d_list, ax_2d_list = visualize.plot_image_2d(img_list)

fig_3d, ax_3d = visualize.plot_image_mip(img)
img_list = [img, img]
fig_3d_list, ax_3d_list = visualize.plot_image_mip(img_list)

smallest_axis = visualize.find_smalldim(img)

hist_tuple = visualize.plot_image_hist(img)


def test_fig_plot_image_2d_given_one():
    """test if fig output is correct type (matplotlib figure)"""
    assert isinstance(fig_2d, matplotlib.figure.Figure)


def test_fig_plot_image_2d_given_list():
    """test if fig output is correct type (matplotlib figure)"""
    assert isinstance(fig_2d_list, matplotlib.figure.Figure)


def test_ax_plot_image_2d_given_one():
    """test if axis output is correct type (matplotlib axes)"""
    assert isinstance(axis_2d, matplotlib.axes._subplots.Subplot)


def test_ax_plot_image_2d_given_list():
    """test if axis output is correct type (matplotlib axes)"""
    for ax in ax_2d_list:
        assert isinstance(ax, matplotlib.axes._subplots.Subplot)


def test_fig_plot_mip_given_one():
    """test if output is correct type (matplotlib figure)"""
    assert isinstance(fig_3d, matplotlib.figure.Figure)


def test_fig_plot_mip_given_list():
    """test if output is correct type (matplotlib figure)"""
    assert isinstance(fig_3d_list, matplotlib.figure.Figure)


def test_ax_plot_mip_given_one():
    """test if output is correct type (matplotlib axes)"""
    assert isinstance(ax_3d, matplotlib.axes._subplots.Subplot)


def test_ax_plot_mip_given_list():
    """test if output is correct type (matplotlib axes)"""
    for ax in ax_3d_list:
        assert isinstance(ax, matplotlib.axes._subplots.Subplot)


def test_find_smalldim():
    """test if output is correct type (int)"""
    assert isinstance(smallest_axis, int)


def test_plot_image_hist():
    """test if output is correct type (tuple (n, bins, patches))"""
    assert isinstance(hist_tuple, tuple)
