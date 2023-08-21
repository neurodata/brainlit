from brainlit.map_neurons.diffeo_gen import interp, diffeo_gen_ara
import numpy as np
import torch


def test_interp():
    xs, ys, zs = (
        np.arange(10, dtype="float"),
        np.arange(10, dtype="float"),
        np.arange(10, dtype="float"),
    )
    xv = [xs, ys, zs]
    I = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xv], indexing="ij"), 0)
    xs, ys, zs = (
        np.arange(1, 11, dtype="float"),
        np.arange(1, 11, dtype="float"),
        np.arange(1, 11, dtype="float"),
    )
    xv2 = [xs, ys, zs]
    phii = torch.stack(
        torch.meshgrid([torch.as_tensor(x) for x in xv2], indexing="ij"), -1
    )

    interpd = interp(xv, I, phii)
    true_interp = torch.permute(phii, (3, 0, 1, 2))

    assert interpd.size() == torch.Size([3, 10, 10, 10])
    assert torch.equal(interpd[:, :9, :9, :9], true_interp[:, :9, :9, :9])
    # not checking the border, where the padding rule is applied


def test_expR():
    shp = [132, 80, 114]
    _, phii1 = diffeo_gen_ara(40)

    for sigma in [40, 320]:
        xv, phii = diffeo_gen_ara(sigma)
        assert all([len(x) == n for x, n in zip(xv, shp)])  # xv has good shape
        assert all(s == t for s, t in zip(phii.shape, shp + [3]))  # phii has good shape
        assert not (np.array_equal(phii, phii1))  # transformations are random
