from brainlit.algorithms.image_processing import Bresenham3D
from pytest import raises

####################
### input checks ###
####################


def test_Bresenham3D_bad_input():
    # test all the inputs must be integer
    [x1, y1, z1, x2, y2, z2] = [1, 2, 3, 4, 5, 6]
    with raises(TypeError):
        Bresenham3D(x1 + 0.5, y1, z1, x2, y2, z2)
    with raises(TypeError):
        Bresenham3D(x1, y1 + 0.5, z1, x2, y2, z2)
    with raises(TypeError):
        Bresenham3D(x1, y1, z1 + 0.5, x2, y2, z2)
    with raises(TypeError):
        Bresenham3D(x1, y1, z1, x2 + 0.5, y2, z2)
    with raises(TypeError):
        Bresenham3D(x1, y1, z1, x2, y2 + 0.5, z2)
    with raises(TypeError):
        Bresenham3D(x1, y1, z1, x2, y2, z2 + 0.5)


####################
#### validation ####
####################


def test_Bresenham3D():
    # use examples from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/

    # example 1
    [x1, y1, z1, x2, y2, z2] = [-1, 1, 1, 5, 3, -1]
    xlist = [-1, 0, 1, 2, 3, 4, 5]
    ylist = [1, 1, 2, 2, 2, 3, 3]
    zlist = [1, 1, 0, 0, 0, -1, -1]
    t_x, t_y, t_z = Bresenham3D(x1, y1, z1, x2, y2, z2)
    assert t_x == xlist
    assert t_y == ylist
    assert t_z == zlist

    # example 2
    [x1, y1, z1, x2, y2, z2] = [-7, 0, -3, 2, -5, -1]
    xlist = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
    ylist = [0, -1, -1, -2, -2, -3, -3, -4, -4, -5]
    zlist = [-3, -3, -3, -2, -2, -2, -2, -1, -1, -1]
    t_x, t_y, t_z = Bresenham3D(x1, y1, z1, x2, y2, z2)
    assert t_x == xlist
    assert t_y == ylist
    assert t_z == zlist
