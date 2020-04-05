import pytest
import brainlit
from brainlit.algorithms.generate_fragments import tube_seg
import numpy as np
from brainlit.utils.ngl_pipeline import NeuroglancerSession
from skimage import draw


def test_pairwise():
    iterable = np.random.randint(10,size = (7,3))
    pair = list(tube_seg.pairwise(iterable))
    
    assert (iterable[:-1,:] == [a[0] for a in pair]).all()
    assert (iterable[1:,:] == [b[1] for b in pair]).all()
    
    
def test_draw_sphere():
    ngl_session = NeuroglancerSession()
    img, _, _ = ngl_session.pull_vertex_list(13, [4], expand = True)
    shape = img.shape
    center = [np.random.randint(shape[0]),np.random.randint(shape[1]),np.random.randint(shape[2])]
    radius = np.random.randint(1,4)
    sphere = tube_seg.draw_sphere(shape,center,radius)
    coords = np.where(sphere < 1)
    d_bg = min(np.sum((np.array(coords).T-center)**2, axis = 1))
    coords = np.where(sphere > 0)
    d_s = max(np.sum((np.array(coords).T-center)**2, axis = 1))
    
    assert sphere.shape == shape
    assert np.unique(sphere).all() in [0,1]
    assert d_bg > radius**2
    assert d_s <= radius**2


def test_draw_tube():
    ngl_session = NeuroglancerSession()
    img, _, _ = ngl_session.pull_vertex_list(13, [4], expand = True)
    shape = img.shape
    vertex0 = [np.random.randint(shape[0]/2),np.random.randint(shape[1]),np.random.randint(shape[2])]
    vertex1 = [np.random.randint(shape[0]/2,shape[0]),np.random.randint(shape[1]),np.random.randint(shape[2])]
    radius = np.random.randint(1,4)
    labels = tube_seg.draw_tube(img,vertex0,vertex1,radius)
    line = draw.line_nd(vertex0, vertex1, endpoint=True)
    coords = np.where(labels < 1)
    d_bg = max(shape)
    for pt in np.array(coords).T:
        distance_min = min(np.sum((np.array(line).T-pt)**2,axis = 1))
        d_bg = min(distance_min,d_bg)
    coords = np.where(labels > 0)
    d_tube = 0
    for pt in np.array(coords).T:
        distance_min = min(np.sum((np.array(line).T-pt)**2,axis = 1))
        d_tube = max(distance_min,d_tube)
    
    assert labels.shape == shape
    assert np.unique(labels).all() in [0,1]
    assert d_bg > radius**2
    assert d_tube <= radius**2


def test_tubes_seg():
    ngl_session = NeuroglancerSession()
    img, _, _ = ngl_session.pull_vertex_list(13, [4], expand = True)
    shape = img.shape
    vertices = np.random.randint(min(shape),size = (4,3))
    radius = np.random.randint(1,4)
    labels = tube_seg.tubes_seg(img,vertices,radius)
    point = np.empty((3,0),dtype = int)
    for i in range(3):
        lines = draw.line_nd(vertices[i], vertices[i+1], endpoint=True)
        point = np.concatenate((point,np.array(lines)),axis = 1) 
    coords = np.where(labels < 1)
    d_bg = max(shape)
    for pt in np.array(coords).T:
        distance_min = min(np.sum((point.T-pt)**2,axis = 1))
        d_bg = min(distance_min,d_bg)
    coords = np.where(labels > 0)
    d_tube = 0
    for pt in np.array(coords).T:
        distance_min = min(np.sum((point.T-pt)**2,axis = 1))
        d_tube = max(distance_min,d_tube)
        
    assert labels.shape == shape
    assert np.unique(labels).all() in [0,1]
    assert d_bg > radius**2
    assert d_tube <= radius**2

