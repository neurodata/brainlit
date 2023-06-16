from brainlit.map_neurons.utils import replace_root, resample_neuron
import ngauge
import pytest 
import numpy as np

@pytest.fixture(scope="session")
def init_swc_text():
    spac = 4
    text = []
    soma = "1 1 0 0 0 1 -1"
    text.append(soma)

    for branch_idx, idx in enumerate(range(2, 5)):
        node = f"{idx} 2 {spac*(branch_idx+1)+1} 0 0 1 {idx-1}"
        text.append(node)
    
    new_branch = f"5 3 0 {spac+1} 0 1 1"
    text.append(new_branch)
    for branch_idx, idx in enumerate(range(6, 8)):
        node = f"{idx} 3 0 {spac*(branch_idx+2)+1} 0 1 {idx-1}"
        text.append(node)

    return text

def check_single_child(start_nodes):
    stack = []
    stack += start_nodes
    while len(stack) > 0:
        child = stack.pop()
        stack += child.children

        assert len(child.children) <= 1

def test_replace_root(init_swc_text):
    text = init_swc_text
    neuron = ngauge.Neuron.from_swc_text(text)

    assert len(neuron.branches) == 2
    check_single_child(neuron.branches)
    neuron = replace_root(neuron)

    # check single root, and each node except for root has at most one child
    assert len(neuron.branches) == 1
    root = neuron.branches[0]
    assert len(root.children) == 2
    check_single_child(root.children)

    # new root is in right location
    assert np.isclose(root.x, 0)
    assert np.isclose(root.y, 0)
    assert np.isclose(root.z, 0)

    # new root has correct children
    assert len(root.children) == 2
    a_child = root.children[1]
    assert np.isclose(np.linalg.norm([a_child.x,a_child.y,a_child.z]), 5)

    # new root's children has correct parent
    for child in root.children:
        assert child.parent == root

def test_resample_neuron(init_swc_text):
    text = init_swc_text
    neuron = ngauge.Neuron.from_swc_text(text)

    # original child if root is 5 away from origin
    a_root = neuron.branches[0]
    a_child = a_root.children[0]
    assert np.isclose(np.linalg.norm([a_child.x,a_child.y,a_child.z]), 5)

    neuron = resample_neuron(neuron, sampling = 1.)

    root = neuron.branches[0]
    stack = []
    stack += root.children
    check_single_child(stack)

    counter = 0
    while len(stack) > 0:
        counter += 1
        child = stack.pop()
        stack += child.children
        parent = child.parent


        pt1 = [parent.x, parent.y, parent.z]
        pt2 = [child.x, child.y, child.z]

        assert np.linalg.norm(np.subtract(pt2, pt1)) <= 1.

    assert counter == 26

