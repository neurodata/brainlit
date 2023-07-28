from brainlit.map_neurons.utils import (
    replace_root,
    resample_neuron,
    split_paths,
    find_longest_path,
    remove_path,
    ZerothFirstOrderNeuron,
)
from brainlit.map_neurons.map_neurons import DiffeomorphismAction
import ngauge
import pytest
import numpy as np
from scipy.interpolate import splprep, splev


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
    for branch_idx, idx in enumerate(range(6, 9)):
        node = f"{idx} 3 0 {spac*(branch_idx+2)+1} 0 1 {idx-1}"
        text.append(node)

    return text


class IdentityAction(DiffeomorphismAction):
    def evaluate(self, position: np.array) -> np.array:
        return position

    def D(self, position, deriv, order=1):
        return deriv


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
    assert np.isclose(np.linalg.norm([a_child.x, a_child.y, a_child.z]), 5)

    # new root's children has correct parent
    for child in root.children:
        assert child.parent == root


def test_resample_neuron(init_swc_text):
    text = init_swc_text
    neuron = ngauge.Neuron.from_swc_text(text)

    # original child if root is 5 away from origin
    a_root = neuron.branches[0]
    a_child = a_root.children[0]
    assert np.isclose(np.linalg.norm([a_child.x, a_child.y, a_child.z]), 5)

    neuron = replace_root(neuron)
    og_root = neuron.branches[0]

    neuron = resample_neuron(neuron, sampling=1.0)
    assert og_root == neuron.branches[0]

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

        assert np.linalg.norm(np.subtract(pt2, pt1)) <= 1.0

    assert counter == 30


def test_find_longest_path(init_swc_text):
    text = init_swc_text
    neuron = ngauge.Neuron.from_swc_text(text)
    neuron = replace_root(neuron)

    _, longest_path = find_longest_path(neuron.branches[0])

    assert len(longest_path) == 5
    assert longest_path[0].y == 0
    assert longest_path[-1].y == 17


def test_remove_path(init_swc_text):
    text = init_swc_text
    neuron = ngauge.Neuron.from_swc_text(text)
    neuron = replace_root(neuron)

    root = neuron.branches[0]
    stack = []
    stack += [root]
    path = [root]

    while len(stack) > 0:
        node = stack.pop()
        children = node.children
        if len(children) > 0:
            path.append(children[-1])
            stack += children
        else:
            break

    subtrees = remove_path(path)

    assert len(subtrees) == 1
    assert subtrees[0][0].parent == subtrees[0][1]
    assert np.linalg.norm([subtrees[0][0].x, subtrees[0][0].y, subtrees[0][0].z]) == 5


def test_split_paths(init_swc_text):
    text = init_swc_text
    neuron = ngauge.Neuron.from_swc_text(text)
    neuron = replace_root(neuron)

    paths = split_paths(neuron.branches[0])

    assert len(paths) == 2

    assert len(paths[0]) == 5
    assert paths[0][-1].y == 17

    assert len(paths[1]) == 4
    assert paths[1][-1].x == 13


def test_ZerothFirstOrderNeuron(init_swc_text):
    sampling = 1
    text = init_swc_text
    da = IdentityAction()

    neuron = ngauge.Neuron.from_swc_text(text)
    zon = ZerothFirstOrderNeuron(neuron, da, sampling=sampling)

    DG = zon.DG

    # check paths
    assert len(DG.nodes) == 2
    assert len(DG.nodes[0]["path"]) == 5
    assert DG.nodes[0]["path"][-1].y == 17
    assert len(DG.nodes[1]["path"]) == 4
    assert DG.nodes[1]["path"][-1].x == 13

    # check gt
    tck, u = DG.nodes[0]["gt"]
    dists = np.diff(u)
    assert np.amax(dists) < sampling
    pt = splev(u[0], tck)
    assert np.isclose(pt, [0, 0, 0]).all()
    pt = splev(u[-1], tck)
    assert np.isclose(pt, [0, 17, 0]).all()

    tck, u = DG.nodes[1]["gt"]
    dists = np.diff(u)
    assert np.amax(dists) < sampling
    pt = splev(u[0], tck)
    assert np.isclose(pt, [0, 0, 0]).all()
    pt = splev(u[-1], tck)
    assert np.isclose(pt, [13, 0, 0]).all()

    neuron = zon.get_gt()
    assert neuron.total_tip_nodes() == 2

    neuron_0, neuron_1 = zon.get_transforms()
    assert neuron_0.total_tip_nodes() == 2
    assert neuron_1.total_tip_nodes() == 2
