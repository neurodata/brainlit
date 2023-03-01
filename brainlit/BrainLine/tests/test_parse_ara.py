import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from brainlit.BrainLine import parse_ara
import json


@pytest.fixture(scope="session")  # one server to rule'em all
def create_tree():
    root_node = parse_ara.Node(0, "A", "Alpha", -1, 0, 0)
    root_node.add_child(parse_ara.Node(1, "B", "Bravo", 0, 1, 1))
    root_node.add_child(parse_ara.Node(2, "C", "Charlie", 0, 1, 1))

    return root_node


@pytest.fixture(scope="session")  # one server to rule'em all
def create_json(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")

    root_json = {
        "id": 0,
        "acronym": "A",
        "name": "Alpha",
        "parent_structure_id": -1,
        "st_level": 1,
        "children": [],
    }
    b_json = {
        "id": 1,
        "acronym": "B",
        "name": "Bravo",
        "parent_structure_id": 0,
        "st_level": 2,
        "children": [],
    }
    c_json = {
        "id": 2,
        "acronym": "C",
        "name": "Charlie",
        "parent_structure_id": 0,
        "st_level": 2,
        "children": [],
    }
    root_json["children"] = [b_json, c_json]

    tree_json = json.dumps(root_json)

    path = data_dir / "tree.json"

    with open(path, "w") as outfile:
        outfile.write(tree_json)

    return path, root_json


############################
### functionality checks ###
############################


def test_build_tree(create_json):
    tree = parse_ara.build_tree(create_json[1])

    assert tree.id == 0
    assert tree.level == 0
    assert tree.st_level == 1
    assert len(tree.children) == 2


def test_get_nodes_at_level(create_tree):
    nodes = parse_ara.get_nodes_at_level(0, create_tree)
    assert len(nodes) == 1
    assert "Alpha" in str(nodes[0])

    nodes = parse_ara.get_nodes_at_level(1, create_tree)
    assert len(nodes) == 2
    assert "Alpha" not in str(nodes[0])


def test_get_all_ids_of_children(create_tree):
    nodes = parse_ara.get_all_ids_of_children(create_tree)
    assert len(nodes) == 2
    assert "Alpha" not in str(nodes[0])


def test_get_parent_dict(create_json):
    id2parent2 = parse_ara.get_parent_dict(create_json[0], level=0)
    assert id2parent2[1] == 0
    assert id2parent2[2] == 0


def test_get_children_dict(create_json):
    id2child = parse_ara.get_children_dict(create_json[0], level=0)

    assert len(id2child.keys()) == 1
    assert id2child[0] == [1, 2]
