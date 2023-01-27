import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from brainlit.lsm_analysis import parse_ara


@pytest.fixture(scope="session")  # one server to rule'em all
def create_tree():
    root_node = parse_ara.Node(0, "A", "Alpha", -1, 0, 0)
    root_node.add_child(parse_ara.Node(1, "B", "Bravo", 0, 1, 1))
    root_node.add_child(parse_ara.Node(2, "C", "Charlie", 0, 1, 1))

    return root_node


############################
### functionality checks ###
############################


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
