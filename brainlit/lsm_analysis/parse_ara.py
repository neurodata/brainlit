import json
from collections import defaultdict


class Node:
    def __init__(self, id_, acronym, name, parent_id, st_level, level=0):
        self.id = id_
        self.acronym = acronym
        self.name = name
        self.level = level
        #         self.color = None
        self.children = []
        self.parent_id = parent_id
        self.st_level = st_level

    def __repr__(self):
        return "name: {},level: {},id: {},children: {}\n".format(
            self.name, self.level, self.id, repr(self.children)
        )

    def add_child(self, child):
        self.children.append(child)


def build_tree(obj, level=0):
    node = Node(
        obj["id"],
        obj["acronym"],
        obj["name"],
        obj["parent_structure_id"],
        obj["st_level"],
        level=level,
    )
    for i in obj["children"]:
        node.add_child(build_tree(i, level=level + 1))
    return node


def get_nodes_at_level(level, tree):
    result = []
    # base case
    if tree.level == level:
        result.append(tree)
    else:
        for i in tree.children:
            result += get_nodes_at_level(level, i)

    return result


def get_all_ids_of_children(tree):
    result = []
    if len(tree.children) == 0:
        result.append(tree.id)
    else:
        for i in tree.children:
            result += get_all_ids_of_children(i)

    return result


def get_parent_dict(json_file, level=1):
    f = json.load(open(json_file, "r"))
    tree = build_tree(f)
    nodes = get_nodes_at_level(level, tree, nodes)
    id2parent = defaultdict(lambda: "unknown")
    for i in nodes:
        x = get_all_ids_of_children(i)
        for j in x:
            id2parent[j] = i.id
    id2parent2 = {i: j for i, j in id2parent.items() if i != j}
    return id2parent2


def get_children_dict(json_file, level=1):
    f = json.load(open(json_file, "r"))
    tree = build_tree(f)
    nodes = get_nodes_at_level(level, tree, nodes)
    id2child = defaultdict(lambda: "unknown")
    for i in nodes:
        get_all_ids_of_children(i)
        if len(x) <= 1:
            continue
        else:
            id2child[i.id] = x
    return id2child


def get_child_nodes_from_ontology(node, id2name):
    id2name[node.id] = node.name
    # base case
    if node.children == []:
        return id2name
    # other case
    for i in range(len(node.children)):
        id2name = get_child_nodes_from_ontology(node.children[i], id2name)
    return id2name
