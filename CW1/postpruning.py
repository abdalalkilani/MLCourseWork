import numpy as np
import copy

from training import decision_tree_learning
from evaluation import evaluate

def replace_parent(tree, parent_node, new_leaf_value):
    for k, v in tree.items():

        if isinstance(v, dict):
            tree[k] = replace_parent(v, parent_node, new_leaf_value)

    if parent_node in tree:
        tree[parent_node] = new_leaf_value

    return tree

def make_new_tree(whole_tree, new_tree, string_leaf):
    while(len(string_leaf)> 0):
        return 
    return whole_tree

def postpruning(built_tree, whole_tree, path_string):

    right_node = built_tree['right']
    left_node = built_tree['left']

    old_tree =  copy.deepcopy(built_tree)
    initial = 0
    current_accuracy = evaluate(built_tree)

    while ((old_tree != built_tree) or (initial == 0)):
        initial = 1

        try:
            left_depth = left_node['depth']
            try:
                right_depth = right_node['depth']
                # could i just remove next 2 lines and just say:
                # new_tree = built_tree
                # new_tree.clear()
                # then continue
                # or
                # temp_tree = built_tree.pop(left_node)
                # new_tree = temp_tree.pop(right_node)
                # new_tree.clear()
                new_tree = {i: left_node.get(i, 0) + right_node.get(i, 0) for i in set(left_node).union(right_node)}
                new_accuracy = evaluate(new_tree)

                if new_accuracy >= current_accuracy:
                    # or is next line
                    # built_tree =  copy.deepcopy(new_tree)
                    built_tree = new_tree
                    current_accuracy = new_accuracy
                    return new_tree
            except KeyError:
                path_string += "R"
                right_node = postpruning(right_node, whole_tree, path_string)

        except KeyError:
            try:
                right_depth = right_node['depth']
                path_string += "L"
                left_node = postpruning(left_node, whole_tree, path_string)
            except KeyError:
                path_string += "L"
                left_node = postpruning(left_node, whole_tree, path_string)
                path_string = path_string[:-1] + "R"
                right_node = postpruning(right_node, whole_tree, path_string)


    return built_tree
