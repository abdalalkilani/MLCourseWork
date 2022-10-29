import numpy as np

from training import decision_tree_learning
from evaluation import evaluate

def replace_parent(tree, parent_node, new_leaf_value):
    for k, v in tree.items():

        if isinstance(v, dict):
            tree[k] = replace_parent(v, parent_node, new_leaf_value)

    if parent_node in tree:
        tree[parent_node] = new_leaf_value

    return tree

def postpruning(built_tree):

    depth = built_tree['depth']
    right_node = built_tree['right']
    left_node = built_tree['left']
    left_depth = left_node['depth']
    right_depth = right_node['depth']

    old_tree = built_tree
    initial = 0
    current_accuracy = evaluate(built_tree)

    while ((old_tree != built_tree) or (initial == 0)):
        initial = 1

        if(depth != 1):
            left_node = postpruning(left_node)
            right_node = postpruning(right_node)

        else:
            temp_tree = built_tree.pop(left_node)
            new_tree = temp_tree.pop(right_node)
            new_tree['value'] = new_tree
            new_accuracy = evaluate(new_tree)

            if new_accuracy >= current_accuracy:
                built_tree = new_tree
                current_accuracy = new_accuracy


    return built_tree
