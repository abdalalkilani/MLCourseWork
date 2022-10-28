import numpy as np

from training import decision_tree_learning
from evaluation import evaluate

def postpruning(built_tree):
    depth = decision_tree_learning

    right_node = built_tree['right']
    left_node = built_tree['left']
    left_depth = left_node['depth']
    right_depth = right_node['depth']

    while(left_depth & right_depth !=0):
        postpruning(left_depth)
        postpruning(right_depth)
        if left_depth & right_depth == 0:
            temp_tree = built_tree.pop(left_node)
            new_tree = temp_tree.pop(right_node)
            if evaluate(new_tree) > evaluate(built_tree):
                built_tree = new_tree
