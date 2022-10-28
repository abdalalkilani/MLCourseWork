import numpy as np

from training import decision_tree_learning
from evaluation import evaluate

def postpruning(built_tree):
    depth = decision_tree_learning

    right_node = built_tree['right']
    left_node = built_tree['left']
    left_depth = left_node['depth']
    right_depth = right_node['depth']

    if left_depth & right_depth != 0:
        new_tree = built_tree
        if evaluate(new_tree) > evaluate(built_tree):
            built_tree = new_tree
