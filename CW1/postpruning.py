import numpy as np

from training import decision_tree_learning
from evaluation import evaluate

def postpruning(built_tree):

    depth = decision_tree_learning
    right_node = built_tree['right']
    left_node = built_tree['left']
    left_depth = left_node['depth']
    right_depth = right_node['depth']

    old_tree = built_tree
    initial = 0
    current_accuracy = evaluate(built_tree)

    while ((old_tree != built_tree) or (initial == 0)):
        initial = 1
        
        if(left_depth != 0 & right_depth !=0):
            postpruning(left_node)
            postpruning(right_node)

        else:
            temp_tree = built_tree.pop(left_node)
            new_tree = temp_tree.pop(right_node)
            new_accuracy = evaluate(new_tree)

            if new_accuracy >= current_accuracy:
                built_tree = new_tree
                current_accuracy = new_accuracy

    return built_tree
