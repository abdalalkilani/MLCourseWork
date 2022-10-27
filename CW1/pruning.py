import numpy as np
from training import decision_tree_learning

def pruning(tree):
    if(tree['left']['depth'] and tree['right']['depth']):
        # replace node with leaf
    else:
        # idk i neeed some cake