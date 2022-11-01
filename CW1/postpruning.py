import numpy as np
import copy
from printing import print_tree

from training import decision_tree_learning, evaluate

import random
import json


test_data = {
    "attribute" : "Col 5",
    "value" : -20,
    "left" : {
        "attribute" : "Col 2",
        "value" : 30,
        "left" : {
            "attribute" : "Col 2",
            "value" : 30,
            "left" : {
                "attribute" : "Col 2",
                "value" : 30,
                "left" : {
                    "attribute" : "Col 2",
                    "value" : 30,
                    "left" : {
                        "label" : {
                            "Col 1" : 3,
                            "Col 2" : 2,
                            "Col 3" : 1
                        }, 
                        "depth" : 0
                    },
                    "right" : {
                        "label" : {
                            "Col 2"  : 5,
                            "Col 1" : 3,
                            "Col 6" : 1
                        }, 
                        "depth" : 0
                    }
                },
                "right" : {
                    "label" : {
                        "Col 2"  : 5,
                        "Col 1" : 3,
                        "Col 6" : 1
                    }, 
                    "depth" : 0
                }
            },
            "right" : {
                "label" : {
                    "Col 2"  : 5,
                    "Col 1" : 3,
                    "Col 6" : 1
                }, 
                "depth" : 0
            }
        },
        "right" : {
            "attribute" : "Col 2",
            "value" : 30,
            "left" : {
                "label" : {
                    "Col 1" : 3,
                    "Col 2" : 2,
                    "Col 3" : 1
                }, 
                "depth" : 0
            },
            "right" : {
                "label" : {
                    "Col 2"  : 5,
                    "Col 1" : 3,
                    "Col 6" : 1
                }, 
                "depth" : 0
            }
        }
    },
    "right" : {
        "attribute" : "Col 7",
        "value" : -5,
        "left" : {
            "label" : {
                "Col 8" : 9,
                "Col 3" : 3,
                "Col 5" : 2
            }, 
            "depth" : 0
        },
        "right" : {
            "label" : {
                "Col 8" : 3,
                "Col 6" : 2,
                "Col 7" : 1
            }, 
            "depth" : 0
        }
    }
}


def replace_parent(tree, parent_node, new_leaf_value):
    for k, v in tree.items():

        if isinstance(v, dict):
            tree[k] = replace_parent(v, parent_node, new_leaf_value)

    if parent_node in tree:
        tree[parent_node] = new_leaf_value

    return tree

# def make_new_tree(whole_tree, new_tree, string_leaf):
#     #  
#     if(len(string_leaf)> 1):
#         if(string_leaf[0] == "R"):
#             current_tree = whole_tree['right']
#         else:
#             current_tree = whole_tree['left']
#         # string_leaf = string_leaf[:-1]
#         return make_new_tree(current_tree, new_tree, string_leaf[:-1])
#     elif(string_leaf == "R"):
#         current_tree["right"] = new_tree
#     else:
#         current_tree["left"] = new_tree

#     return whole_tree

# def make_new_tree2(whole_tree, new_tree, built_tree):
#     for k, v in whole_tree.items():
#         if isinstance(v, dict):
#             return make_new_tree2(v) 
#         else:
#             return d

# def _finditem(obj, key):
#     if key in obj: 
#         return obj[key]
#     for k, v in obj.items():
#         if isinstance(v,dict):
#             return _finditem(v, key) 

# import collections.abc

# def update(d, u):
#     for k, v in u.items():
#         if isinstance(v, collections.abc.Mapping):
#             d[k] = update(d.get(k, {}), v)
#         else:
#             d[k] = v
#     return d

def postpruning(built_tree, whole_tree, path_string):

    right_node = built_tree['right']
    left_node = built_tree['left']

    old_tree =  copy.deepcopy(built_tree)
    initial = 0
    # current_accuracy = evaluate(built_tree)
    current_accuracy = random.uniform(0, 1)

    while ((old_tree != built_tree) or (initial == 0)):
        initial = 1
        print("New Tree")
        print(json.dumps(built_tree, indent=4))
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
                label = {i: left_node['label'].get(i, 0) + right_node['label'].get(i, 0) 
                            for i in set(left_node).union(right_node)}
                new_tree = { 'label': {},
                             'depth': 0}
                new_tree['label'] = label

                built_tree = new_tree
                # new_accuracy = evaluate(whole_tree)
                new_accuracy = random.uniform(0, 1)
                if new_accuracy >= current_accuracy:
                    # or is next line
                    # built_tree =  copy.deepcopy(new_tree)
                    built_tree = new_tree
                    current_accuracy = new_accuracy
                    return new_tree
                else:
                    whole_tree = old_tree
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

postpruning(test_data, test_data, "")