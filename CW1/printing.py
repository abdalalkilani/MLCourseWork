''' 
Description: file containing the functions needed to print out a visualisation of the binary decision tree

decision_tree_learning: - takes in a dataset as a matrix and a depth variable 
                        - outputs a decision tree of specified max depth as a dictionary
                        - 

'''

from array import array
from this import s
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
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
        }
    }
}

squeeze_bounds = 0.8

def modify_tree(decision_tree: dict, depth: int):
    decision_tree["depth"] = depth
    if "label" in decision_tree.keys():
        # It's a leaf node. Find the label for the leaf.
        max_label = ["x", 0]

        for values in decision_tree["label"].keys():
            if decision_tree["label"][values] > max_label[1]:
                max_label[0] = values

        decision_tree["text"] = max_label[0]
        return depth
    else:
        decision_tree["text"] = str(decision_tree["attribute"]) + " < " + str(decision_tree["value"])
        depth1 = modify_tree(decision_tree["left"], depth+1)
        depth2 = modify_tree(decision_tree["right"], depth+1)
        if (depth1 > depth2):
            return depth1
        else:
            return depth2

def draw_diagram(decision_tree: dict, fig: plt.figure, final_depth: int, ax: plt.subplot, x_coord = 0.5):
    y_coord = 1 - ( ( (decision_tree["depth"] / final_depth) * squeeze_bounds ) + 0.1 )
    fontsize = 40 / (decision_tree["depth"]+1)

    fig.text(x_coord, y_coord, decision_tree["text"], transform=fig.transFigure, bbox={'facecolor': 'white', 'pad': 5}, ha='center', va='center', fontsize = fontsize)
    
    if "label" not in decision_tree.keys():
        next_x_modifier = 0.95 / (2 ** (decision_tree["depth"]+2) )
        next_y_modifier = ( (1 / final_depth) * squeeze_bounds )

        ax.plot([x_coord, x_coord - next_x_modifier],[y_coord, y_coord - next_y_modifier], color='blue', linewidth=2, transform=fig.transFigure, figure=fig)
        ax.plot([x_coord, x_coord + next_x_modifier],[y_coord, y_coord - next_y_modifier], color='green', linewidth=2, transform=fig.transFigure, figure=fig)

        draw_diagram(decision_tree["left"], fig, final_depth, ax, x_coord - next_x_modifier)
        draw_diagram(decision_tree["right"], fig, final_depth, ax, x_coord + next_x_modifier)
    

def print_tree(decision_tree: dict, name):
    
    fig = plt.figure()
    final_depth = modify_tree(decision_tree, 0)
    ax = fig.add_subplot()

    draw_diagram(decision_tree, fig, final_depth, ax)
    ax.set_position([0, 0, 1, 1])
    plt.savefig(f'fig_{name}.jpg')

print_tree(test_data, 'test_name')

