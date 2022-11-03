''' 
Description: file containing the functions needed to print out a visualisation of the binary decision tree

decision_tree_learning: - takes in a dataset as a matrix and a depth variable 
                        - outputs a decision tree of specified max depth as a dictionary
                        - 

'''

# from this import s
import numpy as np
import matplotlib.pyplot as plt
import json

final_tree_noisy = {'attribute': 0, 'value': -54.5, 'left': {'attribute': 4, 'value': -59.0, 'left': {'attribute': 3, 'value': -57.0, 'left': {'attribute': 2, 'value': -55.0, 'left': {'label': {1.0: 468}, 'depth': 0}, 'right': {'attribute': 5, 'value': -83.0, 'left': {'attribute': 0, 'value': -61.0, 'left': {'label': {1.0: 4}, 'depth': 0}, 'right': {'attribute': 1, 'value': -53.5, 'left': {'label': {4.0: 3}, 'depth': 0}, 'right': {'label': {1.0: 1}, 'depth': 0}}}, 'right': {'label': {1.0: 24}, 'depth': 0}}}, 'right': {'attribute': 0, 'value': -57.0, 'left': {'attribute': 2, 'value': -52.5, 'left': {'label': {1.0: 3}, 'depth': 0}, 'right': {'attribute': 0, 'value': -59.0, 'left': {'label': {3.0: 2}, 'depth': 0}, 'right': {'label': {4.0: 1}, 'depth': 0}}}, 'right': {'label': {3.0: 10}, 'depth': 0}}}, 'right': {'attribute': 4, 'value': -56.5, 'left': {'attribute': 3, 'value': -58.5, 'left': {'label': {4.0: 4}, 'depth': 0}, 'right': {'label': {3.0: 3}, 'depth': 0}}, 'right': {'label': {4.0: 489}, 'depth': 0}}}, 'right': {'attribute': 0, 'value': -44.0, 'left': {'attribute': 4, 'value': -70.0, 'left': {'attribute': 3, 'value': -49.0, 'left': {'attribute': 2, 'value': -57.0, 'left': {'attribute': 1, 'value': -60.5, 'left': {'label': {3.0: 2}, 'depth': 0}, 'right': {'label': {2.0: 2}, 'depth': 0}}, 'right': {'label': {3.0: 7}, 'depth': 0}}, 'right': {'attribute': 1, 'value': -58.5, 'left': {'attribute': 1, 'value': -59.5, 'left': {'label': {2.0: 5}, 'depth': 0}, 'right': {'label': {3.0: 2}, 'depth': 0}}, 'right': {'label': {2.0: 28}, 'depth': 0}}}, 'right': {'attribute': 5, 'value': -75.0, 'left': {'attribute': 4, 'value': -53.5, 'left': {'attribute': 1, 'value': -55.0, 'left': {'attribute': 0, 'value': -48.0, 'left': {'attribute': 6, 'value': -77.0, 'left': {'label': {3.0: 139}, 'depth': 0}, 'right': {'attribute': 2, 'value': -55.0, 'left': {'label': {2.0: 2}, 'depth': 0}, 'right': {'label': {3.0: 8}, 'depth': 0}}}, 'right': {'attribute': 5, 'value': -78.0, 'left': {'attribute': 6, 'value': -79.0, 'left': {'attribute': 3, 'value': -45.0, 'left': {'label': {3.0: 47}, 'depth': 0}, 'right': {'attribute': 1, 'value': -58.0, 'left': {'label': {2.0: 1}, 'depth': 0}, 'right': {'label': {3.0: 4}, 'depth': 0}}}, 'right': {'attribute': 3, 'value': -49.5, 'left': {'label': {3.0: 9}, 'depth': 0}, 'right': {'attribute': 2, 'value': -52.0, 'left': {'attribute': 5, 'value': -80.0, 'left': {'label': {2.0: 6}, 'depth': 0}, 'right': {'attribute': 0, 'value': -46.5, 'left': {'label': {2.0: 1}, 'depth': 0}, 'right': {'label': {3.0: 2}, 'depth': 0}}}, 'right': {'label': {3.0: 2}, 'depth': 0}}}}, 'right': {'attribute': 4, 'value': -65.5, 'left': {'label': {2.0: 5}, 'depth': 0}, 'right': {'attribute': 1, 'value': -55.5, 'left': {'label': {3.0: 3}, 'depth': 0}, 'right': {'label': {2.0: 1}, 'depth': 0}}}}}, 'right': {'label': {3.0: 248}, 'depth': 0}}, 'right': {'label': {4.0: 3}, 'depth': 0}}, 'right': {'attribute': 2, 'value': -55.0, 'left': {'label': {2.0: 9}, 'depth': 0}, 'right': {'attribute': 6, 'value': -76.5, 'left': {'attribute': 2, 'value': -55.0, 'left': {'label': {2.0: 1}, 'depth': 0}, 'right': {'label': {3.0: 8}, 'depth': 0}}, 'right': {'label': {2.0: 2}, 'depth': 0}}}}}, 'right': {'attribute': 3, 'value': -51.0, 'left': {'attribute': 0, 'value': -41.5, 'left': {'label': {3.0: 4}, 'depth': 0}, 'right': {'label': {2.0: 1}, 'depth': 0}}, 'right': {'label': {2.0: 436}, 'depth': 0}}}}

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
depth_sum = []

def modify_tree(decision_tree: dict, depth: int):

    if len(depth_sum) < depth+1:
        depth_sum.append(1)
    else:
        depth_sum[depth] += 1

    decision_tree["count"] = depth_sum[depth] - 1
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

def draw_diagram(decision_tree: dict, fig: plt.figure, final_depth: int, ax: plt.subplot, color='blue', parent_x = 0.0, parent_y = 0.0):
    y_coord = 1 - ( ( (decision_tree["depth"] / final_depth) * squeeze_bounds ) + 0.1 )
    x_coord = ( (decision_tree['count']+1) / (depth_sum[decision_tree['depth']]+1) )
    fontsize = 20

    fig.text(x_coord, y_coord, decision_tree["text"], transform=fig.transFigure, bbox={'facecolor': 'white', 'pad': 5}, ha='center', va='center', fontsize = fontsize)
    
    if parent_x != 0.0:
        ax.plot([x_coord, parent_x],[y_coord, parent_y], color=color, linewidth=2, transform=fig.transFigure, figure=fig)

    if "label" not in decision_tree.keys():
        draw_diagram(decision_tree["left"], fig, final_depth, ax, 'green', x_coord, y_coord)
        draw_diagram(decision_tree["right"], fig, final_depth, ax, 'blue', x_coord, y_coord)

def print_tree(decision_tree: dict, name):
    
    fig = plt.figure()
    final_depth = modify_tree(decision_tree, 0)

    ax = fig.add_subplot()

    draw_diagram(decision_tree, fig, final_depth, ax)
    ax.set_position([0, 0, 1, 1])
    fig.set_size_inches(40, 20)
    plt.savefig(f'fig_{name}.jpg', dpi=200)

if __name__ == '__main__':
    print_tree(final_tree_noisy, 'test_name')

