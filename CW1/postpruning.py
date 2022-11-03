from tkinter import W
import numpy as np
from printing import print_tree

from training import decision_tree_learning, evaluate

import json
import sys


test_data ={'attribute': 0, 'value': -54.5, 'left': {'attribute': 4, 'value': -59.0, 'left': {'attribute': 3, 'value': -57.0, 'left': {'attribute': 2, 'value': -54.0, 'left': {'label': {1.0: 436}, 'depth': 0}, 'right': {'attribute': 6, 'value': -85.5, 'left': {'attribute': 0, 'value': -60.5, 'left': {'label': {1.0: 1}, 'depth': 0}, 'right': {'label': {4.0: 2}, 'depth': 0}}, 'right': {'label': {1.0: 11}, 'depth': 0}}}, 'right': {'attribute': 2, 'value': -56.0, 'left': {'attribute': 0, 'value': -56.0, 'left': {'label': {1.0: 2}, 'depth': 0}, 'right': {'label': {3.0: 2}, 'depth': 0}}, 'right': {'label': {3.0: 11}, 'depth': 0}}}, 'right': {'attribute': 4, 'value': -56.5, 'left': {'attribute': 0, 'value': -56.0, 'left': {'label': {4.0: 4}, 'depth': 0}, 'right': {'label': {3.0: 2}, 'depth': 0}}, 'right': {'label': {4.0: 441}, 'depth': 0}}}, 'right': {'attribute': 0, 'value': -44.5, 'left': {'attribute': 4, 'value': -70.0, 'left': {'attribute': 3, 'value': -50.0, 'left': {'label': {3.0: 7}, 'depth': 0}, 'right': {'attribute': 3, 'value': -45.0, 'left': {'attribute': 1, 'value': -56.0, 'left': {'attribute': 2, 'value': -55.5, 'left': {'attribute': 0, 'value': -50.5, 'left': {'label': {3.0: 1}, 'depth': 0}, 'right': {'label': {2.0: 3}, 'depth': 0}}, 'right': {'label': {3.0: 3}, 'depth': 0}}, 'right': {'label': {2.0: 5}, 'depth': 0}}, 'right': {'label': {2.0: 23}, 'depth': 0}}}, 'right': {'attribute': 3, 'value': -39.5, 'left': {'attribute': 4, 'value': -53.5, 'left': {'attribute': 2, 'value': -54.0, 'left': {'attribute': 6, 'value': -77.0, 'left': {'attribute': 2, 'value': -55.0, 'left': {'label': {3.0: 79}, 'depth': 0}, 'right': {'attribute': 2, 'value': -55.0, 'left': {'attribute': 0, 'value': -47.0, 'left': {'label': {2.0: 1, 3.0: 3}, 'depth': 0}, 'right': {'label': {2.0: 3}, 'depth': 0}}, 'right': {'attribute': 5, 'value': -77.5, 'left': {'label': {3.0: 31}, 'depth': 0}, 'right': {'label': {2.0: 2, 3.0: 3}, 'depth': 0}}}}, 'right': {'attribute': 1, 'value': -54.0, 'left': {'attribute': 1, 'value': -60.5, 'left': {'label': {3.0: 2}, 'depth': 0}, 'right': {'label': {2.0: 8}, 'depth': 0}}, 'right': {'label': {3.0: 6}, 'depth': 0}}}, 'right': {'attribute': 6, 'value': -72.0, 'left': {'label': {3.0: 294}, 'depth': 0}, 'right': {'label': {2.0: 1}, 'depth': 0}}}, 'right': {'label': {4.0: 3}, 'depth': 0}}, 'right': {'label': {2.0: 7}, 'depth': 0}}}, 'right': {'attribute': 3, 'value': -48.0, 'left': {'attribute': 0, 'value': -42.0, 'left': {'attribute': 2, 'value': -56.0, 'left': {'label': {2.0: 1}, 'depth': 0}, 'right': {'label': {3.0: 6}, 'depth': 0}}, 'right': {'label': {2.0: 9}, 'depth': 0}}, 'right': {'label': {2.0: 387}, 'depth': 0}}}}

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

def replace(leaf_to_add, tree_being_replaced, current_subtree):
    try:
        label = current_subtree['label']
        # print('label')
        # if(tree_being_replaced == current_subtree):
        #     return leaf_to_add
        return current_subtree
    except KeyError:
        if(tree_being_replaced == current_subtree):
            return leaf_to_add
        else:
            current_subtree['left'] = replace(leaf_to_add, tree_being_replaced, dict(current_subtree['left']))
            current_subtree['right'] = replace(leaf_to_add, tree_being_replaced, dict(current_subtree['right']))
            return current_subtree


def postpruning(built_tree, whole_tree, path_string, validation_set):

    right_node = built_tree['right']
    left_node = built_tree['left']

    # old_tree =  copy.deepcopy(whole_tree)
    old_tree = dict(whole_tree)
    if(built_tree==whole_tree):
        built_tree = whole_tree

    # print(id(built_tree))
    # print(id(whole_tree))
    initial = 0
    current_accuracy = evaluate(validation_set, old_tree)
    # current_accuracy = random.uniform(0, 1)
   

    while ((old_tree != whole_tree) or (initial == 0)):
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
                label = {i: left_node['label'].get(i, 0) + right_node['label'].get(i, 0) 
                            for i in set(left_node['label']).union(right_node['label'])}
                # new_tree = { 'label': {},
                #              'depth': 0}

                new_tree = { 'label': {},
                             'depth': 'AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH'}
                new_tree['label'] = label

                print(new_tree)
                print(built_tree)
                print(old_tree == whole_tree)
                print(whole_tree)
                print(id(whole_tree))
                # result1 = any('AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH' in d.values() for d in whole_tree.values())
                whole_tree = replace(new_tree, built_tree, dict(whole_tree))
                print(id(whole_tree))
                print("")
                print(whole_tree)
                print(old_tree == whole_tree)
                # sys.exit("Error message")
                # print(new_tree)
                # print(built_tree)
                # print(json.dumps(whole_tree, indent=4))
                new_accuracy = evaluate(validation_set, whole_tree)
                # result2 = any('AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH' in d.values() for d in whole_tree.values())
                # if result2 > result1:
                #     sys.exit("Error message")
                print(new_accuracy)
                print(current_accuracy)
                # new_accuracy = random.uniform(0, 1)
                
                if new_accuracy >= current_accuracy:
                    # print("New Tree")
                    # print(json.dumps(whole_tree, indent=4))
                    # print_tree(whole_tree)
                    # or is next line
                    # built_tree =  copy.deepcopy(new_tree)
                    # old_tree = dict(whole_tree)
                    current_accuracy = new_accuracy
                    return new_tree, dict(whole_tree)
                    # sys.exit("Error message1")
                else:
                    whole_tree = dict(old_tree)
                    print("YO", old_tree == whole_tree)
                    return built_tree, dict(whole_tree)
                    # sys.exit("Error message2")
            except KeyError:
                # path_string += "R"
                right_node, whole_tree = postpruning(right_node, whole_tree, path_string, validation_set)

        except KeyError:
            try:
                right_depth = right_node['depth']
                # path_string += "L"
                left_node, whole_tree = postpruning(left_node, whole_tree, path_string, validation_set)
            except KeyError:
                # path_string += "L"
                left_node, whole_tree = postpruning(left_node, whole_tree, path_string, validation_set)
                # path_string = path_string[:-1] + "R"
                right_node, whole_tree = postpruning(right_node, whole_tree, path_string, validation_set)

    print("WHOLE TREEEEEEEEEEEEEEEEEEE")
    print(whole_tree)
    return whole_tree, whole_tree

if __name__ == '__main__':

    def read_data(file_name):
        dataset = []
        for line in open(file_name):
            if line.strip() != "": # handle empty rows in file
                try:
                    row = line.strip().split("\t")
                    dataset.append(list(map(float, row))) 
                except ValueError:
                    row = line.strip().split(" ")
                    dataset.append(list(map(float, row)))
                # dataset.append(line[:-1].split('\t'))
        return np.array(dataset)

    def split_data(dataset):
        classes = np.unique(dataset[:,-1])
        training, test = None, None
        for e in classes:
            tmp = dataset[dataset[:,-1]==e]
            rng = np.random.default_rng(12345)
            rng.shuffle(tmp)
            if(type(training)==type(None)):
                training = tmp[:int(0.9*len(tmp))]
            else:
                training = np.concatenate((training, tmp[:int(0.9*len(tmp))]))
            if(type(test)==type(None)):
                test = tmp[int(0.9*len(tmp)):]
            else:
                test = np.concatenate((test, tmp[int(0.9*len(tmp)):]))
        return training, test

    print(postpruning(dict(test_data), dict(test_data), "", split_data(read_data('./intro2ML-coursework1/wifi_db/noisy_dataset.txt'))[1])==test_data)

# top_tree = {'left': { 'left': { 'label': True }, 'right':{ 'label': True }} , 'right': { 'label': True }}
# built_node = top_tree['left']

# left_node = {'left': {'label': False}}

#TESTING REPLACE - IT WORKS
# left_node = {'label': {1.0: 1, 4.0: 2}, 'depth': 'AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH'}
# built_node = {'attribute': 0, 'value': -60.5, 'left': {'label': {1.0: 1}, 'depth': 0}, 'right': {'label': {4.0: 2}, 'depth': 0}}
# print(replace(left_node, built_node, dict(test_data)))
