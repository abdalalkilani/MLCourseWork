from tkinter import W
import numpy as np
from printing import print_tree
from training import decision_tree_learning, evaluate, other_metrics

import json
import sys

# function to facilitate a leaf replacement for pruning
def replace(leaf_to_add, tree_being_replaced, current_subtree):
    try:
        label = current_subtree['label'] # testing for leaf
        return current_subtree
    except KeyError:
        if(tree_being_replaced == current_subtree):
            return leaf_to_add
        else:
            current_subtree['left'] = replace(leaf_to_add, tree_being_replaced, dict(current_subtree['left']))
            current_subtree['right'] = replace(leaf_to_add, tree_being_replaced, dict(current_subtree['right']))
            return current_subtree



class Pruning:
    def __init__(self, tree_to_be_pruned, dataset):
        self.current_best_tree = tree_to_be_pruned
        self.current_best_accuracy = evaluate(dataset, tree_to_be_pruned)
        self.dataset = dataset
        self.changed_tree = False


    def traverse_and_change(self, tree):
        left_node = tree['left']
        right_node = tree['right']
        try:
            left_depth = left_node['depth'] # testing for leaf
            # assert(left_depth==0)
            try:
                right_depth = right_node['depth'] # testing for leaf
                # assert(right_depth == 0)
                label = {i: left_node['label'].get(i, 0) + right_node['label'].get(i, 0) 
                        for i in set(left_node['label']).union(right_node['label'])}
                new_leaf = {'label': label, 'depth': 0}
                current_tree_to_test = dict(replace(new_leaf, tree, dict(self.current_best_tree)))
                new_accuracy = evaluate(self.dataset, current_tree_to_test)
                if(new_accuracy >= self.current_best_accuracy):
                    self.current_best_tree = dict(current_tree_to_test)
                    self.current_best_accuracy = new_accuracy
                    self.changed_tree = True

            except KeyError:
                self.traverse_and_change(right_node)
        except KeyError:
            try:
                right_depth = right_node['depth'] # testing for leaf
                # assert(right_depth == 0)
                self.traverse_and_change(left_node)
            except KeyError:
                self.traverse_and_change(left_node)
                self.traverse_and_change(right_node)

    
    def prune(self):
        self.changed_tree = False
        self.traverse_and_change(self.current_best_tree)
        if(self.changed_tree):
            self.prune()
        return self.current_best_tree

if __name__ == '__main__':
     
    from training import decision_tree_learning

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
    
    tree_depth = 11
    training_set, test_set = split_data(read_data('./intro2ML-coursework1/wifi_db/noisy_dataset.txt'))
    test_data = decision_tree_learning(training_set, tree_depth)
    prune = Pruning(test_data, test_set)
    post_pruning = prune.prune()
    test = split_data(read_data('./intro2ML-coursework1/wifi_db/noisy_dataset.txt'))[1]
    print(post_pruning)
    print(f'accuracy: {evaluate(test, post_pruning)}')
    print(f'other metrics: {other_metrics(test, post_pruning)}')


