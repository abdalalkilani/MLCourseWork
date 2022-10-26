''' 
Description: file containing the functions necessary to train the decision trees

decision_tree_learning: - takes in a dataset as a matrix and a depth variable 
                        - outputs a decision tree of specified max depth as a dictionary
                        - 

'''

import numpy as np

def check_all_samples(dataset):
    sample = dataset[0][-1]
    for data in dataset:
        if data[-1] != sample:
            return False
    return True

def entropy(dataset):
    labels = [e[-1] for e in dataset]
    attributes = set(labels)
    attribute_count = {}
    for e in attributes:
        attribute_count[e] = labels.count(e)
    sum = 0
    for value in attribute_count.values():
        p = value/len(dataset)
        sum += p*np.log2(p)
    return -1 * sum

def information_gain(dataset, left_subset, right_subset):
    return (entropy(dataset) - (len(left_subset)/len(dataset))*entropy(left_subset) - (len(right_subset)/len(dataset))*entropy(right_subset))


def find_split(dataset):
    max_information_gain, attribute, value, index = 0, 0, 0, 0
    for j in range(dataset.shape[1]-1):
        single_attribute_array = dataset[dataset[:, j].argsort()][:, [j,-1]]
        for i in range(1, single_attribute_array.shape[0]):
            left_dataset = single_attribute_array[:i]
            right_dataset = single_attribute_array[i:]
            gain = information_gain(dataset, left_dataset, right_dataset)
            if max_information_gain < gain:
                max_information_gain, attribute, value, index = gain, j, (left_dataset[-1][0]+right_dataset[0][0])/2, i
    return attribute, value, index
            




def decision_tree_learning(dataset, depth):
    if check_all_samples(dataset):
        return { 'label': dataset[0][-1], 'depth': 0 }
    if depth==1:
        labels = [e[-1] for e in dataset]
        attributes = set(labels)
        attribute_count = {}
        for e in attributes:
            attribute_count[e] = labels.count(e)
        return { 'label': max(attribute_count, key = attribute_count.get), 'depth': 0}
    
    split_attribute, split_value, index = find_split(dataset)
    left_node = decision_tree_learning(dataset[:index], depth-1)
    right_node = decision_tree_learning(dataset[index:], depth-1)
    return { 'attribute': split_attribute, 'value': split_value, 'left': left_node, 'right': right_node }

def predict(tree, x):
    y = []
    for row in x:
        if tree['depth'] == 0:
            y.append(tree['label'])
        elif row[tree['attribute']] >= tree['value']:
            y.append(predict(tree['left'], row))
        else:
            y.append(predict(tree['right'], row))
    return np.array(y)

    
    
    
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    with open('./intro2ML-coursework1/wifi_db/clean_dataset.txt') as file:
        dataset = []
        for line in file:
            dataset.append(line[:-1].split('\t'))
        dataset = np.array(dataset).astype(int)
        print(decision_tree_learning(dataset, 3))