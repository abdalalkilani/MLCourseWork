''' 
Description: file containing the functions necessary to train the decision trees

decision_tree_learning: - takes in a dataset as a matrix and a depth variable 
                        - outputs a decision tree of specified max depth as a dictionary
                        - 

'''

import numpy as np
from evaluation import evaluate

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
    left_node = decision_tree_learning(dataset[dataset[:,split_attribute].argsort()][:index], depth-1)
    right_node = decision_tree_learning(dataset[dataset[:,split_attribute].argsort()][index:], depth-1)
    return { 'attribute': split_attribute, 'value': split_value, 'left': left_node, 'right': right_node }

def predict(tree, x):
    y = []
    for row in x:
        if tree['depth'] == 0:
            y.append(tree['label'])
        elif row[tree['attribute']] < tree['value']:
            y.append(predict(tree['left'], row))
        else:
            y.append(predict(tree['right'], row))
    return np.array(y)

class DecisionTreeBuilder:
    def __init__(self, dataset, folds = 10):
        self.dataset = dataset
        self.folds = folds
        self.test_folds_used = []
        self.validation_folds_used = []
        self.split_data_by_label()

    def split_data_by_label(self):
        y = dataset[:,-1]
        labels = np.unique(y)
        data_by_label = []
        self.label_count = 0
        for e in labels:
            tmp = dataset[y==e].copy()
            rng = np.random.default_rng(12345)
            rng.shuffle(tmp)
            data_by_label.append(np.array_split(tmp, self.folds))
            self.label_count += 1
        self.data_by_label = np.array(data_by_label)
    
    def update_test_set(self):
        current_split = -1
        try:
            current_split = self.test_folds_used[-1]-1
        except:
            current_split = self.folds-1
        self.test_folds_used.append(current_split)
        test_set = self.data_by_label[0][current_split]
        rest_set = np.concatenate((self.data_by_label[0][:current_split], self.data_by_label[0][current_split+1:]))
        for i in range(1, self.label_count):
            test_set = np.concatenate((test_set, self.data_by_label[i][current_split]))
            rest_set = np.concatenate((rest_set, self.data_by_label[i][:current_split]))
            rest_set = np.concatenate((rest_set, self.data_by_label[i][current_split+1:]))

        self.current_test_set = test_set
        self.current_rest_set = rest_set
    
    def update_validation_set(self):
        current_split = -1
        try:
            current_split = self.validation_folds_used[-1]-1
        except:
            current_split = self.folds-1
        self.validation_folds_used.append(current_split)
        validation_set = self.data_by_label[0][current_split]
        train_set = np.concatenate((self.data_by_label[0][:current_split], self.data_by_label[0][current_split+1:]))
        for i in range(1, self.label_count):
            validation_set = np.concatenate((validation_set, self.data_by_label[i][current_split]))
            train_set = np.concatenate((train_set, self.data_by_label[i][:current_split]))
            train_set = np.concatenate((train_set, self.data_by_label[i][current_split+1:]))

        self.current_validation_set = validation_set
        self.current_train_set = np.reshape(train_set, (train_set.shape[0]*train_set.shape[1], train_set.shape[2]))
    
    def reset_validation_folds(self):
        self.validation_folds_used = []



    def find_optimal_depth(self, min_depth = 8, max_depth = 8):
        accuracy_map = map()
        for k in range(self.folds):
            self.update_test_set()
            for kk in range(self.folds-1):
                for depth in range(min_depth, max_depth+1):
                    tree = decision_tree_learning(self.dataset, depth)
                    accuracy = evaluate(tree)
                    try:
                        accuracy_map[depth] += [accuracy]
                    except KeyError:
                        accuracy_map[depth] = [accuracy]
                self.update_validation_set()
            self.reset_validation_folds()
        best_accuracy, best_depth = 0, 0
        for depth, accuracy_list in accuracy_map.items():
            average_accuracy = np.average(accuracy_list)
            if average_accuracy > best_accuracy:
                best_accuracy, best_depth = average_accuracy, depth
        return best_depth

    
    
    
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
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
    dataset = read_data('./intro2ML-coursework1/wifi_db/clean_dataset.txt')

    print(decision_tree_learning(dataset, 3))
    # cv = DecisionTreeBuilder(dataset)
    
    # print(cv.find_optimal_depth())
