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
    max_information_gain, attribute, value, index = None, 0, 0, 0
    for j in range(dataset.shape[1]-1):
        single_attribute_array = dataset[dataset[:, j].argsort()][:, [j,-1]]
        for i in range(1, single_attribute_array.shape[0]):
            left_dataset = single_attribute_array[:i]
            right_dataset = single_attribute_array[i:]
            gain = information_gain(dataset, left_dataset, right_dataset)
            if type(max_information_gain) == type(None) or max_information_gain < gain:
                max_information_gain, attribute, value, index = gain, j, (left_dataset[-1][0]+right_dataset[0][0])/2, i
    return attribute, value, index


def decision_tree_learning(dataset, depth):
    if check_all_samples(dataset):
        return { 'label': {dataset[0][-1]: len(dataset)}, 'depth': 0 }
    if depth==1:
        labels = [e[-1] for e in dataset]
        attributes = set(labels)
        attribute_count = {}
        for e in attributes:
            attribute_count[e] = labels.count(e)
        return { 'label': attribute_count, 'depth': 0}
    
    split_attribute, split_value, index = find_split(dataset)
    left_node = decision_tree_learning(dataset[dataset[:,split_attribute].argsort()][:index], depth-1)
    right_node = decision_tree_learning(dataset[dataset[:,split_attribute].argsort()][index:], depth-1)
    return { 'attribute': split_attribute, 'value': split_value, 'left': left_node, 'right': right_node }

def predict(tree, x):
    y = []
    for row in x:
            y.append(predict_row(tree, row))
    return np.array(y).astype(float)

def predict_row(tree, row):
    try:
        depth = tree['depth']
        if(depth == 0):
            return max(tree['label'], key = tree['label'].get)
    except KeyError:
        if(row[tree['attribute']] < tree['value']):
            return predict_row(tree['left'], row)
        else:
            return predict_row(tree['right'], row)
class DecisionTreeBuilder:
    def __init__(self, dataset, folds = 10):
        self.dataset = dataset
        self.folds = folds
        self.test_folds_used = []
        self.validation_folds_used = []
        self.split_data_by_label()

    def split_data_by_label(self):
        y = self.dataset[:,-1]
        labels = np.unique(y)
        data_by_label = []
        self.label_count = 0
        tmp = self.dataset
        rng = np.random.default_rng(12345)
        rng.shuffle(tmp)
        for e in labels:
        #     tmp = self.dataset[y==e].copy()
        #     data_by_label.append(np.array_split(tmp, self.folds))
            self.label_count += 1
        self.data_by_label = np.array(np.array_split(tmp, self.folds))
    
    def update_test_set(self):
        current_split = -1
        try:
            current_split = self.test_folds_used[-1]-1
        except:
            current_split = self.folds-1
        self.test_folds_used.append(current_split)
        test_set = self.data_by_label[current_split]
        rest_set = np.concatenate((self.data_by_label[:current_split], self.data_by_label[current_split+1:]))
        # for i in range(1, self.label_count):
        #     test_set = np.concatenate((test_set, self.data_by_label[i][current_split]))
        #     rest_set = np.concatenate((rest_set, self.data_by_label[i][:current_split]))
        #     rest_set = np.concatenate((rest_set, self.data_by_label[i][current_split+1:]))

        self.current_test_set = test_set
        self.current_rest_set = rest_set
    
    def update_validation_set(self):
        current_split = -1
        try:
            current_split = self.validation_folds_used[-1]-1
        except:
            current_split = self.folds-1
        self.validation_folds_used.append(current_split)
        validation_set = self.data_by_label[current_split]
        train_set = np.concatenate((self.data_by_label[:current_split], self.data_by_label[current_split+1:]))
        # for i in range(1, self.label_count):
        #     validation_set = np.concatenate((validation_set, self.data_by_label[i][current_split]))
        #     train_set = np.concatenate((train_set, self.data_by_label[i][:current_split]))
        #     train_set = np.concatenate((train_set, self.data_by_label[i][current_split+1:]))

        self.current_validation_set = validation_set
        self.current_train_set = np.reshape(train_set, (train_set.shape[0]*train_set.shape[1], train_set.shape[2]))
        # print(self.current_validation_set.shape)
    
    def reset_validation_folds(self):
        self.validation_folds_used = []



    def find_optimal_depth(self, min_depth = 8, max_depth = 8):
        accuracy_map = {}
        for k in range(self.folds):
            self.update_test_set()
            for kk in range(self.folds-1):
                self.update_validation_set()
                for depth in range(min_depth, max_depth+1):
                    tree = decision_tree_learning(self.current_train_set, depth)
                    accuracy = evaluate(self.current_validation_set, tree)
                    try:
                        accuracy_map[depth] += [accuracy]
                    except KeyError:
                        accuracy_map[depth] = [accuracy]
            self.reset_validation_folds()
        best_accuracy, best_depth = 0, 0
        for depth, accuracy_list in accuracy_map.items():
            average_accuracy = np.average(accuracy_list)
            if average_accuracy > best_accuracy:
                best_accuracy, best_depth = average_accuracy, depth
        return best_depth

# returns accuracy
def evaluate(test_db, trained_tree):
    x_test = test_db[:,:-1]
    y_test = test_db[:,-1]
    y_predict = predict(trained_tree, x_test)

    assert len(y_test) == len(y_predict)

    try:
        return np.sum(y_predict == y_test) / len(y_test)
    except ZeroDivisionError:
        return 0

# returns cmatrix and other metrics
def other_metrics(test_db, trained_tree):
    
    x_test = test_db[:,:-1]
    y_test = test_db[:,-1]

    y_predict = predict(trained_tree, x_test)
    assert len(y_test) == len(y_predict)

    allclasses = np.unique(y_test)
    classes = len(allclasses)
    cmatrix = np.zeros((classes, classes))
    #j is predicted, i is actual
    for i in range(classes):
        for j in range(classes):
            cmatrix[i, j] = np.sum((y_test==allclasses[i]) & (y_predict==allclasses[j]))

    # three rows for precision, recall, f1
    metrics = np.zeros((3, classes))
    for i in range(classes):
        if np.sum(cmatrix[:,i]) > 0:
            metrics[0,i] = cmatrix[i,i] / np.sum(cmatrix[:,i])
        if np.sum(cmatrix[i,:]) > 0:
            metrics[1,i] = cmatrix[i,i] / np.sum(cmatrix[i,:])
        if ((metrics[0,i]+metrics[1,i]) > 0):
            metrics[2,i] = (2*metrics[0,i]*metrics[1,i]) / (metrics[0,i]+metrics[1,i])
    
    return cmatrix, metrics

    
    
    
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
