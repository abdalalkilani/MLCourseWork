import numpy as np
from training import decision_tree_learning, predict
from numpy.random import default_rng
from random import seed
from random import randrange

depth = 10

test_split = 0.2

#function from labs
def read_data(file_name):
    x = []
    y_labels = []
    for line in open(file_name):
        if line.strip() != "": # handle empty rows in file
            row = line.strip().split(" ")
            x.append(list(map(float, row[:-1]))) 
            y_labels.append(row[-1])
    
    [classes, y] = np.unique(y_labels, return_inverse=True) 

    x = np.array(x)
    y = np.array(y)
    return (x, y, classes)

# REMEMBER our final tree will not split the data and will be trained with all available data
# splitting data is only for evaluation purposes
def split_data(x, y, test_proportion, random_generator=default_rng()):

    shuffled_indices = random_generator.permutation(len(x))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test
    x_train = x[shuffled_indices[:n_train]]
    y_train = y[shuffled_indices[:n_train]]
    x_test = x[shuffled_indices[n_train:]]
    y_test = y[shuffled_indices[n_train:]]
    return (x_train, x_test, y_train, y_test)

def cross_validation_split(dataset, folds=5):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
folds = cross_validation_split(dataset, 5)
print(folds)

if __name__ == '__name__':
    # import data
    with open('./intro2ML-coursework1/wifi_db/clean_dataset.txt') as file:
        dataset = []
        for line in file:
            dataset.append(line[:-1].split('\t'))
        dataset = np.array(dataset).astype(int)

    # TO-DO: split the dataset into a training dataset ,an evaluation dataset and a test dataset

    # tree building by calling decision_tree_training()
    tree = decision_tree_learning(dataset, depth)

    # pruning

    # evaluation - cross validation