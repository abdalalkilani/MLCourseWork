from math import remainder
import numpy as np
import copy
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
def split_data(x, y, test_proportion, folds = 5, random_generator=default_rng()):

	shuffled_indices = random_generator.permutation(len(x))
	n_test = round(len(x) * test_proportion)
	n_train = len(x) - n_test
	x_train = x[shuffled_indices[:n_train]]
	y_train = y[shuffled_indices[:n_train]]
	x_test = x[shuffled_indices[n_train:]]
	y_test = y[shuffled_indices[n_train:]]
	return (x_train, x_test, y_train, y_test)



	return (x_train, x_test, y_train, y_test)

def cross_validation_split(x_dataset, y_dataset, folds=5):
	x_dataset_split = list()
	x_test_splits = list()
	x_train_splits = list()
	x_dataset_copy = list(x_dataset)
	fold_size = int(len(x_dataset) / folds)
	y_dataset_split = list()
	y_test_splits = list()
	y_train_splits = list()
	y_dataset_copy = list(y_dataset)


	for i in range(folds):
		x_fold = list()
		y_fold = list()
		while len(x_fold) < fold_size:
			index = randrange(len(x_dataset_copy))
			x_fold.append(x_dataset_copy.pop(index))
			y_fold.append(y_dataset_copy.pop(index))

		x_dataset_split.append(x_fold)
		x_test_splits.append(x_fold)
		y_dataset_split.append(y_fold)
		y_test_splits.append(y_fold)

		x_leftover = copy.deepcopy(x_dataset)
		y_leftover = copy.deepcopy(y_dataset)

		for j in range(fold_size):
			x_leftover.remove(x_fold[j])
			y_leftover.remove(y_fold[j])

		x_train_splits.append(x_leftover)
		y_train_splits.append(y_leftover)

	 
	return x_dataset_split, x_train_splits, y_dataset_split, y_train_splits

#testing k-fold cross splitting
seed(2)
x_dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
y_dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
x_test_folds, x_train_folds, y_test_folds, y_train_folds = cross_validation_split(x_dataset, y_dataset, 5)

print("x_test_folds", x_test_folds)
print("x_train_folds", x_train_folds)
print("y test_folds", y_test_folds)
print("y train_folds", y_train_folds)


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