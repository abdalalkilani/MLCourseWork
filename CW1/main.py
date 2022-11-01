import numpy as np
from postpruning import postpruning
from training import DecisionTreeBuilder
from training import decision_tree_learning
from random import seed
from random import randrange

depth = 10

test_split = 0.2

#imports the data file as shown from function in labs
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
        

# REMEMBER our final tree will not split the data and will be trained with all available data
# splitting data is only for evaluation purposes
# def split_data(x, y, test_proportion, folds = 10, random_generator=default_rng()):
#     shuffled_indices = random_generator.permutation(len(x))
#     n_test = round(len(x) * test_proportion)
#     n_train = len(x) - n_test
#     x_train = x[shuffled_indices[:n_train]]
#     y_train = y[shuffled_indices[:n_train]]
#     x_test = x[shuffled_indices[n_train:]]
#     y_test = y[shuffled_indices[n_train:]]

#     return (x_train, x_test, y_train, y_test)

#Splits data into a test and training set using k-fold cross validation
#This is in order not to bias the evaluation based on what goes into test and what goes into training data
#We will take the mean for the evaluation matrix for all k folds to determine our true evaluation metrics
def XYcross_validation_split(x_dataset, y_dataset, folds=5):
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

def cross_validation_split(dataset, folds=10):
    dataset_split = list()
    test_splits = list()
    train_splits = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)



    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))

        
        test_splits.append(fold)

        leftover = list()

        for row in dataset:
            if not row in fold:
                leftover.append(row)
            

        train_splits.append(leftover)

     
    return dataset_split, train_splits

#testing k-fold cross splitting
#seed in order to keep random splits consistent while changing code

# seed(2)
# dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
# y_dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
# x_test_folds, x_train_folds, y_test_folds, y_train_folds = cross_validation_split(dataset, 10)

# print("x_test_folds", x_test_folds)
# print("x_train_folds", x_train_folds)
# print("y test_folds", y_test_folds)
# print("y train_folds", y_train_folds)

class TreeBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        
    
    def get_optimal_hyperparameters(self):
        pass

class CrossValidation:
    def __init__(self, dataset, folds = 10):
        if(self.initialised):
            return
        self.initialised = True
        self.dataset = dataset
        self.folds = folds
    
    def split_data_by_label(self):
        [classes, y] = np.unique(dataset[:-1])

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





if __name__ == '__main__':
    # import data
    np.set_printoptions(threshold=np.inf)
    dataset = [read_data('./intro2ML-coursework1/wifi_db/clean_dataset.txt'), read_data('./intro2ML-coursework1/wifi_db/noisy_dataset.txt')]


    # TO-DO: split the dataset into a training dataset ,an evaluation dataset and a test dataset

    DTB = DecisionTreeBuilder(dataset[0])
    best_depth = DTB.find_optimal_depth(8,8)
    pre_pruning = decision_tree_learning(split_data(dataset[0])[0])
    print(pre_pruning)

    # pruning

    post_pruning = postpruning(pre_pruning)
    print(post_pruning)

    # evaluation - cross validation