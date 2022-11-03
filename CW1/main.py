import numpy as np
from postpruning import Pruning
from training import DecisionTreeBuilder
from training import decision_tree_learning, evaluate, other_metrics
from random import seed
from random import randrange
from printing import print_tree

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
    for i, type_ in enumerate(['noisy']):
        DTB = DecisionTreeBuilder(dataset[1]) # CHANGE INDEX LATER
        # best_depth = DTB.find_optimal_depth(15, 15)
        training, test = split_data(dataset[1]) # CHANGE INDEX LATER
        pre_pruning = decision_tree_learning(training, 15) #CHANGE THIS LATER
        # print(f'pre pruning tree: {pre_pruning}')
        p = Pruning(dict(pre_pruning), test)

        # pruning

        post_pruning = dict(p.prune())
        # post_pruning = postpruning(pre_pruning, pre_pruning, "", test)
        print(f'post pruning tree: {post_pruning}')
        # evaluation - cross validation

        print(f'accuracy {type_}: {evaluate(test, post_pruning)}')
        print(f'metrics {type_}: {other_metrics(test, post_pruning)}')

        print(f'final_tree {type_}: {decision_tree_learning(dataset[0], best_depth)}')
        print_tree(dict(pre_pruning), f'pre_pruning_{type_}')
        print_tree(dict(post_pruning), f'post_pruning_{type_}')