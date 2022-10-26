import numpy as np
from training import decision_tree_learning, predict

depth = 10

if __name__ == '__name__':
    # import data
    with open('./intro2ML-coursework1/wifi_db/clean_dataset.txt') as file:
        dataset = []
        for line in file:
            dataset.append(line[:-1].split('\t'))
        dataset = np.array(dataset).astype(int)

    # tree building by calling decision_tree_training()
    tree = decision_tree_learning(dataset, depth)

    # pruning

    # evaluation