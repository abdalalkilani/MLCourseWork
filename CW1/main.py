import numpy as np
from postpruning import Pruning
from training import DecisionTreeBuilder
from training import decision_tree_learning, evaluate, other_metrics
from printing import print_tree

depth = 10

test_split = 0.1

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

rng = np.random.default_rng()

def split_data(dataset):
    classes = np.unique(dataset[:,-1])
    training, test = None, None
    for e in classes:
        tmp = dataset[dataset[:,-1]==e]
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
    import sys
    try: 
        path_to_data = sys.argv[1]
        dataset = read_data(path_to_data)

        try:
            best_depth = int(sys.argv[2])
            training, test = split_data(dataset)
            pre_pruning = decision_tree_learning(training, best_depth)
            pre_pruning_accuracy = evaluate(dict(pre_pruning), dataset, rng)
            print(f'accuracy before pruning: {pre_pruning_accuracy}')
            cmatrix, metrics = other_metrics(pre_pruning, dataset, rng)
            print(f'confusion matrix before pruning:\n{cmatrix}')
            print(f'precision: {metrics[0]}\nrecall:{metrics[1]}\nF1:{metrics[2]}')
            p = Pruning(dict(pre_pruning), test)
            post_pruning = dict(p.prune())
            training, test = split_data(dataset)
            post_pruning_accuracy = evaluate(dict(post_pruning), dataset, rng)
            if post_pruning_accuracy < pre_pruning_accuracy:
                print(f'accuracy after pruning: {pre_pruning_accuracy}')
                cmatrix, metrics = other_metrics(pre_pruning, dataset, rng)
                print(f'confusion matrix after pruning:\n{cmatrix}')
                print(f'precision:{metrics[0]}\nrecall:{metrics[1]}\nF1:{metrics[2]}')
                print_tree(dict(pre_pruning), f'post_pruning')
            else:
                print(f'accuracy after pruning: {post_pruning_accuracy}')
                cmatrix, metrics = other_metrics(post_pruning, dataset, rng)
                print(f'confusion matrix after pruning:\n{cmatrix}')
                print(f'precision:{metrics[0]}\nrecall:{metrics[1]}\nF1:{metrics[2]}')
                print_tree(dict(post_pruning), f'post_pruning')
                

        except IndexError:


            DTB = DecisionTreeBuilder(dataset)
            best_depth = DTB.find_optimal_depth(3, 18)
            training, test = split_data(dataset)
            pre_pruning = decision_tree_learning(training, best_depth)
            print(f'accuracy before pruning: {evaluate(dict(pre_pruning), dataset, rng)}')
            cmatrix, metrics = other_metrics(test, pre_pruning)
            print(f'confusion matrix before pruning:\n {cmatrix}')
            print(f'precision: {metrics[0]}\n recall: {metrics[1]}\n F1: {metrics[2]}')

            p = Pruning(dict(pre_pruning), test)
            post_pruning = dict(p.prune())
            # post_pruning = postpruning(pre_pruning, pre_pruning, "", test)
            # evaluation - cross validation

            print(f'accuracy after pruning: {evaluate(test, post_pruning)}')
            cmatrix, metrics = other_metrics(test, post_pruning)
            print(f'confusion matrix after pruning:\n {cmatrix}')
            print(f'precision: {metrics[0]}\n recall: {metrics[1]}\n F1: {metrics[2]}')
            
            # print_tree(dict(pre_pruning), f'pre_pruning_{type_}')
            print_tree(dict(post_pruning), f'post_pruning')
            # print_tree(dict(final_tree), f'final_tree')
    except IndexError:


    # import data
        dataset = [read_data('./intro2ML-coursework1/wifi_db/clean_dataset.txt'), read_data('./intro2ML-coursework1/wifi_db/noisy_dataset.txt')]


        # TO-DO: split the dataset into a training dataset ,an evaluation dataset and a test dataset
        for i, type_ in enumerate(['clean', 'noisy']):
            DTB = DecisionTreeBuilder(dataset[i])
            best_depth = DTB.find_optimal_depth(3, 18)
            training, test = split_data(dataset[i])
            pre_pruning = decision_tree_learning(training, best_depth)
            print(f'accuracy before pruning {type_}: {evaluate(test, dict(pre_pruning))}')
            # cmatrix, metrics = other_metrics(test, pre_pruning)
            # print(f'confusion matrix before pruning {type_}:\n {cmatrix}')
            # print(f'precision: {metrics[0]}\n recall: {metrics[1]}\n F1: {metrics[2]}')

            # pruning

            p = Pruning(dict(pre_pruning), test)
            post_pruning = dict(p.prune())
            # post_pruning = postpruning(pre_pruning, pre_pruning, "", test)
            # evaluation - cross validation

            print(f'accuracy after pruning {type_}: {evaluate(test, post_pruning)}')
            # cmatrix, metrics = other_metrics(test, post_pruning)
            # print(f'confusion matrix after pruning {type_}:\n {cmatrix}')
            # print(f'precision: {metrics[0]}\n recall: {metrics[1]}\n F1: {metrics[2]}')

            # print_tree(dict(pre_pruning), f'pre_pruning_{type_}')
            print_tree(dict(post_pruning), f'post_pruning_{type_}')
            # print_tree(dict(final_tree), f'final_tree_{type_}')