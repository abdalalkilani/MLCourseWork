''' 
Description: file containing the functions necessary to train the decision trees

decision_tree_learning: - takes in a dataset as a matrix and a depth variable 
                        - outputs a decision tree of specified max depth as a dictionary
                        - 

'''

def check_all_samples(dataset):
    sample = dataset[0][-1]
    for data in dataset:
        if data[-1] != sample:
            return False
    return True

def decision_tree_learning(dataset, depth):
    if check_all_samples(dataset):
        return { 'label': dataset[0][-1], 'depth': 0 }
    
    
    