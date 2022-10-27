import numpy as np
from training import predict

# 10-fold cross-validation on both datasets
    # - must return accuracy of tree

def evaluate(test_db, trained_tree):
    x_test = test_db[:-1]
    y_test = test_db[-1]

    y_predict = predict(x_test)

    assert len(y_test) == len(y_predict)

    try:
        return np.sum(y_predict == y_test) / len(y_test)
    except ZeroDivisionError:
        return 0

def other_metrics(test_db, trained_tree):
    
    x_test = test_db[:-1]
    y_test = test_db[-1]

    y_predict = predict(x_test)
    assert len(y_test) == len(y_predict)

    allclasses = np.unique(y_test)
    classes = len(allclasses)
    cmatrix = np.zeroes(classes, classes)
    #j is predicted, i is actual
    for i in range(classes):
        for j in range(classes):
            cmatrix[i, j] = np.sum((y_test==allclasses[i]) & (y_predict==allclasses[j]))

    # three rows for precision, recall, f1
    metrics = np.zeroes(classes, 3)
    for i in range(classes):
        metrics[0,i] = cmatrix[i,i] / np.sum(cmatrix[:,i])
        metrics[1,i] = cmatrix[i,i] / np.sum(cmatrix[i,:])
        metrics[2,i] = (2*metrics[0,i]*metrics[1,i]) / (metrics[0,i]+metrics[1,i])
    
    return cmatrix, metrics