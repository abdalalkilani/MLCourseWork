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

