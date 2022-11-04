## Producing evaluation metrics
### Producing evaluation metrics without a depth requirement

**Warning: running this will take a very very long time (approximately 10 hours for a dataset of size 2000x7)**
The reason behind this is that we treat the maximum depth of the tree as a hyperparamter and tune it using 10 fold cross-validation. While this does produce accurate results, it does take a lot of time to run.

To do this: 
1. Ensure that you are in the directory containing the `main.py` file
2. Run one of the following commands:
    - `python3 main.py path/to/data` where `path/to/data` is the path to the dataset you want to train
    - `python3 main.py` if you want to train trees on both datasets given in the specs (Beware this takes ~17 hours to run)

This will print into terminal the accuracy of the tree, the confusion matrix as well as the precision, recall and F1-measure (this will be done twice if you are training both datasets given). This will also save a jpg image named `fig_post_pruning.jpg` (and 2 images in case you train on both datasets)

### Producing evaluation metrics with a depth requirement (strongly recommended)
Here, you have to specify the depth you want your tree to be. This will bypass the hyperparameter tuning, significantly reducing runtime.
To do this:
1. Ensure that you are in the directory containing the `main.py` file
2. Run the command: `python3 main.py path/to/data depth` where `path/to/data` is the path to the dataset you want to train and `depth` is the desired maximum depth of the tree you are training

This will print into terminal the accuracy of the tree, the confusion matrix as well as the precision, recall and F1-measure. This will also save a jpg image named `fig_post_pruning.jpg`.

**If you wish to obtain the results for the given datasets without waiting 17 hours:**
- Clean dataset: run the command `python3 main.py ./intro2ML-coursework1/wifi_db/clean_dataset.txt 8`
- Noisy dataset: run the command `python3 main.py ./intro2ML-coursework1/wifi_db/noisy_dataset.txt 11`

## Using individual functions

### Loading data and seperating it
The following code snippet shows how to import data and how to split it into a training and a test set. `dataset` is a numpy array of dimension (number_of_rows_in_data, number_of_columns_in_data). Both `training_set` and `test_set` are also numpy arrays following a 90/10 split.
```python
from main import read_data, split_data

dataset = read_data(path_to_data) # reads the data
training_set, test_set = split_data(dataset) # splits the data into a training set and a test set
```

### Finding the optimal depth for a tree
The following code snippet shows how to tune the depth paramter for your tree. `DTB.find_optimal_depth` will try every integer between `min_depth` and `max_depth` and returns the depth for which the average accuracy over all cross-validation data is the largest. Note: `dataset` is the whole dataset, not simply the training dataset.

```python
from main import DecisionTreeBuilder

DTB = DecisionTreeBuilder(dataset)
best_depth = DTB.find_optimal_depth(min_depth, max_depth)
```

### Building a tree
The following code snippeet show how to build a tree once you have a desired `maximum_depth` for the tree. The `decision_tree_learning` function will return a dictionary.

```python
from main import decision_tree_learning

tree = decision_tree_learning(training_set, maximum_depth)
```

### Pruning a tree
The following code snippet shows how to prune a tree once you have a tree. Here, `tree` is the tree you want to be pruned and `pruned_tree` is the tree after being pruned. Note: we use dict() to make copies of the tree as to not accidentaly modify the tree.

```python
from main import Pruning

p = Pruning(dict(tree), test_set)
pruned_tree = dict(p.prune())
```

### Getting evaluation metrics
The following code snippet shows how to obtain evaluation metrics for your tree.

```python
from main import evaluate, other_metrics

accuracy = evaluate(test_set, tree)
confusion_matrix, others = other_metrics(test_set, tree)

```