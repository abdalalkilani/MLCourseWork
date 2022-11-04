# Producing evaluation metrics
## Producing evaluation metrics without a depth requirement

**Warning: running this will take a very very long time (approximately 10 hours for a dataset of size 2000x7)**
The reason behind this is that we treat the maximum depth of the tree as a hyperparamter and tune it using 10 fold cross-validation. While this does produce accurate results, it does take a lot of time to run.

To do this: 
1. Ensure that you are in the directory containing the `main.py` file
2. Run one of the following commands:
    - `python3 main.py path/to/data` where `path/to/data` is the path to the dataset you want to train
    - `python3 main.py` if you want to train trees on both datasets given in the specs (Beware this takes ~17 hours to run)

This will print into terminal the accuracy of the tree, the confusion matrix as well as the precision, recall and F1-measure (this will be done twice if you are training both datasets given). This will also save a jpg image named `fig_post_pruning.jpg` (and 2 images in case you train on both datasets)

## Producing evaluation metrics with a depth requirement (strongly recommended)
Here, you have to specify the depth you want your tree to be. This will bypass the hyperparameter tuning, significantly reducing runtime.
To do this:
1. Ensure that you are in the directory containing the `main.py` file
2. Run the command: `python3 main.py path/to/data depth` where `path/to/data` is the path to the dataset you want to train and `depth` is the desired maximum depth of the tree you are training

This will print into terminal the accuracy of the tree, the confusion matrix as well as the precision, recall and F1-measure. This will also save a jpg image named `fig_post_pruning.jpg`.

**If you wish to obtain the results for the given datasets without waiting 17 hours:**
- Clean dataset: run the command `python3 main.py ./intro2ML-coursework1/wifi_db/clean_dataset.txt 8`
- Noisy dataset: run the command `python3 main.py ./intro2ML-coursework1/wifi_db/noisy_dataset.txt 11`

# Using individual functions



