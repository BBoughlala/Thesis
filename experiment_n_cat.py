import argparse
import pandas as pd
import multiprocessing
from new_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

# Define the discretize function
def discretize(X:np.ndarray, n_cat:int, min_ar:np.ndarray=None, max_ar:np.ndarray=None) -> np.ndarray:
    """
    Digitizes the given data into a given number of categories.

        Parameters:
            X (np.ndarray): The data to digitize.
            n_cat (int): The number of categories to digitize the data into.
            min_ar (np.ndarray): The minimum value of the data.
            max_ar (np.ndarray): The maximum value of the data.

        Returns:
            np.ndarray: The digitized data.
    """
    new_X = np.zeros(X.shape)
    if min_ar is None:
        min_ar = np.min(X, axis=0)
    if max_ar is None:
        max_ar = np.max(X, axis=0)
    for i in range(X.shape[1]):
        new_X[:,i] = np.digitize(X[:,i], np.linspace(min_ar[i], max_ar[i], n_cat), False)
    return new_X, min_ar, max_ar

# Define the training function
def train_model(n_cat, X, y, kf, results_queue):
    tree = DecisionTree(1, 10, 2, [0,1,2,3], 3)
    sktree = DecisionTreeClassifier(criterion='entropy')
    
    tree_results = []
    sktree_results = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, min, max = discretize(X_train, n_cat)
        X_test, _, _ = discretize(X_test, n_cat, min, max)

        tree.fit(X_train, y_train)
        sktree.fit(X_train, y_train)

        y_tree_pred = tree.predict(X_test)
        y_sktree_pred = sktree.predict(X_test)

        tree_results.append(np.mean(y_tree_pred == y_test))
        sktree_results.append(np.mean(y_sktree_pred == y_test))

    results_queue.put({
        'n_cat': n_cat,
        'accuracy_custom': np.mean(tree_results),
        'accuracy_sklearn': np.mean(sktree_results)
    })

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train a decision tree on the iris dataset.')
    parser.add_argument('-min', type=int, default=2, help='The minimum number of bins to digitize the data.')
    parser.add_argument('-max', type=int, default=2, help='The maximum number of bins to digitize the data.')
    parser.add_argument('-output', type=str, default='n_cat.csv', help='The output file to save the results.')
    args = parser.parse_args()
    config = vars(args)

    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    #initiate KFold
    kf = KFold(n_splits=10, shuffle=True)

    # Create processes and run the training function
    processes = []
    results_queue = multiprocessing.Queue()
    for n_cat in range(config['min'], config['max'] + 1):
        process = multiprocessing.Process(target=train_model, args=(n_cat, X, y, kf, results_queue))
        process.start()
        processes.append(process)
        
    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("DONE")

    # Retrieve results from the queue
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Write results to CSV file
    df = pd.DataFrame(results)
    df.to_csv(config['output'], index=False)
