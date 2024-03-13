from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from minimal_credal_tree import DecisionTree
from datasets import zoo, wave, vehicle, sponge, iris, krkp, hepatitis, heartStat, dermatology, german, horse
import numpy as np
import pandas as pd
import multiprocessing

def generate_results(dataset_index):
    datasets = [zoo, wave, vehicle, sponge, iris, krkp, hepatitis, heartStat, dermatology, german, horse]
    dataset = datasets[dataset_index]
    X, y = dataset.fetch()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    custom_scores = []
    sk_scores = []
    custom_depths = []
    sk_depths = []

    for train_index, test_index in kf.split(X):
        print(f"Processing dataset {dataset_index}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        custom = DecisionTree(s=1, maxiter=100, base=2)
        custom.fit(X_train, y_train)

        sk = DecisionTreeClassifier()
        sk.fit(X_train, y_train)

        custom_pred = custom.predict(X_test)
        sk_pred = sk.predict(X_test)

        custom_score = accuracy_score(y_test, custom_pred)
        sk_score = accuracy_score(y_test, sk_pred)

        custom_depth = custom.get_depth()
        sk_depth = sk.get_depth()

        custom_scores.append(custom_score)
        sk_scores.append(sk_score)

        custom_depths.append(custom_depth)
        sk_depths.append(sk_depth)

    return np.mean(custom_scores), np.mean(sk_scores), np.mean(custom_depths), np.mean(sk_depths)

def main():
    # Number of datasets
    num_datasets = 11
    
    results = []

    # Use multiprocessing to parallelize dataset processing
    with multiprocessing.Pool() as pool:
        results = pool.map(generate_results, range(num_datasets))

    results = pd.DataFrame(results, columns=["Custom", "Sklearn", "Custom Depth", "Sklearn Depth"], index=["Zoo", "Wave", "Vehicle", "Sponge", "Iris", "Krkp", "Hepatitis", "HeartStat", "Dermatology", "German", "Horse"])
    results = results.round(3)
    results.to_csv("results.csv")

if __name__ == "__main__":
    main()
    print("Results saved to results.csv")