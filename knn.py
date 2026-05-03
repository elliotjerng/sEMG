import numpy as np
from helper_fn import split, plot_accuracy
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

def sklearn_KNN(train_set, test_set, show = True, save = False):
    X, y, X_test, y_test = split(train_set, test_set)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)

    predictions = neigh.predict(X_test)

    avg_test_accuracy = plot_accuracy("sklearn KNN.png", predictions, y_test, show, save)
    return avg_test_accuracy

def predict(x, y, X):
    distances = np.linalg.norm(X - x, axis=1)

    k_indices = np.argsort(distances)[:3]
    k_nearest_labels = y[k_indices]

    return Counter(k_nearest_labels).most_common(1)[0][0]

def KNN(train_set, test_set, show = True, save = False):
    X, y, X_test, y_test = split(train_set, test_set)

    predictions = np.array([predict(x, y, X) for x in X_test])

    avg_test_accuracy = plot_accuracy("KNN from scratch.png", predictions, y_test, show, save)
    return avg_test_accuracy
