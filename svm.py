from sklearn import svm
from helper_fn import split, plot_accuracy
import numpy as np
from collections import Counter


def sklearn_SVM(train_set, test_set, kernel, show = True, save = False):
    """
    sanity check of dataset using sklearn's SVM module
    """
    X, y, X_test, y_test = split(train_set, test_set)

    linear_svm = svm.SVC(kernel=kernel)
    linear_svm.fit(X, y)
    predictions = linear_svm.predict(X_test)

    avg_test_accuracy = plot_accuracy(f"{kernel}_sklearn SVM.png", predictions, y_test, show, save)
    return avg_test_accuracy

def rbf_feature_map(X, D=500, gamma=0.1):
    n_features = X.shape[1]
    
    W = np.random.normal(0, np.sqrt(2 * gamma), size=(n_features, D))
    b = np.random.uniform(0, 2*np.pi, size=D)
    
    Z = np.sqrt(2 / D) * np.cos(X @ W + b)
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)
    
    return Z, W, b

def linear_SVM(train_set, test_set, show = True, save = False):
    """
    L2 norm

    hinge loss

    C (regularization parameter): 1

    w.dot(xi) + b >= 1 if yi = class 1

    w.dot(xi) + b <= -1 if yi = class -1

    """
    X, y, X_test, y_test = split(train_set, test_set)

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8

    X = (X - mean) / std
    X_test = (X_test - mean) / std

    n_samples, n_features = X.shape
    
    lr = 0.001
    lambd = 0.01

    classes = np.unique(y)
    n_classes = len(classes)

    W = np.zeros((n_classes, n_features))
    b = np.zeros(n_classes)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[label] for label in y])

    for _ in range(300):
        indices = np.random.permutation(n_samples)

        for idx in indices:
            xi = X[idx]                 # shape: (n_features,)
            k = y_idx[idx]              # true class index

            scores = W @ xi + b         # shape: (n_classes,)

            y_vec = -np.ones(n_classes)
            y_vec[k] = 1

            margins = y_vec * scores
            violated = margins < 1      # shape: (n_classes,)

            gradW = 2 * lambd * W
            gradb = np.zeros(n_classes)

            gradW[violated] -= y_vec[violated, None] * xi
            gradb[violated] -= y_vec[violated]

            W -= lr * gradW
            b -= lr * gradb


    scores = X_test @ W.T + b
    pred_idx = np.argmax(scores, axis=1)
    predictions = classes[pred_idx]


    avg_test_accuracy = plot_accuracy("linear SVM from scratch.png", predictions, y_test, show, save)

    return avg_test_accuracy

def rbf_SVM(train_set, test_set, show = True, save = False):
    """
    L2 norm

    C (regularization parameter): 1

    w.dot(xi) + b >= 1 if yi = class 1

    w.dot(xi) + b <= -1 if yi = class -1

    """
    X, y, X_test, y_test = split(train_set, test_set)

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8

    X = (X - mean) / std
    X_test = (X_test - mean) / std

    # BEFORE training
    D = 2000
    Z, W, b_rff = rbf_feature_map(X, D=D, gamma=0.01)

    # Apply SAME transform to test
    Z_test = np.sqrt(2 / D) * np.cos(X_test @ W + b_rff)

    n_samples, n_features = Z.shape
    
    lr = 0.0001
    lambd = 0.01

    classes = np.unique(y)
    
    pred_arr = np.zeros((len(classes), y_test.shape[0]))

    for i, c in enumerate(classes):
        w = np.zeros(n_features)
        b = 0

        class_y = np.where(y == c, 1, -1)

        for _ in range(300):
            indices = np.random.permutation(len(Z))
            for idx in indices:
                xi, yi = Z[idx], class_y[idx]
                condition = yi * (w.dot(xi) + b) >= 1

                if condition:
                    w -= lr * (2 * lambd * w)
                else:
                    w -= lr * (2 * lambd * w - yi * xi)
                    b -= lr * yi
        
        class_predictions = Z_test.dot(w) + b
        pred_arr[i] = class_predictions
    
    # print(pred_arr.mean(axis=1), pred_arr.std(axis=1))
    
    predictions = classes[np.argmax(pred_arr, axis=0)]

    avg_test_accuracy = plot_accuracy("rbf SVM from scratch.png", predictions, y_test, show, save)

    return avg_test_accuracy