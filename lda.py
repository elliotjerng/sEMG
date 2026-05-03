import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import Counter
from helper_fn import split, plot_accuracy
import matplotlib.pyplot as plt

def dim_reduction(train_set, test_set, show = True, save = False, subset = None):
    X, y, X_test, y_test = split(train_set, test_set)
    
    if subset != None:
        mask = np.isin(y, subset)
        X = X[mask]
        y = y[mask]

    classes = np.unique(y)
    full_mean = np.mean(X, axis = 0)
    c = Counter(y)

    W = np.zeros((X.shape[1], X.shape[1]))
    B = np.zeros((X.shape[1], X.shape[1]))

    for i in classes:
        n = c[i]
        class_data = X[y[:] == i]
        class_mean = np.mean(class_data, axis = 0)

        # B
        diff = (class_mean - full_mean).reshape(-1, 1)
        B += n * (diff).dot(diff.T)

        # W
        diff = class_data - class_mean
        mat_mul = diff.T.dot(diff)
        W += mat_mul

    # calculate separation S
    W_inv = np.linalg.pinv(W)
    S = W_inv.dot(B)

    eigenvalues, eigenvectors = np.linalg.eig(S)
    weights = eigenvectors[:, np.argmax(eigenvalues)]

    scores = X.dot(weights)

    mappings = dict()
    for cls in classes:
        mappings[cls] = scores[y == cls]

    for x, vals in mappings.items():
        jitter = np.random.normal(loc=0, scale=0.05, size=len(vals))
        plt.scatter(np.full(len(vals), x) + jitter, vals, alpha=0.6)

    plt.xticks(list(mappings.keys()), list(mappings.keys()))
    plt.xlabel("Movement Label")
    plt.ylabel("Score")
    plt.title("LDA Projection Scores by Class")

    if save:
        plt.savefig("LDA_projections.png", dpi=300)
    if show:
        plt.show()

def LDA(train_set, test_set, show = True, save = False, subset = None):
    """
    LDA Classifer from scratch
    """
    X, y, X_test, y_test = split(train_set, test_set)
    
    if subset != None:
        mask = np.isin(y, subset)
        X = X[mask]
        y = y[mask]

    classes = np.unique(y)

    # priors
    c = Counter(y)
    priors = {key: value / X.shape[0] for key, value in c.items()}
    
    # shared covariance matrix, pooled across all classes
    M = np.zeros((X.shape[1], X.shape[1]))
    class_means = dict()

    for i in classes:
        class_data = X[y[:] == i]
        class_means[i] = np.mean(class_data, axis = 0)

        # M
        diff = class_data - class_means[i]
        mat_mul = diff.T.dot(diff)
        M += mat_mul
    
    M /= (X.shape[0] - len(classes))
    M_inv = np.linalg.pinv(M)

    # predict
    predictions = []
    for x in X_test:
        class_prob = []
        for i in classes:
            var = x - class_means[i]

            prediction = (-1/2) * (var).T.dot(M_inv).dot(var) + np.log(priors[i])
            class_prob.append(prediction)

        predictions.append(np.argmax(class_prob))

    predictions = np.array(predictions)
    
    avg_test_accuracy = plot_accuracy("LDA from scratch.png", predictions, y_test, show, save)

    return avg_test_accuracy

def sklearn_LDA(train_set, test_set, show = True, save = False):
    """
    sanity check of dataset using sklearn's LDA module
    """
    X, y, X_test, y_test = split(train_set, test_set)

    lda = LinearDiscriminantAnalysis(n_components=1, solver="eigen", shrinkage=1e-4)
    lda.fit(X, y)
    predictions = lda.predict(X_test)

    eigenvectors = lda.scalings_
    eigenvalues = lda.explained_variance_ratio_
    classes = np.unique(y)

    weights = eigenvectors[:, np.argmax(eigenvalues)]
    scores = X.dot(weights)

    mappings = dict()
    for cls in classes:
        mappings[cls] = scores[y == cls]

    for x, vals in mappings.items():
        jitter = np.random.normal(loc=0, scale=0.05, size=len(vals))
        plt.scatter(np.full(len(vals), x) + jitter, vals, alpha=0.6)

    plt.xticks(list(mappings.keys()), list(mappings.keys()))
    plt.xlabel("Movement Label")
    plt.ylabel("Score")
    plt.title("LDA Projection Scores by Class")

    if save:
        plt.savefig("LDA_projections.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    avg_test_accuracy = plot_accuracy("sklearn LDA.png", predictions, y_test, show, save)

    return avg_test_accuracy