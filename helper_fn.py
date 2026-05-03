import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def split(train_set, test_set):
    X = train_set[:,:-1]
    y = train_set[:,-1]
    X_test = test_set[:, :-1]
    y_test = test_set[:,-1]

    return X, y, X_test, y_test

def plot_accuracy(method, predictions, y_test, show, save):
    acc_dict = dict()
    for i in range(len(set(predictions))):
        true_mask = (y_test == i)
        correct = np.sum(predictions[true_mask] == i)

        avg = correct/ np.sum(true_mask)
        acc_dict[i] = avg

    avg_test_accuracy = np.mean(predictions == y_test)
    
    plt.bar(acc_dict.keys(), acc_dict.values())
    plt.xticks(range(0, len(acc_dict.keys())), acc_dict.keys(), fontsize=5)
    plt.xlabel("Movement Label")
    plt.ylabel("Test Accuracy")
    plt.title(f"{method}, total accuracy = {avg_test_accuracy}")
    if save:
        plt.savefig(f"{method}_acc.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    return avg_test_accuracy
    