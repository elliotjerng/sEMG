import numpy as np
import scipy.io as sio
from compute_features import get_features
from collections import Counter
from helper_fn import split
from scipy.special import softmax

def load_data(files, feature, window_len = 400, save = False, show_data = False):
    """
    each file contains data for an excercise type

    data is a dictionary, with these keys:
    emg, stimulus, glove, subject, exercise, repetition, restimulus, rerepetition

    data["emg"] is a numpy array of shape (101014, 10), where the 10 columns correspond to different recording areas, and each column contains emg signal values.
    """
    features = get_all_features(files, window_len, feature)

    # split train and test based on repetition number
    train_set, test_set = split_dataset(features)
    
    # subsample train set by factor of 10
    subsample_factor = 10
    idx = np.random.choice(train_set.shape[0], train_set.shape[0] // subsample_factor, replace=False)
    train_set = train_set[idx]
    
    if show_data:
        X, y, X_test, y_test = split(train_set, test_set)
        probabilities = softmax(list(Counter(y).values()))
        print("y counter values:", Counter(y))
        print("y counter keys:", Counter(y).keys())
        print("y counter values softmaxed", probabilities)
        print("train set shape:", train_set.shape)
        print("test set shape:", test_set.shape)


    if save:
        np.save(f'train_set_{window_len}_{feature}.npy', train_set)
        np.save(f'test_set_{window_len}_{feature}.npy', test_set)
    
    return train_set, test_set

def get_all_features(files, window_len, feature):
    maxi, mini = global_max_min(files)

    all_features = []
    label_offset = 0

    for file_name in files:
        data = sio.loadmat(file_name)

        features = [get_features(recording, window_len, maxi, mini, feature) for recording in data["emg"].T]
        features = np.concatenate(features, axis=1)

        raw_movement_label = np.array([data["restimulus"][int(i - 1 + window_len / 10)] for i in range(features.shape[0])]).squeeze()

        movement_label = raw_movement_label.copy()
        nonzero = movement_label != 0
        movement_label[nonzero] += label_offset

        if np.any(nonzero):
            label_offset += int(raw_movement_label[nonzero].max())

        repetition_label = np.array([data["rerepetition"][int(i - 1 + window_len / 10)] for i in range(features.shape[0])]).squeeze()

        file_features = np.concatenate((features, movement_label.reshape(-1, 1), repetition_label.reshape(-1, 1)), axis=1)
        all_features.append(file_features)

    return np.concatenate(all_features, axis=0)

def global_max_min(files):

    train_set_repetitions = [1, 3, 4, 5, 9]
    all_data = [sio.loadmat(f) for f in files]
    all_train_emg = np.concatenate([d["emg"][np.isin(d["rerepetition"].squeeze(), train_set_repetitions),:] for d in all_data], axis=0)

    maxi = np.percentile(all_train_emg, 99)
    mini = np.min(all_train_emg)

    return maxi, mini

def split_dataset(features):
    # split train and test based on repetition number
    train_reps = [1, 3, 4, 5, 9]
    test_reps  = [2, 6, 7, 8, 10]

    reps = features[:, -1].copy()
    assigned_reps = reps.copy()

    for i in range(len(reps) - 2, -1, -1):
        if assigned_reps[i] == 0:
            assigned_reps[i] = assigned_reps[i + 1]

    train_mask = np.isin(assigned_reps, train_reps)
    test_mask  = np.isin(assigned_reps, test_reps)

    train_set = features[train_mask][:, :-1]
    test_set  = features[test_mask][:, :-1]

    return train_set, test_set