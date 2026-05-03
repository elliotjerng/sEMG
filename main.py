import numpy as np
from pathlib import Path
from lda import sklearn_LDA, LDA
from svm import sklearn_SVM, linear_SVM, rbf_SVM
from knn import sklearn_KNN, KNN
from mlp import run_MLP
from load_data import load_data
import pickle


if __name__ == "__main__":
    """
    load the data:
    - each train/test set:
    - 2D array: [num windows/input samples, per each input sample --> (18 features for each window * 10 recordings) + 1 movement label (target)]
    """
    window_len = 400 # either 100, 200, or 400, or 500
    files = ["Ninapro Data/s1/S1_A1_E1.mat", "Ninapro Data/s1/S1_A1_E2.mat", "Ninapro Data/s1/S1_A1_E3.mat"]
    features = ["all"]

    for feature in features:
        if not Path(f"test_set_{window_len}_{feature}.npy").exists() and not Path(f"train_set_{window_len}_{feature}.npy").exists():
            train_set, test_set = load_data(files, window_len = window_len, feature = feature, save = True, show_data = False)
        else:
            train_set = np.load(f"train_set_{window_len}_{feature}.npy")
            test_set = np.load(f'test_set_{window_len}_{feature}.npy')
        
        acc_dict = dict()
        
        # LDA
        # acc_dict["LDA (sklearn)"] = sklearn_LDA(train_set, test_set, show=False, save=True)
        # acc_dict["LDA"] = LDA(train_set, test_set, show=False, save=True)

        # print("acc_dict:", acc_dict)

        # # # SVM
        # acc_dict["SVM (sklearn lin.)"] = sklearn_SVM(train_set, test_set, "linear", show=False, save=True)
        # acc_dict["SVM (sklearn RBF)"] = sklearn_SVM(train_set, test_set, "rbf", show=False, save=True)
        # acc_dict["SVM (lin.)"] = linear_SVM(train_set, test_set, show=False, save=True)
        # # acc_dict["SVM (RBF)"] = rbf_SVM(train_set, test_set, show=False, save=True)

        # print("acc_dict:", acc_dict)

        # # k nearest neighbors
        # acc_dict["k-NN (sklearn)"] = sklearn_KNN(train_set, test_set, show=False, save=True)
        # acc_dict["k-NN"] = KNN(train_set, test_set, show=False, save=True)

        # print("acc_dict:", acc_dict)

        # MLP
        acc_dict["MLP (copy)"] = run_MLP(train_set, test_set, model_config="paper_config", show=False, save=True)
        acc_dict["MLP (mod)"] = run_MLP(train_set, test_set, model_config="my_config", show=False, save=True)

        print("acc_dict:", acc_dict)

        # with open(f'acc_dict_{window_len}_{feature}.pickle', 'wb') as f:
        #     pickle.dump(acc_dict, f)
    
    