from helper_fn import split, plot_accuracy
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
    



def weight_changes(initial_params, trained_params):
    """
    initial weights vs trained weights plot for all layers
    
    """
    #@title What fraction of parameters flip sign?
    for key in initial_params:
        n_weights=initial_params[key].numel()
        n_flip = (initial_params[key].sign()*trained_params[key].sign()<0).count_nonzero().item()
        print(key + ' flipped: % .2f%% (%d/%d)' %(100*n_flip/n_weights, n_flip, n_weights))

    for key in initial_params:
        n_weights=initial_params[key].numel()
        n_changed = (initial_params[key] != trained_params[key]).count_nonzero().item()
        print(key + ' changed: % .2f%% (%d/%d)' %(100*n_changed/n_weights, n_changed, n_weights))

    # Plot initial vs trained values

    fig, axs = plt.subplots(3,int(math.ceil(len(trained_params)/3)),figsize=(20, 7))
    plt.subplots_adjust(hspace=0.5)

    ax = 0
    for i, ax in enumerate(axs.flatten()):
        if i >= len(trained_params):
            ax.axis('off')
            continue
        key = list(initial_params)[i]
        ax.scatter(initial_params[key].numpy(),trained_params[key].numpy(),s=10,alpha=0.5)
        ax.axhline(y=0, linewidth=2, color = 'r', ls='--')
        ax.axvline(x=0, linewidth=2, color = 'r', ls='--')
        ax.set_title(key)

    plt.tight_layout()
    plt.show()

def run_MLP(train_set, test_set, model_config, show = True, save = False):
    X, y, X_test, y_test = split(train_set, test_set)

    model_config = {"paper_config": paper_config, "my_config": my_config}[model_config]
    
    # model params
    in_features = X.shape[1]
    out_features = y.shape[0]
    
    # save for accuracy calculation later
    numpy_y_test = y_test

    # reformat labels

    # reformat data
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(X, y)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # params
    model = model_config(in_features=in_features, h1=512, h2=512, out_features=out_features, batch_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    epochs = 10
    losses = []

    # record initial params
    initial_params = model.record_params()

    # train
    for _ in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
    
    # plt.plot(losses)
    # plt.title("Loss")
    # plt.show()
    # plt.close()

    trained_params = model.record_params()

    weight_changes(initial_params, trained_params)

    # test
    predictions = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            pred = pred.squeeze().tolist()
            predictions.extend(pred)

    predictions = np.array(predictions)
    avg_test_accuracy = plot_accuracy(f"MLP_{model_config}.png", predictions, numpy_y_test, show, save)

    return avg_test_accuracy


class paper_config(nn.Module):
    def __init__ (self, in_features=180, h1=512, h2=None, out_features = 12, batch_size = 32):
        super().__init__()

        self.w1 = nn.Linear(in_features, h1)
        self.out = nn.Linear(h1, out_features)

        self.sigmoid = nn.Sigmoid()

        self.init_weights = self.record_params(calc_sign=False)

        

    def forward(self, x):
        x = self.w1(x)
        x = self.sigmoid(x)

        x = self.out(x)

        return x
    
    def record_params(self, calc_sign: bool = True, print_sign: bool = False):
        """Save the network weights/biases for all nn.Linear layers."""
        recorded_params = {}

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                recorded_params[f"{name}.weight"] = module.weight.detach().cpu().clone()
                if module.bias is not None:
                    recorded_params[f"{name}.bias"] = module.bias.detach().cpu().clone()

        if calc_sign:
            for name, cur_data in recorded_params.items():
                frac_pos = 100 * (torch.sum(cur_data > 0).item() / cur_data.numel())
                frac_zero = 100 * (torch.sum(cur_data == 0).item() / cur_data.numel())
                frac_neg = 100 * (torch.sum(cur_data < 0).item() / cur_data.numel())
                print(f"{name}: Positive: {frac_pos:.2f}%; Negative: {frac_neg:.2f}%; Zero: {frac_zero:.2f}%")

        return recorded_params

class my_config(nn.Module):
    def __init__ (self, in_features=180, h1=512, h2=512, out_features = 12, batch_size = 32):
        super().__init__()

        self.w1 = nn.Linear(in_features, h1)
        self.w2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

        self.relu = nn.ReLU()

        self.init_weights = self.record_params(calc_sign=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.relu(x)

        x = self.w2(x)
        x = self.relu(x)

        x = self.out(x)

        return x
    
    def record_params(self, calc_sign: bool = True, print_sign: bool = False):
        """Save the network weights/biases for all nn.Linear layers."""
        recorded_params = {}

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                recorded_params[f"{name}.weight"] = module.weight.detach().cpu().clone()
                if module.bias is not None:
                    recorded_params[f"{name}.bias"] = module.bias.detach().cpu().clone()

        if calc_sign:
            for name, cur_data in recorded_params.items():
                frac_pos = 100 * (torch.sum(cur_data > 0).item() / cur_data.numel())
                frac_zero = 100 * (torch.sum(cur_data == 0).item() / cur_data.numel())
                frac_neg = 100 * (torch.sum(cur_data < 0).item() / cur_data.numel())
                print(f"{name}: Positive: {frac_pos:.2f}%; Negative: {frac_neg:.2f}%; Zero: {frac_zero:.2f}%")

        return recorded_params

