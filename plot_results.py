import pickle
import numpy as np
import matplotlib.pyplot as plt

features = ["MAV", "HIST", "WL", "VAR", "CC", "all"]
window_len = 400

all_results = {}

for feature in features:
    with open(f'acc_dict_{window_len}_{feature}.pickle', 'rb') as f:
        all_results[feature] = pickle.load(f)

all_results["VAR"]["MLP (mod)"] = 0.6766605868907718


classifier_order = [
    "SVM (sklearn lin.)",
    "SVM (lin.)",
    "MLP (copy)",
    "MLP (mod)",
    "k-NN (sklearn)",
    "k-NN",
    "SVM (sklearn RBF)",
    "LDA (sklearn)",
    "LDA"
]

classifiers = classifier_order

x = np.arange(len(features))

group_width = 0.8                         # total width of each feature group
bar_width = group_width / len(classifiers)

fig, ax = plt.subplots(figsize=(16, 6))

# Plot tightly packed bars within each group
for i, clf in enumerate(classifiers):
    offset = -group_width/2 + i * bar_width + bar_width/2
    accuracies = [all_results[f][clf] for f in features]
    ax.bar(x + offset, accuracies, bar_width, label=clf)

# --- Classifier labels under each bar (horizontal, compact) ---
bar_positions = []
bar_labels = []

for j in range(len(features)):
    for i, clf in enumerate(classifiers):
        offset = -group_width/2 + i * bar_width + bar_width/2
        bar_positions.append(x[j] + offset)
        bar_labels.append(clf)

ax.set_xticks(bar_positions)
ax.set_xticklabels(bar_labels, fontsize=7, rotation=90)

# --- Feature labels under groups ---
for j, feature in enumerate(features):
    ax.text(
        x[j],
        -0.3,
        feature,
        ha='center',
        va='top',
        transform=ax.get_xaxis_transform(),
        fontsize=11,
        fontweight='bold'
    )

ax.set_ylabel("Accuracy")
ax.set_ylim(0.4, 1.0)

from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(ncol=len(classifiers), loc='upper center', bbox_to_anchor=(0.5, 1.15))

plt.tight_layout()
plt.show()