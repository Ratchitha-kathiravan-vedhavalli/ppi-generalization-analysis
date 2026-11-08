

import pandas as pd
import matplotlib.pyplot as plt
import os

# Project folder
project_folder = os.path.dirname(os.path.abspath(__file__))

results_path = os.path.join(
    project_folder, "results", "twenty_seed_results.csv"
)

# Load experiment results
df = pd.read_csv(results_path)

# Calculate mean and standard deviation
summary = df.groupby("split").agg(
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    auc_mean=("roc_auc", "mean"),
    auc_std=("roc_auc", "std")
)

print(summary)

names = ["Random", "Protein-disjoint"]
summary = summary.loc[names]

# Accuracy graph
plt.figure(figsize=(8, 5))
plt.bar(
    names,
    summary["accuracy_mean"],
    yerr=summary["accuracy_std"],
    capsize=8
)
plt.title("Accuracy Comparison")
plt.ylabel("Mean Accuracy")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(
    os.path.join(project_folder, "results", "accuracy_comparison.png"),
    dpi=300
)
plt.close()

print("Accuracy graph saved!")

# ROC-AUC graph
plt.figure(figsize=(8, 5))
plt.bar(
    names,
    summary["auc_mean"],
    yerr=summary["auc_std"],
    capsize=8
)
plt.title("ROC-AUC Comparison")
plt.ylabel("Mean ROC-AUC")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(
    os.path.join(project_folder, "results", "roc_auc_comparison.png"),
    dpi=300
)
plt.close()

print("ROC-AUC graph saved!")

print("\nBoth graphs generated successfully!")
