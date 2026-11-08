

import pandas as pd
import numpy as np
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# 1. PROJECT FOLDER AND FILE PATHS
# --------------------------------------------------

project_folder = os.path.dirname(os.path.abspath(__file__))

sequence_path = os.path.join(
    project_folder, "data", "raw", "SHS27k.seqs.tsv"
)

positive_path = os.path.join(
    project_folder, "processed", "positive_pairs.csv"
)

binary_path = os.path.join(
    project_folder, "processed", "binary_dataset_random.csv"
)

results_folder = os.path.join(project_folder, "results")
os.makedirs(results_folder, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

sequences_df = pd.read_csv(
    sequence_path,
    sep="\t",
    header=None,
    dtype=str
)

sequences = dict(zip(sequences_df.iloc[:, 0], sequences_df.iloc[:, 1]))

positive_pairs = pd.read_csv(positive_path, dtype=str)
binary_dataset = pd.read_csv(binary_path, dtype=str)

for df in [positive_pairs, binary_dataset]:
    df["protein_a"] = df["protein_a"].astype(str)
    df["protein_b"] = df["protein_b"].astype(str)
    df["label"] = df["label"].astype(int)

print("Sequences loaded:", len(sequences))
print("Positive pairs loaded:", len(positive_pairs))
print("Binary dataset loaded:", len(binary_dataset))


# --------------------------------------------------
# 3. CREATE AMINO ACID COMPOSITION FEATURES
# --------------------------------------------------

amino_acids = "ACDEFGHIKLMNPQRSTVWY"


def calculate_aac(sequence):

    sequence = str(sequence).upper()

    valid_count = sum(sequence.count(aa) for aa in amino_acids)

    if valid_count == 0:
        return np.zeros(20)

    features = []

    for aa in amino_acids:
        features.append(sequence.count(aa) / valid_count)

    return np.array(features)


# Calculate features once for every protein
protein_features = {}

for protein_id, sequence in sequences.items():
    protein_features[protein_id] = calculate_aac(sequence)


def create_pair_features(data):

    X = []
    y = []

    for _, row in data.iterrows():

        protein_a = row["protein_a"]
        protein_b = row["protein_b"]

        feature_a = protein_features[protein_a]
        feature_b = protein_features[protein_b]

        pair_feature = np.concatenate([
            np.abs(feature_a - feature_b),
            feature_a * feature_b
        ])

        X.append(pair_feature)
        y.append(int(row["label"]))

    return np.array(X), np.array(y)


# --------------------------------------------------
# 4. GENERATE NEGATIVE PAIRS FOR PROTEIN-DISJOINT
# --------------------------------------------------

def make_pair(a, b):
    return tuple(sorted((a, b)))


def generate_negative_pairs(protein_group, number_needed,
                            known_positive_pairs, seed):

    rng = random.Random(seed)
    proteins = list(protein_group)

    negative_set = set()

    while len(negative_set) < number_needed:

        a, b = rng.sample(proteins, 2)

        pair = make_pair(a, b)

        if pair not in known_positive_pairs:
            negative_set.add(pair)

    negative_df = pd.DataFrame(
        list(negative_set),
        columns=["protein_a", "protein_b"]
    )

    negative_df["label"] = 0

    return negative_df


# --------------------------------------------------
# 5. TRAIN AND EVALUATE THE MODEL
# --------------------------------------------------

def evaluate_model(train_data, test_data, split_name, seed):

    X_train, y_train = create_pair_features(train_data)
    X_test, y_test = create_pair_features(test_data)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=seed
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    return {
        "seed": seed,
        "split": split_name,
        "train_pairs": len(train_data),
        "test_pairs": len(test_data),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# --------------------------------------------------
# 6. RUN 20 EXPERIMENTS
# --------------------------------------------------

seeds = list(range(20))

all_results = []

known_positive_pairs = set(
    make_pair(row["protein_a"], row["protein_b"])
    for _, row in positive_pairs.iterrows()
)

all_proteins = sorted(sequences.keys())

for seed in seeds:

    print("\n----------------------------------")
    print("RUNNING SEED:", seed)
    print("----------------------------------")

    # ----------------------------------------------
    # A. RANDOM SPLIT
    # ----------------------------------------------

    random_train, random_test = train_test_split(
        binary_dataset,
        test_size=0.2,
        random_state=seed,
        stratify=binary_dataset["label"]
    )

    random_result = evaluate_model(
        random_train,
        random_test,
        "Random",
        seed
    )

    all_results.append(random_result)

    print("Random split completed.")
    print("Accuracy:", round(random_result["accuracy"], 4))
    print("ROC-AUC:", round(random_result["roc_auc"], 4))

    # ----------------------------------------------
    # B. PROTEIN-DISJOINT SPLIT
    # ----------------------------------------------

    rng = random.Random(seed)

    shuffled_proteins = all_proteins.copy()
    rng.shuffle(shuffled_proteins)

    split_index = int(0.8 * len(shuffled_proteins))

    train_proteins = set(shuffled_proteins[:split_index])
    test_proteins = set(shuffled_proteins[split_index:])

    # Keep only positive pairs where both proteins
    # belong to the same partition.

    disjoint_train_positive = positive_pairs[
        positive_pairs["protein_a"].isin(train_proteins) &
        positive_pairs["protein_b"].isin(train_proteins)
    ].copy()

    disjoint_test_positive = positive_pairs[
        positive_pairs["protein_a"].isin(test_proteins) &
        positive_pairs["protein_b"].isin(test_proteins)
    ].copy()

    # Generate separate negative pairs for each group.

    train_negative = generate_negative_pairs(
        train_proteins,
        len(disjoint_train_positive),
        known_positive_pairs,
        seed + 1000
    )

    test_negative = generate_negative_pairs(
        test_proteins,
        len(disjoint_test_positive),
        known_positive_pairs,
        seed + 2000
    )

    disjoint_train = pd.concat(
        [disjoint_train_positive, train_negative],
        ignore_index=True
    )

    disjoint_test = pd.concat(
        [disjoint_test_positive, test_negative],
        ignore_index=True
    )

    # Verify that no protein is shared.

    train_used = set(disjoint_train["protein_a"]) | set(
        disjoint_train["protein_b"]
    )

    test_used = set(disjoint_test["protein_a"]) | set(
        disjoint_test["protein_b"]
    )

    shared_proteins = train_used.intersection(test_used)

    print("Shared proteins:", len(shared_proteins))

    assert len(shared_proteins) == 0, (
        "ERROR: Protein overlap detected!"
    )

    # Evaluate protein-disjoint model.

    disjoint_result = evaluate_model(
        disjoint_train,
        disjoint_test,
        "Protein-disjoint",
        seed
    )

    all_results.append(disjoint_result)

    print("Protein-disjoint split completed.")
    print("Accuracy:", round(disjoint_result["accuracy"], 4))
    print("ROC-AUC:", round(disjoint_result["roc_auc"], 4))


# --------------------------------------------------
# 7. SAVE ALL RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

results_path = os.path.join(
    results_folder,
    "twenty_seed_results.csv"
)

results_df.to_csv(results_path, index=False)


# --------------------------------------------------
# 8. CALCULATE MEAN AND STANDARD DEVIATION
# --------------------------------------------------

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

summary_df = results_df.groupby("split")[metrics].agg(
    ["mean", "std"]
)

summary_path = os.path.join(
    results_folder,
    "twenty_seed_summary.csv"
)

summary_df.to_csv(summary_path)


# --------------------------------------------------
# 9. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\n======================================")
print("20-SEED EXPERIMENT COMPLETED")
print("======================================")

print("\nMEAN AND STANDARD DEVIATION:")
print(summary_df)

print("\nIndividual experiment results saved to:")
print(results_path)

print("\nSummary saved to:")
print(summary_path)
