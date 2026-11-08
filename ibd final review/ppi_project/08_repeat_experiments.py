
import pandas as pd
import numpy as np
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ----------------------------------
# 1. LOAD DATA
# ----------------------------------

folder = os.path.dirname(os.path.abspath(__file__))
processed = os.path.join(folder, "processed")

positive = pd.read_csv(
    os.path.join(processed, "positive_pairs.csv")
)

binary = pd.read_csv(
    os.path.join(processed, "binary_dataset_random.csv")
)

seq_df = pd.read_csv(
    os.path.join(folder, "data", "raw", "SHS27k.seqs.tsv"),
    sep="\t", header=None, dtype=str
)

protein_ids = seq_df.iloc[:, 0].tolist()

sequence_dict = dict(
    zip(seq_df.iloc[:, 0], seq_df.iloc[:, 1])
)

positive["label"] = 1
binary["label"] = binary["label"].astype(int)

# ----------------------------------
# 2. CALCULATE AAC FEATURES
# ----------------------------------

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

def calculate_aac(sequence):
    sequence = sequence.upper()
    valid = [aa for aa in sequence if aa in amino_acids]

    if not valid:
        raise ValueError("Sequence has no standard amino acids.")

    return np.array([
        valid.count(aa) / len(valid)
        for aa in amino_acids
    ])

protein_features = {
    pid: calculate_aac(seq)
    for pid, seq in sequence_dict.items()
}

def make_features(data):
    X = []

    for _, row in data.iterrows():
        a = protein_features[row["protein_a"]]
        b = protein_features[row["protein_b"]]

        X.append(np.concatenate([
            np.abs(a - b),
            a * b
        ]))

    return np.array(X)

# ----------------------------------
# 3. GENERATE NEGATIVE PAIRS
# ----------------------------------

positive_set = set(
    zip(positive["protein_a"], positive["protein_b"])
)

def generate_negatives(group, count, seed):
    rng = random.Random(seed)
    group = list(group)
    negatives = set()

    while len(negatives) < count:
        a = rng.choice(group)
        b = rng.choice(group)

        if a == b:
            continue

        pair = tuple(sorted([a, b]))

        if pair not in positive_set and pair not in negatives:
            negatives.add(pair)

    result = pd.DataFrame(
        list(negatives),
        columns=["protein_a", "protein_b"]
    )

    result["label"] = 0
    return result

# ----------------------------------
# 4. TRAIN AND EVALUATE
# ----------------------------------

def evaluate(train, test, split_name, seed):

    X_train = make_features(train)
    X_test = make_features(test)

    y_train = train["label"].astype(int)
    y_test = test["label"].astype(int)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return {
        "seed": seed,
        "split": split_name,
        "train_pairs": len(train),
        "test_pairs": len(test),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, prob)
    }

# ----------------------------------
# 5. REPEAT FIVE TIMES
# ----------------------------------

all_results = []
seeds = [42, 100, 200, 300, 400]

for seed in seeds:

    print("\nRunning seed:", seed)

    # RANDOM SPLIT
    random_train, random_test = train_test_split(
        binary,
        test_size=0.2,
        random_state=seed,
        stratify=binary["label"]
    )

    r = evaluate(
        random_train, random_test, "Random", seed
    )
    all_results.append(r)

    # PROTEIN-DISJOINT SPLIT
    train_ids, test_ids = train_test_split(
        protein_ids,
        test_size=0.2,
        random_state=seed
    )

    train_ids = set(train_ids)
    test_ids = set(test_ids)

    train_pos = positive[
        positive["protein_a"].isin(train_ids) &
        positive["protein_b"].isin(train_ids)
    ].copy()

    test_pos = positive[
        positive["protein_a"].isin(test_ids) &
        positive["protein_b"].isin(test_ids)
    ].copy()

    if len(train_pos) == 0 or len(test_pos) == 0:
        raise ValueError("A protein group has no positive pairs.")

    train_neg = generate_negatives(
        train_ids, len(train_pos), seed + 1000
    )

    test_neg = generate_negatives(
        test_ids, len(test_pos), seed + 2000
    )

    disjoint_train = pd.concat(
        [train_pos, train_neg], ignore_index=True
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    disjoint_test = pd.concat(
        [test_pos, test_neg], ignore_index=True
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Verify no shared proteins
    train_used = (
        set(disjoint_train["protein_a"]) |
        set(disjoint_train["protein_b"])
    )

    test_used = (
        set(disjoint_test["protein_a"]) |
        set(disjoint_test["protein_b"])
    )

    assert len(train_used & test_used) == 0

    d = evaluate(
        disjoint_train, disjoint_test,
        "Protein-disjoint", seed
    )

    all_results.append(d)

# ----------------------------------
# 6. SAVE DETAILED RESULTS
# ----------------------------------

results = pd.DataFrame(all_results)

results.to_csv(
    os.path.join(folder, "results", "five_seed_results.csv"),
    index=False
)

print("\nDETAILED RESULTS")
print(results.to_string(index=False))

# ----------------------------------
# 7. CALCULATE MEAN AND STD
# ----------------------------------

summary = results.groupby("split")[
    ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
].agg(["mean", "std"])

summary.to_csv(
    os.path.join(folder, "results", "five_seed_summary.csv")
)

print("\nMEAN AND STANDARD DEVIATION")
print(summary)

print("\nAll five experiments completed!")
