
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# -----------------------------------
# 1. PROJECT PATHS
# -----------------------------------

project_folder = os.path.dirname(os.path.abspath(__file__))

sequence_path = os.path.join(
    project_folder, "data", "raw", "SHS27k.seqs.tsv"
)

processed_folder = os.path.join(
    project_folder, "processed"
)

# -----------------------------------
# 2. LOAD PROTEIN SEQUENCES
# -----------------------------------

sequences_df = pd.read_csv(
    sequence_path,
    sep="\t",
    header=None,
    dtype=str
)

sequence_dict = dict(
    zip(sequences_df.iloc[:, 0], sequences_df.iloc[:, 1])
)

# The 20 standard amino acids
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# -----------------------------------
# 3. CALCULATE AAC FEATURES
# -----------------------------------

def calculate_aac(sequence):

    sequence = sequence.upper()

    valid = [aa for aa in sequence if aa in amino_acids]

    if len(valid) == 0:
        raise ValueError("Sequence has no standard amino acids.")

    features = []

    for aa in amino_acids:
        features.append(valid.count(aa) / len(valid))

    return np.array(features)


# Calculate features for every protein
protein_features = {}

for protein_id, sequence in sequence_dict.items():

    protein_features[protein_id] = calculate_aac(sequence)

print("AAC features calculated for all proteins.")
print("Number of proteins:", len(protein_features))
print("Features per protein:", len(amino_acids))

# -----------------------------------
# 4. CONVERT PAIRS INTO 40 FEATURES
# -----------------------------------

def create_pair_features(data):

    X = []

    for _, row in data.iterrows():

        protein_a = row["protein_a"]
        protein_b = row["protein_b"]

        feature_a = protein_features[protein_a]
        feature_b = protein_features[protein_b]

        difference = np.abs(feature_a - feature_b)
        product = feature_a * feature_b

        combined = np.concatenate(
            [difference, product]
        )

        X.append(combined)

    return np.array(X)


# -----------------------------------
# 5. MODEL TRAINING AND EVALUATION
# -----------------------------------

def train_and_evaluate(train_file, test_file, split_name):

    train_data = pd.read_csv(train_file, dtype=str)
    test_data = pd.read_csv(test_file, dtype=str)

    y_train = train_data["label"].astype(int)
    y_test = test_data["label"].astype(int)

    X_train = create_pair_features(train_data)
    X_test = create_pair_features(test_data)

    # Fit scaler ONLY on training data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create Logistic Regression model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluate
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
    auc = roc_auc_score(y_test, probabilities)

    print("\n------------------------------")
    print(split_name)
    print("------------------------------")

    print("Training pairs:", len(train_data))
    print("Testing pairs:", len(test_data))

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    return {
        "split": split_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }


# -----------------------------------
# 6. RUN BOTH EXPERIMENTS
# -----------------------------------

random_results = train_and_evaluate(
    os.path.join(processed_folder, "random_train.csv"),
    os.path.join(processed_folder, "random_test.csv"),
    "RANDOM SPLIT"
)

disjoint_results = train_and_evaluate(
    os.path.join(processed_folder, "disjoint_train.csv"),
    os.path.join(processed_folder, "disjoint_test.csv"),
    "PROTEIN-DISJOINT SPLIT"
)

# -----------------------------------
# 7. SAVE RESULTS
# -----------------------------------

results = pd.DataFrame(
    [random_results, disjoint_results]
)

results_path = os.path.join(
    project_folder, "results", "baseline_results.csv"
)

results.to_csv(results_path, index=False)

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print(results.to_string(index=False))

print("\nResults saved to:", results_path)
