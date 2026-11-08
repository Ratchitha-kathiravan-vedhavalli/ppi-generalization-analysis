
import pandas as pd
import os
import random

project_folder = os.path.dirname(os.path.abspath(__file__))

# File paths
positive_path = os.path.join(
    project_folder, "processed", "positive_pairs.csv"
)

binary_path = os.path.join(
    project_folder, "processed", "binary_dataset_random.csv"
)

sequence_path = os.path.join(
    project_folder, "data", "raw", "SHS27k.seqs.tsv"
)

# Load datasets
positive = pd.read_csv(positive_path, dtype=str)
binary = pd.read_csv(binary_path, dtype=str)
sequences = pd.read_csv(
    sequence_path, sep="\t", header=None, dtype=str
)

# Convert labels to integers
binary["label"] = binary["label"].astype(int)
positive["label"] = positive["label"].astype(int)

# --------------------------------
# PART A: RANDOM 80:20 SPLIT
# --------------------------------

rng = random.Random(42)

train_parts = []
test_parts = []

for label in [0, 1]:
    class_data = binary[binary["label"] == label].copy()

    # Shuffle each class
    class_data = class_data.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    split_index = int(len(class_data) * 0.8)

    train_parts.append(class_data.iloc[:split_index])
    test_parts.append(class_data.iloc[split_index:])

random_train = pd.concat(train_parts).sample(
    frac=1, random_state=42
).reset_index(drop=True)

random_test = pd.concat(test_parts).sample(
    frac=1, random_state=42
).reset_index(drop=True)

random_train.to_csv(
    os.path.join(project_folder, "processed",
                 "random_train.csv"), index=False
)

random_test.to_csv(
    os.path.join(project_folder, "processed",
                 "random_test.csv"), index=False
)

# --------------------------------
# PART B: PROTEIN-DISJOINT SPLIT
# --------------------------------

protein_ids = sequences.iloc[:, 0].tolist()

rng.shuffle(protein_ids)

# Divide proteins: 80% training, 20% testing
split_index = int(len(protein_ids) * 0.8)

train_proteins = set(protein_ids[:split_index])
test_proteins = set(protein_ids[split_index:])

# Keep positive pairs whose two proteins belong
# entirely to the same group
train_positive = positive[
    positive["protein_a"].isin(train_proteins) &
    positive["protein_b"].isin(train_proteins)
].copy()

test_positive = positive[
    positive["protein_a"].isin(test_proteins) &
    positive["protein_b"].isin(test_proteins)
].copy()

# Count positive pairs crossing the two groups
cross_pairs = positive[
    ~(
        (positive["protein_a"].isin(train_proteins) &
         positive["protein_b"].isin(train_proteins)) |
        (positive["protein_a"].isin(test_proteins) &
         positive["protein_b"].isin(test_proteins))
    )
]

# All known positive pairs, to avoid false negatives
positive_set = set(
    zip(positive["protein_a"], positive["protein_b"])
)

# Function to generate negative pairs inside one group
def generate_negatives(group_proteins, number, seed):
    local_rng = random.Random(seed)
    group_list = list(group_proteins)
    negatives = set()

    while len(negatives) < number:
        a = local_rng.choice(group_list)
        b = local_rng.choice(group_list)

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

# Generate negatives separately in each group
train_negative = generate_negatives(
    train_proteins, len(train_positive), 101
)

test_negative = generate_negatives(
    test_proteins, len(test_positive), 202
)

# Combine positive and negative pairs
disjoint_train = pd.concat(
    [
        train_positive[["protein_a", "protein_b", "label"]],
        train_negative
    ],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

disjoint_test = pd.concat(
    [
        test_positive[["protein_a", "protein_b", "label"]],
        test_negative
    ],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

# Save protein-disjoint datasets
disjoint_train.to_csv(
    os.path.join(project_folder, "processed",
                 "disjoint_train.csv"), index=False
)

disjoint_test.to_csv(
    os.path.join(project_folder, "processed",
                 "disjoint_test.csv"), index=False
)

# --------------------------------
# PART C: VERIFY THE SPLITS
# --------------------------------

train_ids = set(disjoint_train["protein_a"]) | set(
    disjoint_train["protein_b"]
)

test_ids = set(disjoint_test["protein_a"]) | set(
    disjoint_test["protein_b"]
)

print("RANDOM SPLIT")
print("Training pairs:", len(random_train))
print("Testing pairs:", len(random_test))

print("\nPROTEIN-DISJOINT SPLIT")
print("Training proteins:", len(train_proteins))
print("Testing proteins:", len(test_proteins))
print("Training positive pairs:", len(train_positive))
print("Testing positive pairs:", len(test_positive))
print("Cross-group positive pairs excluded:", len(cross_pairs))
print("Training dataset size:", len(disjoint_train))
print("Testing dataset size:", len(disjoint_test))

print("\nVERIFICATION")
print("Shared proteins between train and test:", 
      len(train_ids.intersection(test_ids)))

print("\nAll files saved successfully!")
