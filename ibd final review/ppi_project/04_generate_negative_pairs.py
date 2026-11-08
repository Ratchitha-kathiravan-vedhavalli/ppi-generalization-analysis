
import pandas as pd
import os
import random

# Find project folder
project_folder = os.path.dirname(os.path.abspath(__file__))

# File paths
positive_path = os.path.join(
    project_folder, "processed", "positive_pairs.csv"
)

sequence_path = os.path.join(
    project_folder, "data", "raw", "SHS27k.seqs.tsv"
)

# Load positive pairs
positive = pd.read_csv(
    positive_path, dtype=str
)

# Load protein IDs from sequence file
sequences = pd.read_csv(
    sequence_path, sep="\t", header=None, dtype=str
)

protein_ids = sequences.iloc[:, 0].tolist()

# Store all known positive pairs
positive_set = set(
    zip(positive["protein_a"], positive["protein_b"])
)

# Number of negative pairs to generate
number_of_negatives = len(positive)

# Random generator for reproducibility
rng = random.Random(42)

negative_set = set()

# Generate pairs that are not in the positive dataset
while len(negative_set) < number_of_negatives:

    protein_a = rng.choice(protein_ids)
    protein_b = rng.choice(protein_ids)

    # Do not allow a protein to pair with itself
    if protein_a == protein_b:
        continue

    # Keep pair order consistent
    pair = tuple(sorted([protein_a, protein_b]))

    # Exclude known positive pairs and duplicates
    if pair not in positive_set and pair not in negative_set:
        negative_set.add(pair)

# Convert negative pairs into a DataFrame
negative = pd.DataFrame(
    list(negative_set),
    columns=["protein_a", "protein_b"]
)

negative["label"] = 0

# Save negative dataset
negative_path = os.path.join(
    project_folder, "processed", "negative_pairs_random.csv"
)

negative.to_csv(negative_path, index=False)

# Combine positive and negative pairs
positive_binary = positive[
    ["protein_a", "protein_b", "label"]
]

binary_dataset = pd.concat(
    [positive_binary, negative],
    ignore_index=True
)

# Shuffle the dataset
binary_dataset = binary_dataset.sample(
    frac=1, random_state=42
).reset_index(drop=True)

# Save combined dataset
combined_path = os.path.join(
    project_folder, "processed", "binary_dataset_random.csv"
)

binary_dataset.to_csv(combined_path, index=False)

print("Negative dataset created successfully!")

print("\nPositive pairs:", len(positive_binary))
print("Negative pairs:", len(negative))
print("Total pairs:", len(binary_dataset))

print("\nLabel distribution:")
print(binary_dataset["label"].value_counts())

print("\nSaved files:")
print(negative_path)
print(combined_path)
