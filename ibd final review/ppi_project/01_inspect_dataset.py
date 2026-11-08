
import pandas as pd

# Dataset paths
actions_path = "data/raw/SHS27k.actions.txt"
sequences_path = "data/raw/SHS27k.seqs.tsv"

# Read the files without assuming their column structure
actions = pd.read_csv(
    actions_path, sep="\t", header=0, dtype=str
)


sequences = pd.read_csv(
    sequences_path,
    sep="\t",
    header=None,
    dtype=str
)

print("INTERACTION DATA")
print(actions.head())
print("\nInteraction shape:", actions.shape)

print("\nSEQUENCE DATA")
print(sequences.head())
print("\nSequence shape:", sequences.shape)

print("\nMissing values in interactions:")
print(actions.isnull().sum())

print("\nMissing values in sequences:")
print(sequences.isnull().sum())
print("\n========== CORRECTED INTERACTION DATA ==========")

print("Column names:")
print(actions.columns.tolist())

print("\nFirst 5 actual records:")
print(actions.head().to_string())

print("\nDataset shape:")
print(actions.shape)

print("\nMissing values:")
print(actions.isnull().sum())

print("\nUnique values in each column:")
for column in actions.columns:
    print("\n", column)
    print(actions[column].value_counts(dropna=False).head(10))
print("\n========== DETAILED DATASET INSPECTION ==========")

# Display all column names
print("\nColumn names:")
for col in actions.columns:
    print(col)

# Display 10 complete interaction records
print("\nFirst 10 interaction records:")
print(actions.head(10).to_string(index=False))

# Count unique proteins in interaction data
proteins_in_actions = set(actions["item_id_a"]) | set(actions["item_id_b"])

print("\nUnique proteins in interaction data:")
print(len(proteins_in_actions))

# Compare with sequence dataset
sequence_ids = set(sequences.iloc[:, 0])

print("\nProteins with available sequences:")
print(len(sequence_ids))

print("\nProteins without sequences:")
print(len(proteins_in_actions - sequence_ids))

print("\nSequences not found in interactions:")
print(len(sequence_ids - proteins_in_actions))
print("\n========== PAIR ANALYSIS ==========")

# Treat A-B and B-A as the same pair
pair_keys = actions.apply(
    lambda row: tuple(sorted([
        row["item_id_a"],
        row["item_id_b"]
    ])),
    axis=1
)

actions["pair_key"] = pair_keys

print("Total interaction records:", len(actions))

print("Unique protein pairs:",
      actions["pair_key"].nunique())

print("Repeated pair records:",
      len(actions) - actions["pair_key"].nunique())

print("\nInteraction modes:")
print(actions["mode"].value_counts())

print("\nUnique pairs in each interaction mode:")
print(
    actions.groupby("mode")["pair_key"]
    .nunique()
)
