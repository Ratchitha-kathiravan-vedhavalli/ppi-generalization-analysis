
import pandas as pd
import os

project_folder = os.path.dirname(os.path.abspath(__file__))

actions_path = os.path.join(
    project_folder, "data", "raw", "SHS27k.actions.txt"
)

actions = pd.read_csv(
    actions_path, sep="\t", header=0, dtype=str
)

# Create an order-independent protein pair
actions["protein_a"] = actions[
    ["item_id_a", "item_id_b"]
].min(axis=1)

actions["protein_b"] = actions[
    ["item_id_a", "item_id_b"]
].max(axis=1)

# Count records for each unique pair
pair_summary = actions.groupby(
    ["protein_a", "protein_b"]
).agg(
    number_of_records=("mode", "size"),
    interaction_modes=(
        "mode",
        lambda x: ", ".join(sorted(set(x.dropna())))
    )
).reset_index()

print("Total unique protein pairs:", len(pair_summary))

print("\nFirst 15 unique protein pairs:")
print(pair_summary.head(15).to_string(index=False))

print("\nPairs with multiple records:")
print(
    (pair_summary["number_of_records"] > 1).sum()
)

# Save the summary
output_path = os.path.join(
    project_folder, "results", "pair_summary.csv"
)

pair_summary.to_csv(output_path, index=False)

print("\nPair summary saved successfully!")
