
import pandas as pd
import os

# Find the project folder
project_folder = os.path.dirname(os.path.abspath(__file__))

# Load the pair summary created in Step 2
input_path = os.path.join(
    project_folder, "results", "pair_summary.csv"
)

pairs = pd.read_csv(input_path)

# Every pair in this file has at least one recorded interaction
pairs["label"] = 1

# Keep the columns needed for binary classification
positive_pairs = pairs[
    ["protein_a", "protein_b", "label", "interaction_modes"]
].copy()

# Save the positive dataset
output_path = os.path.join(
    project_folder, "processed", "positive_pairs.csv"
)

positive_pairs.to_csv(output_path, index=False)

print("Positive dataset created successfully!")

print("\nTotal positive protein pairs:", len(positive_pairs))
print("\nFirst 10 positive pairs:")
print(positive_pairs.head(10).to_string(index=False))

print("\nSaved to:", output_path)
