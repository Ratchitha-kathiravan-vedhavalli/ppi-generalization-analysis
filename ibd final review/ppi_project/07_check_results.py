
import pandas as pd
import os

folder = os.path.dirname(os.path.abspath(__file__))
processed = os.path.join(folder, "processed")

files = [
    "random_train.csv",
    "random_test.csv",
    "disjoint_train.csv",
    "disjoint_test.csv"
]

for filename in files:

    path = os.path.join(processed, filename)
    data = pd.read_csv(path)

    print("\n----------------------")
    print(filename)
    print("----------------------")

    print("Total pairs:", len(data))
    print("\nClass distribution:")
    print(data["label"].value_counts().sort_index())

    print("\nClass percentages:")
    print(
        (data["label"].value_counts(normalize=True) * 100)
        .sort_index()
    )
