"""
preprocessing.py
----------------
Loads and cleans the customer support ticket data.

Other files (train.py, predict.py, app.py) import clean_text() from here,
so the SAME cleaning is used for training and for new tickets.

Run from the project root to see every preprocessing step:
    python src/preprocessing.py
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tickets.csv"


def load_data(file_path=DATA_PATH):
    """Step 1: Read the CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)


def inspect_data(df):
    """Step 2: Look at the data (first rows, shape, info, missing, duplicates)."""
    print("=== First 5 records ===")
    print(df.head())

    print("\n=== Shape (rows, columns) ===")
    print(df.shape)

    print("\n=== Dataset info ===")
    df.info()

    print("\n=== Missing values per column ===")
    print(df.isnull().sum())

    # Two tickets with the same text but different ticket_id are still duplicates,
    # so we check the description column only.
    print("\n=== Duplicate descriptions ===")
    print(df.duplicated(subset="ticket_description").sum())


def handle_missing_values(df):
    """
    Step 3: Handle missing values.
    - ticket_description / category missing -> drop the row
      (we cannot learn from a ticket without text or a label).
    - priority / status missing -> fill with 'Unknown'
      (these columns are only used for analysis, so we keep the ticket).
    """
    df = df.dropna(subset=["ticket_description", "category"]).copy()
    df["priority"] = df["priority"].fillna("Unknown")
    df["status"] = df["status"].fillna("Unknown")
    return df


def remove_duplicates(df):
    """Step 4: Remove repeated tickets (same description), keep the first one."""
    return df.drop_duplicates(subset="ticket_description", keep="first").copy()


def clean_text(text):
    """
    Step 5: Clean one ticket description.
    1. Convert to lowercase.
    2. Replace anything that is not a letter, number or space with a space.
    3. Collapse multiple spaces into one and trim the ends.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data(file_path=DATA_PATH):
    """Run the full pipeline and return a clean DataFrame."""
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df["clean_description"] = df["ticket_description"].apply(clean_text)
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw_df = load_data()
    inspect_data(raw_df)

    clean_df = preprocess_data()
    print("\n=== After cleaning ===")
    print("Rows before:", len(raw_df), "| Rows after:", len(clean_df))
    print("Missing values left:\n", clean_df.isnull().sum())
    print("\nOriginal vs cleaned text:")
    for i in range(3):
        print("-", clean_df.loc[i, "ticket_description"])
        print("  ->", clean_df.loc[i, "clean_description"])
