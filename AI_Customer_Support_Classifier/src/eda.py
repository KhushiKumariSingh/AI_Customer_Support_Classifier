"""
eda.py
------
Exploratory Data Analysis (EDA) for the ticket dataset.
Prints counts and saves 3 charts into the screenshots folder.

Run from the project root:
    python src/eda.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Make "from preprocessing import ..." work when running this file directly
sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import PROJECT_ROOT, preprocess_data

SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def print_summary(df):
    """Print ticket counts and distributions."""
    print("Total tickets:", len(df))

    for column in ["category", "priority", "status"]:
        counts = df[column].value_counts()
        percentages = (df[column].value_counts(normalize=True) * 100).round(1)
        print(f"\n=== {column.title()}-wise ticket count ===")
        print(counts.to_string())
        print(f"\n{column.title()} distribution (%)")
        print(percentages.to_string())

    print("\n=== Category vs Priority ===")
    print(df.groupby(["category", "priority"]).size().unstack(fill_value=0))


def plot_count_chart(df, column, title, order, filename, palette):
    """Draw one bar chart with the count written on top of each bar."""
    plt.figure(figsize=(9, 5))
    ax = sns.countplot(data=df, x=column, order=order, hue=column,
                       palette=palette, legend=False)
    for bar in ax.patches:
        ax.annotate(int(bar.get_height()),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(column.title())
    ax.set_ylabel("Number of tickets")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / filename, dpi=150)
    plt.show()


def main():
    df = preprocess_data()
    print_summary(df)

    plot_count_chart(df, "category", "Ticket Category Distribution",
                     df["category"].value_counts().index,
                     "category_distribution.png", "viridis")

    plot_count_chart(df, "priority", "Ticket Priority Distribution",
                     ["High", "Medium", "Low", "Unknown"],
                     "priority_distribution.png", "rocket")

    plot_count_chart(df, "status", "Ticket Status Distribution",
                     ["Open", "In Progress", "Resolved", "Closed", "Unknown"],
                     "status_distribution.png", "mako")

    print("\nCharts saved in:", SCREENSHOT_DIR)


if __name__ == "__main__":
    main()
