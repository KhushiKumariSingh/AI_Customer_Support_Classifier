"""
make_figures.py
---------------
Creates three images in the screenshots folder from REAL code output:
    dataset.png     - first rows of the dataset
    analysis.png    - category, priority and status charts side by side
    prediction.png  - results for the 10 new test tickets

NOTE: these are images rendered by code, not screen captures. For the app,
take real screenshots of the Streamlit page and save them in screenshots/.

Run from the project root:
    python src/make_figures.py
"""

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from predict import TEST_TICKETS, load_model, predict_ticket
from preprocessing import PROJECT_ROOT, load_data, preprocess_data

OUT_DIR = PROJECT_ROOT / "screenshots"
OUT_DIR.mkdir(exist_ok=True)


def draw_table(rows, columns, col_widths, title, filename, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    table = ax.table(cellText=rows, colLabels=columns, colWidths=col_widths,
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.9)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", weight="bold")
    plt.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_dataset_image():
    raw = load_data().head(10)
    rows = [[row.ticket_id, textwrap.shorten(row.ticket_description, 62),
             row.category, row.priority, row.status] for row in raw.itertuples()]
    draw_table(rows, ["ticket_id", "ticket_description", "category", "priority", "status"],
               [0.1, 0.44, 0.19, 0.12, 0.13], "tickets.csv: first 10 records",
               "dataset.png", (12, 4.5))


def make_analysis_image():
    df = preprocess_data()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    charts = [("category", df["category"].value_counts().index, "viridis", "Category distribution"),
              ("priority", ["High", "Medium", "Low", "Unknown"], "rocket", "Priority distribution"),
              ("status", ["Open", "In Progress", "Resolved", "Closed", "Unknown"], "mako",
               "Status distribution")]
    for axis, (column, order, palette, title) in zip(axes, charts):
        sns.countplot(data=df, x=column, order=order, hue=column, palette=palette,
                      legend=False, ax=axis)
        axis.bar_label(axis.containers[0])
        axis.set_title(title, weight="bold")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle(f"Exploratory data analysis ({len(df)} tickets)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "analysis.png", dpi=150)
    plt.close(fig)


def make_prediction_image():
    model = load_model()
    rows = []
    for text, expected in TEST_TICKETS:
        predicted, confidence = predict_ticket(model, text)
        rows.append([textwrap.fill(text, 55), expected, predicted, f"{confidence:.1f}%",
                     "PASS" if predicted == expected else "FAIL"])
    draw_table(rows, ["Test input", "Expected", "Predicted", "Confidence", "Result"],
               [0.46, 0.15, 0.15, 0.12, 0.1], "Predictions for 10 new tickets",
               "prediction.png", (13, 6.5))


if __name__ == "__main__":
    make_dataset_image()
    make_analysis_image()
    make_prediction_image()
    print("Images saved in", OUT_DIR)
