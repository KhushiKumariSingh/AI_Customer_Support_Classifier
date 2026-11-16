"""
train.py
--------
Trains the ticket classifier (TF-IDF + Logistic Regression),
evaluates it on unseen test data and saves the model.

Run from the project root:
    python src/train.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import PROJECT_ROOT, preprocess_data

MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "trained_model.joblib"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"
RANDOM_STATE = 42


def build_pipeline():
    """
    A Pipeline chains the steps together:
    1. TfidfVectorizer: turns text into numbers.
    2. LogisticRegression: learns to predict the category from those numbers.
    Using a Pipeline means the vectorizer is fitted ONLY on training data
    when we call pipeline.fit(X_train, y_train) -> no data leakage.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),  # single words + word pairs
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def plot_confusion_matrix(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Test Set)", fontsize=14, weight="bold")
    plt.xlabel("Predicted category")
    plt.ylabel("Actual category")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    plt.savefig(SCREENSHOT_DIR / "confusion_matrix.png", dpi=150)
    plt.show()


def main():
    # 1. Load clean data
    df = preprocess_data()
    X = df["clean_description"]
    y = df["category"]

    # 2. Split: 80% train, 20% test, same category proportions in both (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training tickets: {len(X_train)} | Test tickets: {len(X_test)}")

    # 3. Train (the vectorizer learns its vocabulary from training text only)
    model = build_pipeline()
    model.fit(X_train, y_train)

    # 4. Predict on the unseen test set
    y_pred = model.predict(X_test)

    # 5. Evaluate (all numbers come from the real predictions)
    print("\n=== Evaluation on test set ===")
    print(f"Accuracy           : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (macro)  : {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall (macro)     : {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1 score (macro)   : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall (weighted)   : {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 score (weighted) : {f1_score(y_test, y_pred, average='weighted'):.4f}")

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred))

    class_names = sorted(y.unique())
    print("=== Confusion matrix (rows = actual, columns = predicted) ===")
    print("Order:", class_names)
    print(confusion_matrix(y_test, y_pred, labels=class_names))

    # 6. Extra check: 5-fold cross-validation (a more stable estimate on small data)
    cv_scores = cross_val_score(build_pipeline(), X, y, cv=5, scoring="accuracy")
    print("\n=== 5-fold cross-validation accuracy ===")
    print("Scores:", [round(score, 3) for score in cv_scores])
    print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 7. Save the trained pipeline
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("\nModel saved to:", MODEL_PATH)

    plot_confusion_matrix(y_test, y_pred, class_names)


if __name__ == "__main__":
    main()
