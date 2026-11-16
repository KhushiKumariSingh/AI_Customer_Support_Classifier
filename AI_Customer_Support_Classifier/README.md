# Customer Support Ticket Classification System

An AI/ML project that reads a customer support ticket and predicts its category
(**Login Issue, Application Error, Report, Account Update, Performance**).
It uses **TF-IDF + Logistic Regression** and includes a **Streamlit** web app.

> **Honest notes**
> - The dataset is **synthetic**. It was written with AI assistance and is not real company data.
> - This project was planned and written with the help of an AI assistant (Claude by Anthropic).
>   The author ran, tested and is responsible for understanding the code.
> - All metrics below come from running the code. Real-world accuracy would likely be lower.

---

## 1. Problem Statement

A company receives customer support requests through email, chat and other channels.
Each ticket has an ID, description, priority, status and category. The company wants to
**automatically predict the category of a new ticket from its description**, for example:

| Ticket description | Category |
|---|---|
| I forgot my password and cannot login | Login Issue |
| The application is showing an error while saving data | Application Error |
| Please help me generate my monthly sales report | Report |
| I need to change my registered mobile number | Account Update |
| The application is very slow today | Performance |

## 2. Project Overview

```
Ticket text -> Cleaning -> TF-IDF -> Logistic Regression -> Category + Confidence -> Streamlit UI
```

The project covers data creation, preprocessing, EDA, model training, evaluation,
new-ticket prediction, a web interface, and an optional rule-based auto-reply.

## 3. Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.x | Programming language |
| Pandas, NumPy | Data handling |
| Matplotlib, Seaborn | Charts |
| Scikit-learn | TF-IDF, Logistic Regression, metrics |
| Joblib | Saving the trained model |
| Streamlit | Web interface |
| Jupyter Notebook | Step-by-step analysis |

## 4. Dataset Details

- File: `data/tickets.csv`, created by `src/create_dataset.py`
- Columns: `ticket_id`, `ticket_description`, `category`, `priority`, `status`
- 212 raw rows, 204 after removing 8 duplicate descriptions
- 5 categories: Login Issue (46), Application Error (44), Account Update (40),
  Performance (38), Report (36) after cleaning
- A few missing `priority`/`status` values and duplicates were added on purpose to practise cleaning
- Category is based on the meaning of the text. Priority and status are random metadata used only for EDA

## 5. Folder Structure

```
AI_Customer_Support_Classifier/
├── data/
│   └── tickets.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── create_dataset.py   # builds tickets.csv
│   ├── preprocessing.py    # load + clean data
│   ├── eda.py              # counts and charts
│   ├── train.py            # train, evaluate, save model
│   ├── predict.py          # predict new tickets
│   ├── responses.py        # rule-based auto-reply
│   ├── make_figures.py     # renders dataset/analysis/prediction images
│   └── app.py              # Streamlit app
├── model/
│   └── trained_model.joblib
├── screenshots/            # charts, confusion matrix, dataset/analysis/prediction images
├── requirements.txt
├── README.md
└── report.pdf
```

## 6. Installation (Windows + Anaconda)

Open **Anaconda Prompt**, then:

```
cd path\to\AI_Customer_Support_Classifier
conda create -n ticket_env python=3.10 -y
conda activate ticket_env
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`: pandas, numpy, matplotlib, seaborn,
scikit-learn, joblib, streamlit, notebook.

The project was developed and tested with recent versions of these libraries
(for example scikit-learn 1.9, pandas 3.0). Older versions listed as minimums should also work.

## 7. How to Run

Run every command from the project root folder.

```
python src\create_dataset.py    # optional: regenerate data\tickets.csv
python src\preprocessing.py     # inspect and clean the data
python src\eda.py               # EDA, saves charts to screenshots\
python src\train.py             # trains, evaluates, saves model\trained_model.joblib
python src\make_figures.py     # optional: renders dataset/analysis/prediction images
```

### Predict new tickets (command-line interface)

```
python src\predict.py
```

Type a ticket description when asked and press Enter (type `exit` to quit):

```
========================================
 Customer Support Classifier
========================================
Type a ticket description. Type 'exit' to quit.

Enter Ticket Description:
> I am unable to login because my password is not working.

Predicted Category: Login Issue
Confidence: 46.5%

Suggested Response:
Hello, sorry you are having trouble signing in. Please try resetting your password ...
```

Other ways to run it:

```
python src\predict.py "I forgot my password and cannot sign in"   # one ticket, then exit
python src\predict.py --test                                       # runs 10 new test tickets
```

The 10 test tickets are new (not in `tickets.csv`). All 10 were classified correctly, with
confidence between about 24% and 65%. The confidence values are modest because the dataset is small.
This is a quick sanity check, not a measure of real-world accuracy.

### Run the Streamlit app

```
streamlit run src\app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

Sample input: `The app crashes when I click the save button.`
Sample output: **Application Error**, with a confidence score, a probability bar chart and a suggested reply.

### Run the notebook

```
jupyter notebook
```

Open `notebooks\analysis.ipynb` and choose **Kernel > Restart & Run All**.

## 8. Screenshots

The `screenshots/` folder contains `category_distribution.png`, `priority_distribution.png`,
`status_distribution.png`, `confusion_matrix.png`, and three images rendered by `src/make_figures.py`
from real code output: `dataset.png`, `analysis.png` and `prediction.png`.
These three are generated images, not screen captures.

## 9. Results (test set of 41 tickets)

| Metric | Value |
|---|---|
| Accuracy | 0.854 |
| Precision (macro) | 0.883 |
| Recall (macro) | 0.852 |
| F1 score (macro) | 0.855 |
| 5-fold cross-validation accuracy | 0.902 (+/- 0.041) |

Limitations: the dataset is small and synthetic, the test set is tiny (one ticket is about 2.4%),
and predicted probabilities are low and soft. See `report.pdf` for details.

## 10. Optional Feature: Auto-Reply

`src/responses.py` suggests a reply template for each predicted category (rule-based).
It could later be upgraded by sending the ticket and category to an LLM API and letting a human review the reply.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Use **Anaconda Prompt** |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: tickets.csv` | Run commands from the project root; check `data\tickets.csv` exists |
| "Model not found" | Run `python src\train.py` first |
| `streamlit` not recognized | `python -m streamlit run src\app.py` |
| Port 8501 busy | `streamlit run src\app.py --server.port 8502` |
| Notebook cannot import `preprocessing` | Keep `analysis.ipynb` inside `notebooks\` |

## 12. AI Assistance Statement

An AI assistant (Claude by Anthropic) helped plan the project, write the synthetic dataset and
code, and draft this documentation. All code was run and the reported results were produced by that code.
