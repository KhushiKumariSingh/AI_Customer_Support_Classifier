# Customer Support Ticket Classification System

An end-to-end Machine Learning project that automatically classifies customer support tickets into different categories based on their descriptions.

The system uses **TF-IDF Vectorization and Logistic Regression** to predict ticket categories and provides an interactive interface using **Streamlit**.

## 1. Project Overview

Customer support teams receive a large number of queries related to login problems, application errors, account updates, reports, and performance issues.

Manually sorting these tickets can be time-consuming. This project aims to automate the classification process by analyzing ticket descriptions and predicting their appropriate categories.

**Project Workflow:**

Ticket Description → Text Preprocessing → TF-IDF Vectorization → Logistic Regression → Category Prediction → Suggested Response

## 2. Key Features

- Automatic classification of customer support tickets.
- Text cleaning and preprocessing.
- TF-IDF-based text feature extraction.
- Logistic Regression classification model.
- Prediction confidence scores.
- Exploratory Data Analysis (EDA) with visualizations.
- Model evaluation using classification metrics.
- Interactive Streamlit web application.
- Rule-based suggested responses for predicted categories.
- Command-line prediction support.

## 3. Ticket Categories

The system classifies tickets into five categories:

- Login Issue
- Application Error
- Report
- Account Update
- Performance

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | Machine Learning and text vectorization |
| TF-IDF | Text feature extraction |
| Logistic Regression | Classification algorithm |
| Matplotlib & Seaborn | Data visualization |
| Joblib | Model serialization |
| Streamlit | Interactive web application |
| Jupyter Notebook | Data analysis and experimentation |

## 5. Dataset Information

- Dataset file: `data/tickets.csv`
- Total raw records: 212
- Records after duplicate removal: 204
- Number of categories: 5
- Features include ticket ID, description, category, priority, and status.

**Note:** The dataset is synthetic and intended for educational and experimental purposes. It does not contain real company support records.

## 6. Project Structure

```text
AI_Customer_Support_Classifier/
│
├── data/
│   └── tickets.csv
│
├── model/
│   └── trained_model.joblib
│
├── notebooks/
│   └── analysis.ipynb
│
├── screenshots/
│   ├── category_distribution.png
│   ├── priority_distribution.png
│   ├── status_distribution.png
│   ├── confusion_matrix.png
│   ├── dataset.png
│   ├── analysis.png
│   └── prediction.png
│
├── src/
│   ├── create_dataset.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train.py
│   ├── predict.py
│   ├── responses.py
│   ├── make_figures.py
│   └── app.py
│
├── requirements.txt
├── README.md
└── report.pdf
```

## 7. Installation and Setup

**Step 1: Clone the repository**

```bash
git clone https://github.com/Aliahjamil/AI-Customer-Support-Classifier.git
```

**Step 2: Navigate to the project directory**

```bash
cd AI-Customer-Support-Classifier
```

**Step 3: Create a virtual environment**

```bash
python -m venv venv
```

**Step 4: Activate the environment (Windows)**

```bash
venv\Scripts\activate
```

**Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

## 8. Run the Project

Run the following commands from the project root directory.

**Generate Dataset (Optional)**

```bash
python src/create_dataset.py
```

**Preprocess Data**

```bash
python src/preprocessing.py
```

**Perform Exploratory Data Analysis**

```bash
python src/eda.py
```

**Train the Machine Learning Model**

```bash
python src/train.py
```

**Run Predictions**

```bash
python src/predict.py
```

**Launch the Streamlit Application**

```bash
streamlit run src/app.py
```

Open the application in your browser:

`http://localhost:8501`

## 9. Model Performance

The Logistic Regression model was evaluated on a test dataset of 41 tickets.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 85.4% |
| Macro Precision | 88.3% |
| Macro Recall | 85.2% |
| Macro F1-Score | 85.5% |
| 5-Fold Cross-Validation Accuracy | 90.2% |

**Note:** These results are based on a small synthetic dataset. Performance on real-world customer support data may differ.

## 10. Application Screenshots

The `screenshots/` directory contains visualizations and examples of the project, including:

- Category distribution
- Priority distribution
- Status distribution
- Confusion matrix
- Dataset overview
- Exploratory data analysis
- Ticket prediction results

## 11. Future Improvements

- Train the model on larger real-world datasets.
- Experiment with advanced NLP techniques and classification algorithms.
- Improve prediction confidence and model generalization.
- Integrate an LLM-based response generation system.
- Add ticket priority prediction.
- Deploy the application on a cloud platform.

## 12. Author

**Khushi Kumari Singh**
