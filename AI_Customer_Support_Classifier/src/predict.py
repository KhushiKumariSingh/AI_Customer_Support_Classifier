"""
predict.py
----------
Loads the trained model and predicts the category of a new ticket.

Usage (from the project root):
    python src/predict.py                         -> interactive: type a ticket, get a category
    python src/predict.py "I forgot my password"  -> predicts one ticket and exits
    python src/predict.py --test                  -> runs the 10 new test tickets
"""

import sys
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import PROJECT_ROOT, clean_text, preprocess_data
from responses import get_auto_reply

MODEL_PATH = PROJECT_ROOT / "model" / "trained_model.joblib"


def load_model(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'python src/train.py' first."
        )
    return joblib.load(model_path)


def predict_ticket(model, ticket_text):
    """Return (predicted_category, confidence_percent) for one ticket."""
    cleaned_text = clean_text(ticket_text)           # same cleaning as training
    probabilities = model.predict_proba([cleaned_text])[0]
    best_index = probabilities.argmax()
    predicted_category = model.classes_[best_index]
    confidence = probabilities[best_index] * 100
    return predicted_category, confidence


# New tickets written AFTER training. They are not in tickets.csv.
TEST_TICKETS = [
    ("My password was changed by someone and now I cannot get into my account.", "Login Issue"),
    ("The mobile app crashes with an error whenever I tap the pay now button.", "Application Error"),
    ("Please send me the quarterly expenses summary for the finance team.", "Report"),
    ("I moved to a new city and want to update my address on my profile.", "Account Update"),
    ("The website takes ages to open and the pages keep lagging.", "Performance"),
    ("Verification code never reaches my phone so I am unable to sign in.", "Login Issue"),
    ("Getting an exception while uploading my profile photo, the app shows an error.", "Application Error"),
    ("Need a spreadsheet listing customers who cancelled their subscription last month.", "Report"),
    ("Can you change the account holder name to my company name?", "Account Update"),
    ("The reports page is very slow to load when I select a full year.", "Performance"),
]


def run_tests(model):
    # Safety check: make sure test tickets are really new
    existing = set(preprocess_data()["clean_description"])
    correct = 0
    print(f"{'#':<3}{'Expected':<19}{'Predicted':<19}{'Conf.':<8}{'Result':<8}Input")
    print("-" * 110)
    for number, (text, expected) in enumerate(TEST_TICKETS, start=1):
        assert clean_text(text) not in existing, "Test ticket exists in training data!"
        predicted, confidence = predict_ticket(model, text)
        result = "PASS" if predicted == expected else "FAIL"
        correct += result == "PASS"
        print(f"{number:<3}{expected:<19}{predicted:<19}{confidence:5.1f}%  {result:<8}{text}")
    print(f"\nCorrect: {correct} out of {len(TEST_TICKETS)}")


def interactive_mode(model):
    """Simple command-line interface: type a ticket, get the predicted category."""
    print("=" * 40)
    print(" Customer Support Classifier")
    print("=" * 40)
    print("Type a ticket description. Type 'exit' to quit.")
    while True:
        try:
            ticket = input("\nEnter Ticket Description:\n> ").strip()
        except EOFError:
            break
        if ticket.lower() in ("exit", "quit", "q"):
            break
        if not ticket:
            print("Please enter a ticket description.")
            continue
        category, score = predict_ticket(model, ticket)
        print(f"\nPredicted Category: {category}")
        print(f"Confidence: {score:.1f}%")
        print(f"\nSuggested Response:\n{get_auto_reply(category)}")


if __name__ == "__main__":
    trained_model = load_model()
    arguments = sys.argv[1:]
    if arguments == ["--test"]:
        run_tests(trained_model)
    elif arguments:
        ticket = " ".join(arguments)
        category, score = predict_ticket(trained_model, ticket)
        print(f"Ticket    : {ticket}")
        print(f"Category  : {category}")
        print(f"Confidence: {score:.1f}%")
    else:
        interactive_mode(trained_model)
