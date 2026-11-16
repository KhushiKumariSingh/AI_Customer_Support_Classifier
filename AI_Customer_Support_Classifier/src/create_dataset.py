"""
create_dataset.py
-----------------
Builds data/tickets.csv for the Customer Support Ticket Classification project.

IMPORTANT (be honest in your report):
- The ticket descriptions below are SYNTHETIC. They were written by hand
  (with AI assistance) to look like real support tickets. They do not come
  from a real company.
- The CATEGORY of each ticket is decided by what the text is about.
  It is never assigned randomly.
- Priority and status are generated with a fixed random seed, because they
  are only extra information for EDA (they are NOT used for prediction).
- A few missing values and duplicate rows are added ON PURPOSE so the
  preprocessing step has something real to clean.

Run from the project root:
    python src/create_dataset.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Ticket descriptions written by category
# ---------------------------------------------------------------------------
TICKETS = {
    "Login Issue": [
        "I cannot log in to my account. It says invalid username or password but I am sure they are correct.",
        "Forgot my password and the reset email never arrives in my inbox.",
        "My account got locked after too many failed login attempts. Please unlock it.",
        "The login page keeps refreshing and sends me back to the sign in screen.",
        "I am not receiving the OTP on my mobile number when I try to sign in.",
        "Password reset link says it has expired as soon as I click it.",
        "Unable to sign in using my company email. The system says user not found.",
        "Two factor authentication code is rejected every time I enter it.",
        "Login works on my laptop but not on my phone. It just shows an error after entering the password.",
        "I changed my password yesterday and now the new password does not work.",
        "Single sign on button does nothing when I click it.",
        "Getting the message 'session expired' immediately after logging in.",
        "My account is showing as disabled and I cannot access the dashboard.",
        "Captcha on the login page does not load so I cannot sign in.",
        "Cannot sign in with my Google account, it keeps asking me to choose an account again and again.",
        "The verification email for my new account was never delivered, so I cannot log in.",
        "I keep getting logged out every few minutes and have to enter my password again.",
        "New employee here, my login credentials do not work on the portal.",
        "Login screen shows 'access denied' even though I was able to enter yesterday.",
        "I am locked out of the admin panel after resetting my password.",
        "Security question answer is not accepted so I cannot recover my account.",
        "Please help, I can't get into my account. Password is rejected.",
        "The sign in button is greyed out and I cannot click it.",
        "I receive a 'too many requests' message when I try to log in once.",
        "My username is not recognised after the company moved to the new portal.",
        "Reset password page opens but the submit button does not work.",
        "The authenticator app code does not match, so the system will not let me in.",
        "Unable to access my account from a new device, it keeps asking for verification that never comes.",
        "Login via mobile app fails with 'authentication failed' but the website works.",
        "My login was working until this morning. Now it just says incorrect credentials.",
        "I try to sign in and the page goes blank after I press enter.",
        "Temporary password from HR does not let me log in for the first time.",
        "Email address on the login form is rejected as invalid even though it is correct.",
        "Account locked for suspicious activity and I need to regain access urgently.",
        "I cannot log in using Microsoft sign in. It redirects me back to the login page.",
        "Remember me option is not working, I have to type my credentials every time.",
        "Customer portal login says 'your account is pending approval' for two days now.",
        "Login not working after the latest update, the password field does not accept input.",
        "I keep getting a wrong password error but I just reset it five minutes ago.",
        "The security code sent to my email arrives after it has already expired.",
        "Cannot log in on Safari, same account works fine in Chrome.",
        "Sign in fails with error code 401 every time I try.",
        "My teammate cannot login to the shared account, it asks for a password we never set.",
        "I get stuck on the sign in loading spinner and never reach my account.",
        "Account recovery option does not send any code to my backup email.",
        "Login credentials stopped working after I returned from leave.",
    ],
    "Application Error": [
        "The app crashes every time I click the submit button on the order form.",
        "I get a 500 internal server error when I open the invoice page.",
        "Error message 'something went wrong' appears when I try to save my changes.",
        "The mobile app closes by itself right after I open it.",
        "Getting a null pointer exception on the checkout screen.",
        "Uploading a file gives an unexpected error and nothing is saved.",
        "The page shows a blank white screen with a JavaScript error in the console.",
        "Application shows error code 404 when I click on the dashboard link.",
        "The save button throws an error 'database connection failed'.",
        "After the latest update the app freezes on the home screen and shows an error popup.",
        "Cannot submit the form, I keep seeing a validation error even though all fields are filled.",
        "The search feature returns 'unexpected error occurred' for any keyword.",
        "App crashed while I was editing a record and I lost my work.",
        "I see a bad gateway error when I try to open the settings page.",
        "Clicking the export button gives an error 'request failed'.",
        "The payment page throws an exception and the transaction does not complete.",
        "Error 'cannot read property of undefined' appears when I open the profile page.",
        "The application stops working when I try to attach a PDF document.",
        "A popup says 'unhandled error' every time I switch tabs.",
        "The notification feature is broken and the app displays an error message.",
        "Application throws a timeout error 504 while loading the order details.",
        "The web app displays 'service unavailable' when I try to create a new ticket.",
        "Buttons on the edit screen do nothing and a red error banner appears at the top.",
        "I get an error saying 'failed to load module' on opening the analytics tab.",
        "The app keeps force closing on my Android phone since the update.",
        "Clicking delete on a record causes the application to crash with an error.",
        "Error code E1023 is shown whenever I try to sync my data.",
        "The booking form gives a server error and does not confirm my reservation.",
        "I see a stack trace on the screen instead of the normal page.",
        "Adding an item to the cart results in an error and the cart stays empty.",
        "Copying a record produces a duplicate key error message.",
        "Dropdown menus show an error and do not load any options.",
        "The app displays 'invalid response from server' after I press save.",
        "Our users get an unknown error when they try to update the inventory page.",
        "The installer fails with an error and the application does not open.",
        "Camera feature inside the app causes it to crash immediately.",
        "I received an 'out of memory' error when opening the dashboard widget.",
        "The comment box throws an error and my message does not post.",
        "Getting an error popup saying 'API call failed' on the home page.",
        "Application hangs and shows 'not responding' when I open the tickets list.",
        "The import function stops halfway and displays an error without any explanation.",
        "Whenever I click the print icon the app shows an error and closes.",
        "The checkout page returns a 'bad request' error even with correct details.",
        "I get a runtime error on the screen when I press the back button.",
    ],
    "Report": [
        "I need a monthly sales report for the last quarter in Excel format.",
        "The weekly summary report is missing the data for Friday.",
        "Can you generate a report of all closed tickets for October?",
        "The totals in the revenue report do not match the numbers in my records.",
        "Please add a department column to the attendance report.",
        "I would like a custom report showing customer complaints by region.",
        "The downloaded report shows wrong dates in the date column.",
        "How can I schedule a daily report to be emailed to my manager?",
        "The pie chart in the dashboard report shows incorrect percentages.",
        "I need the annual performance report of my team in PDF format.",
        "The report export contains duplicate rows for several customers.",
        "Request for a report listing all users who joined this year.",
        "The inventory report is not showing the latest stock count.",
        "Please remove the confidential columns from the monthly finance report.",
        "Filters in the report are not applying, I always get the full data.",
        "I need a report that compares this month's expenses with last month.",
        "The summary report is missing the grand total at the bottom.",
        "Can the report be exported in CSV instead of PDF?",
        "Our regional report shows the wrong manager name for the East team.",
        "Please help me create a report for tickets resolved per agent.",
        "The scheduled weekly report did not arrive in my email this Monday.",
        "I need a report of overdue invoices older than thirty days.",
        "Numbers in the quarterly report are different from the dashboard figures.",
        "Requesting a report on customer satisfaction scores for the last six months.",
        "The report layout is broken and the columns overlap when printed.",
        "I want to add a bar graph to the sales report. How do I do that?",
        "The date range selector in reports does not include the last day of the month.",
        "Please share the audit report for user activity from last week.",
        "The generated report has no headers so it is hard to read.",
        "I need an executive summary report of project status for the board meeting.",
        "Report shows zero sales for a product that definitely sold units.",
        "Can I get a report that groups orders by city?",
        "The profit and loss report is missing the marketing expense category.",
        "Need a report of tickets by priority for the monthly review.",
        "Currency symbol in the financial report appears wrong for European customers.",
        "The report preview works but the downloaded file is empty.",
    ],
    "Account Update": [
        "I want to change the email address linked to my account.",
        "Please update my phone number to the new one I shared.",
        "My name is spelled wrong on my profile and I need it corrected.",
        "I need to update my billing address for the next invoice.",
        "How do I change my account password from the settings page?",
        "Please change my subscription plan from basic to premium.",
        "I would like to add my colleague as a second user on my account.",
        "Can you update the company name on our account to the new legal name?",
        "I need to remove an old payment card and add a new one.",
        "Please change the primary contact person for our organisation.",
        "I want to update my profile picture but cannot find the option.",
        "Kindly update my shipping address, I have moved to a new apartment.",
        "I need to change my username to something more professional.",
        "Please deactivate my old email and set the new one as default.",
        "I would like to change the language preference on my account to Spanish.",
        "My job title changed, please update it in my profile.",
        "Can you update my notification settings so I only get email alerts?",
        "Need to transfer account ownership to my manager.",
        "Please update the tax identification number on our company account.",
        "I want to close my account and delete my personal details.",
        "I need to change my date of birth, it was entered wrongly during registration.",
        "Please upgrade my account storage limit to the business plan.",
        "How can I update my time zone in my profile settings?",
        "I want to unlink my old mobile number and link a new one.",
        "Requesting a change of my registered email because I no longer use the old one.",
        "Please change the role of one of our users from viewer to editor.",
        "I need my account name changed after my marriage.",
        "Can you update our company logo on the account profile?",
        "I want to switch from monthly billing to yearly billing.",
        "Please add a new delivery address to my saved addresses.",
        "I need to update the emergency contact details in my profile.",
        "Kindly update my department from Sales to Marketing.",
        "I would like to change the display name that appears on my posts.",
        "Please remove one user from our team account, he has left the company.",
        "I need to edit the invoice email address so bills go to the finance team.",
        "Please merge my two accounts into one profile.",
        "Update my communication preferences, I do not want promotional messages.",
        "I need to correct the GST number on my account details.",
        "Please change my plan renewal date to the first of every month.",
        "I want to update the bank details linked to my account for refunds.",
    ],
    "Performance": [
        "The application is extremely slow and takes more than a minute to load the home page.",
        "Pages take forever to open since this morning.",
        "The system lags badly when more than ten people are using it.",
        "Dashboard loads very slowly, almost five minutes, during office hours.",
        "Searching for a customer record takes too long to return results.",
        "The website becomes unresponsive when I upload a large file.",
        "Everything has been slow since the last release, even simple clicks take seconds.",
        "The app uses too much memory and my computer becomes very slow.",
        "Our team experiences constant delays when saving records in the system.",
        "The mobile app is laggy and scrolling is not smooth.",
        "Opening the orders list is painfully slow compared to last week.",
        "The system hangs for a long time every time I apply a filter.",
        "Response time of the portal is very poor in the afternoon.",
        "The application takes too long to start up after I click the icon.",
        "High CPU usage on my machine whenever the app is open in the background.",
        "Loading the product catalogue takes a very long time and sometimes times out.",
        "The software is sluggish and keeps buffering while I switch between modules.",
        "Large data sets make the page freeze for several minutes before it responds.",
        "Video playback inside the app is slow and keeps stuttering.",
        "Everything on the website is crawling today, even the images load slowly.",
        "Sync takes hours to finish and slows down my whole device.",
        "The system speed drops sharply at the end of every month.",
        "API requests are taking more than ten seconds to respond.",
        "The system is slower than usual and the delay is affecting our customers.",
        "The editor lags when I type, the text appears seconds after I press the key.",
        "Taking too long to generate the page after I click next.",
        "Application performance has degraded a lot over the past two weeks.",
        "The chat window is very slow to send and receive messages.",
        "The page keeps loading and loading, it is extremely slow on our office network but fine at home.",
        "Dashboard charts take ages to render and the browser almost freezes.",
        "Bulk operations are very slow, updating a hundred records takes over twenty minutes.",
        "Memory usage grows continuously and the app slows down the longer it stays open.",
        "Slow response while exporting data, the progress bar barely moves.",
        "The server seems overloaded because every request is delayed.",
        "Opening any attachment inside the portal takes a very long time.",
        "The application is noticeably slower for our team in the evening hours.",
        "Typing in the search box lags and the suggestions appear very late.",
        "Why is the app so slow today? Even small actions take ages.",
    ],
}

# ---------------------------------------------------------------------------
# 2. Extra metadata (only used for EDA, NOT for prediction)
# ---------------------------------------------------------------------------
PRIORITY_WEIGHTS = {  # (High, Medium, Low) chances for each category
    "Login Issue": [0.40, 0.45, 0.15],
    "Application Error": [0.45, 0.40, 0.15],
    "Report": [0.15, 0.45, 0.40],
    "Account Update": [0.10, 0.40, 0.50],
    "Performance": [0.30, 0.50, 0.20],
}
PRIORITIES = ["High", "Medium", "Low"]
STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
STATUS_WEIGHTS = [0.30, 0.25, 0.28, 0.17]


def build_dataset():
    rows = []
    for category, descriptions in TICKETS.items():
        for text in descriptions:
            priority = np.random.choice(PRIORITIES, p=PRIORITY_WEIGHTS[category])
            status = np.random.choice(STATUSES, p=STATUS_WEIGHTS)
            rows.append(
                {
                    "ticket_description": text,
                    "category": category,
                    "priority": priority,
                    "status": status,
                }
            )

    df = pd.DataFrame(rows)

    # Add a few exact duplicate tickets (real systems often have these)
    duplicate_rows = df.sample(n=8, random_state=RANDOM_SEED)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Shuffle and create ticket ids
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df.insert(0, "ticket_id", [f"TKT-{1001 + i}" for i in range(len(df))])

    # Add a few missing values on purpose (priority / status only)
    missing_priority_idx = df.sample(n=3, random_state=1).index
    missing_status_idx = df.sample(n=2, random_state=2).index
    df.loc[missing_priority_idx, "priority"] = np.nan
    df.loc[missing_status_idx, "status"] = np.nan
    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "tickets.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset()
    dataset.to_csv(output_path, index=False)

    print(f"Saved {len(dataset)} tickets to {output_path}")
    print("\nCategory counts:")
    print(dataset["category"].value_counts())
    print("\nMissing values:")
    print(dataset.isnull().sum())
    print("\nDuplicate descriptions:", dataset.duplicated(subset="ticket_description").sum())
