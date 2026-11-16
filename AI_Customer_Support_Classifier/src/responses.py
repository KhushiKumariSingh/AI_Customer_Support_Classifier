"""
responses.py
------------
Optional feature: suggest an automatic reply based on the predicted category.

Version 1 (this file): simple rule-based replies, one template per category.

How to upgrade to an LLM later (idea only, not implemented here):
    1. Keep the classifier to find the category.
    2. Send the ticket text + category to an LLM API with a prompt such as:
       "You are a polite support agent. The ticket category is {category}.
        Write a short, helpful reply to: {ticket_text}"
    3. Show the LLM reply to a human agent for review before sending.
"""

AUTO_REPLIES = {
    "Login Issue": (
        "Hello, sorry you are having trouble signing in. Please try resetting "
        "your password using the 'Forgot password' link and check your spam "
        "folder for the email. If your account is locked, our team will help "
        "unlock it. We have raised your ticket."
    ),
    "Application Error": (
        "Hello, sorry about the error. Our technical team has been notified. "
        "Please share a screenshot or the exact error message and the steps "
        "that led to it, so we can fix it faster."
    ),
    "Report": (
        "Hello, thanks for your report request. Our team will review the "
        "details, such as the date range and format you need, and share the "
        "report or fix any data issue as soon as possible."
    ),
    "Account Update": (
        "Hello, we received your account update request. For security, our "
        "team may ask you to verify your identity before we apply the change. "
        "You will be notified once it is done."
    ),
    "Performance": (
        "Hello, sorry the system is running slowly. We are checking the "
        "performance on our side. Meanwhile, please try refreshing the page "
        "or clearing your browser cache, and tell us the time the slowness began."
    ),
}

DEFAULT_REPLY = (
    "Hello, thanks for contacting support. Your ticket has been received and "
    "a support agent will get back to you shortly."
)


def get_auto_reply(category):
    """Return the reply template for a category (or a safe default)."""
    return AUTO_REPLIES.get(category, DEFAULT_REPLY)
