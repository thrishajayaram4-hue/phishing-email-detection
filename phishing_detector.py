import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample phishing and legitimate email dataset
data = {
    "email": [
        "Congratulations! You won a prize. Click here to claim your reward.",
        "Urgent! Your bank account will be suspended. Verify your account now.",
        "You have received a free gift. Click the link to claim it.",
        "Your password has expired. Login immediately to verify your account.",
        "Important security alert. Confirm your bank details using this link.",
        "Limited time offer! You have won $1000. Claim now.",
        "Please update your payment information immediately.",
        "Click here to receive your free coupon.",
        "Your account has been selected for a special reward.",
        "Verify your identity now to avoid account suspension.",
        "Hi, are we still meeting for the project tomorrow?",
        "Please find attached the notes from today's meeting.",
        "Your order has been shipped and will arrive tomorrow.",
        "Thank you for submitting your assignment.",
        "The meeting has been scheduled for Monday at 10 AM.",
        "Please review the project report and send your feedback.",
        "Your electricity bill is ready. You can view it in your account.",
        "Reminder: Your appointment is scheduled for tomorrow.",
        "Here is the document we discussed yesterday.",
        "Thank you for your email. We will respond shortly."
    ],
    "label": [
        "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
        "Safe", "Safe", "Safe", "Safe", "Safe",
        "Safe", "Safe", "Safe", "Safe", "Safe"
    ]
}

df = pd.DataFrame(data)

# Extract URL-related features
def extract_url_features(text):
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    return len(urls)

df["url_count"] = df["email"].apply(extract_url_features)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["email"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

# Convert email text into numerical features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train machine learning model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("PHISHING EMAIL DETECTION MODEL")
print("=" * 50)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test new emails
test_emails = [
    "Urgent! Click this link to verify your bank account immediately.",
    "Hi, please send me the project report when you are free."
]

test_features = vectorizer.transform(test_emails)
predictions = model.predict(test_features)

print("\nEmail Predictions:")
for email, prediction in zip(test_emails, predictions):
    print(f"\nEmail: {email}")
    print(f"Prediction: {prediction}")
