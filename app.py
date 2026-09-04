import streamlit as st
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
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
def extract_url_features(text):
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    return len(urls)

df["url_count"] = df["email"].apply(extract_url_features)

df["combined_text"] = df.apply(
    lambda row: row["email"] + " URLCOUNT_" + str(row["url_count"]),
    axis=1
)
X_train, X_test, y_train, y_test = train_test_split(
    df["combined_text"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=["Safe", "Phishing"])
st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="📧"
)

st.title("📧 Phishing Email Detection Model")
st.write("Enter an email below to check whether it is Phishing or Safe.")

st.divider()

email = st.text_area(
    "Enter Email Text:",
    placeholder="Example: Urgent! Verify your bank account immediately..."
)

if st.button("🔍 Check Email"):

    if email.strip() == "":
        st.warning("Please enter an email message.")

    else:
        url_count = extract_url_features(email)

        combined_text = email + " URLCOUNT_" + str(url_count)

        features = vectorizer.transform([combined_text])

        prediction = model.predict(features)[0]

        if prediction == "Phishing":
            st.error("🚨 PHISHING EMAIL DETECTED")
        else:
            st.success("✅ SAFE EMAIL")

        st.write("### Analysis")
        st.write(f"URLs detected: {url_count}")
        st.write(f"Prediction: {prediction}")