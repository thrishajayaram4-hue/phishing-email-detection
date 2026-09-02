# phishing-email-detection
# Phishing Email Detection Model

## 📌 Project Description

This project is a Machine Learning based Phishing Email Detection Model developed using Python and Scikit-learn.

The model analyzes email text and classifies emails into two categories:

- 🔴 Phishing
- 🟢 Safe

## 🎯 Objectives

- Detect phishing emails using machine learning.
- Analyze email text and keywords.
- Extract URL-related features.
- Classify emails as Phishing or Safe.
- Evaluate the model using accuracy and a confusion matrix.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Regular Expressions

## ⚙️ Machine Learning Process

1. Collect phishing and legitimate email samples.
2. Preprocess the email data.
3. Extract URL-related features.
4. Convert text into numerical features using TF-IDF.
5. Train a Logistic Regression model.
6. Test the model with unseen emails.
7. Calculate accuracy.
8. Display the confusion matrix and classification report.

## 📊 Output

The program displays:

- Model accuracy
- Confusion matrix
- Classification report
- Prediction for new emails

## 🚀 How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
