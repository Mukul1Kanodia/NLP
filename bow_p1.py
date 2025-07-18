# -*- coding: utf-8 -*-
"""BoW_p1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12BtA_l0KAl7yPCnJupOWjwODcChbge7L
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
# Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Make sure the 'spam.csv' file is in the same directory as your script/notebook.
df = pd.read_csv('/content/spam.csv', encoding='latin-1')

# --- Data Cleaning ---
# Keep only the necessary columns and rename them for clarity
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Remove any null rows if they exist
df.dropna(inplace=True)

print("Dataset Head:")
print(df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# Initialize the CountVectorizer
# This will convert text into a matrix of token counts
bow_vectorizer = CountVectorizer(stop_words='english', lowercase=True)

# Fit the vectorizer on the training data and transform it
# .fit() learns the vocabulary from the training data
# .transform() converts the text into a sparse matrix of word counts
X_train_bow = bow_vectorizer.fit_transform(X_train)

# ONLY transform the test data using the already-fitted vectorizer
# We do this to ensure the test data is vectorized using the same vocabulary as the training data
X_test_bow = bow_vectorizer.transform(X_test)

print("\nShape of the BoW training matrix:", X_train_bow.shape)
# This will show (number of training messages, size of vocabulary)`

# Initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_bow, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_bow)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from google.colab import drive
drive.mount('/content/drive')