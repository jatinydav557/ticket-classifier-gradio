# ticket_classifier_app.py

import pandas as pd
import re
import nltk
import gradio as gr
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Download NLTK resources (only needs to run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Entity extraction
PRODUCT_LIST = ['SmartWatch V2', 'UltraClean Vacuum', 'SoundWave 300', 'EcoBreeze AC',
                'PhotoSnap Cam', 'Vision LED TV', 'RoboChef Blender', 'FitRun Treadmill',
                'PowerMax Battery']
KEYWORDS = ['broken', 'late', 'error', 'malfunction', 'lost', 'issue', 'not working', 'no response']

def extract_entities(text):
    entities = {
        'products': [p for p in PRODUCT_LIST if p.lower() in text.lower()],
        'complaints': [k for k in KEYWORDS if k in text.lower()]
    }
    return entities

# Load and clean data
df = pd.read_csv("csv_data/ticket.csv")
df.dropna(subset=['ticket_text', 'issue_type', 'urgency_level'], inplace=True)
df['cleaned_text'] = df['ticket_text'].apply(preprocess)

# Label encoding
issue_enc = LabelEncoder()
urgency_enc = LabelEncoder()
df['issue_encoded'] = issue_enc.fit_transform(df['issue_type'])
df['urgency_encoded'] = urgency_enc.fit_transform(df['urgency_level'])

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_text'])

# Train-test split
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X, df['issue_encoded'], test_size=0.2, stratify=df['issue_encoded'])
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X, df['urgency_encoded'], test_size=0.2, stratify=df['urgency_encoded'])

# Train models
issue_model = LogisticRegression(max_iter=1000)
urgency_model = LogisticRegression(max_iter=1000)
issue_model.fit(X_train_i, y_train_i)
urgency_model.fit(X_train_u, y_train_u)

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(issue_model, "models/issue_model.pkl")
joblib.dump(urgency_model, "models/urgency_model.pkl")

# Gradio Prediction Function
def predict(text):
    cleaned = preprocess(text)
    vector = vectorizer.transform([cleaned])
    issue_pred = issue_model.predict(vector)[0]
    urgency_pred = urgency_model.predict(vector)[0]
    issue = issue_enc.inverse_transform([issue_pred])[0]
    urgency = urgency_enc.inverse_transform([urgency_pred])[0]
    entities = extract_entities(text)
    return issue, urgency, entities

# Gradio Interface
app = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, label="Enter Ticket Text"),
    outputs=[
        gr.Text(label="Issue Type"),
        gr.Text(label="Urgency Level"),
        gr.JSON(label="Extracted Entities")
    ],
    title="Simple Ticket Classifier",
    description="Predict issue type and urgency level from customer support tickets. Also extracts product names and complaint keywords."
)

if __name__ == "__main__":
    app.launch()
