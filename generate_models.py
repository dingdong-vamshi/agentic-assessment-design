import os
import json
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create models directory
os.makedirs("models", exist_ok=True)

# 1. Create a dummy dataset
data = [
    # Easy
    ("how to print hello world in python", "Easy"),
    ("what is a variable", "Easy"),
    ("how to add two numbers", "Easy"),
    ("what does html stand for", "Easy"),
    ("how to make a list in javascript", "Easy"),
    
    # Medium
    ("how to reverse a linked list", "Medium"),
    ("explain binary search algorithm", "Medium"),
    ("what is the difference between an abstract class and an interface", "Medium"),
    ("how to merge two dictionaries in python 3", "Medium"),
    ("explain MVC architecture", "Medium"),
    
    # Hard
    ("how to write a custom linux kernel module", "Hard"),
    ("implement a lock free queue in c++", "Hard"),
    ("explain the inner workings of the v8 javascript engine", "Hard"),
    ("prove p vs np", "Hard"),
    ("how to optimize a neural network from scratch using cuda", "Hard"),
]

X_texts = [item[0] for item in data]
y_labels = [item[1] for item in data]

# 2. Train TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X_texts)

# 3. Train LogisticRegression
model = LogisticRegression(class_weight="balanced", random_state=42)
model.fit(X_tfidf, y_labels)

# 4. Generate metrics
y_pred = model.predict(X_tfidf)
acc = accuracy_score(y_labels, y_pred)
report = classification_report(y_labels, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_labels, y_pred, labels=model.classes_)

# 5. Save everything
joblib.dump(model, "models/logistic_regression_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

with open("models/model_metrics.json", "w") as f:
    json.dump({"accuracy": acc, "report": report}, f, indent=2)

np.save("models/confusion_matrix.npy", cm)

print("✅ Successfully created mocked machine learning models and metrics in the models/ directory!")
