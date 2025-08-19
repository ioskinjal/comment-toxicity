# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("data/train.csv")

# Check dataset
print(df.head())

# 2. Features & Labels
X = df["comment_text"]
y = df["toxic"]   # Must exist in your CSV

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 7. Save model & vectorizer
joblib.dump(model, "models/toxicity_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model trained & saved in models/")