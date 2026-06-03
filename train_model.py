import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

df = pd.read_csv("data/diabetes_prediction_dataset.csv")

# Encode categoricals
le_gender = LabelEncoder()
le_smoking = LabelEncoder()
df["gender"] = le_gender.fit_transform(df["gender"])
df["smoking_history"] = le_smoking.fit_transform(df["smoking_history"])

features = ["age", "gender", "bmi", "hypertension", "heart_disease",
            "smoking_history", "HbA1c_level", "blood_glucose_level"]
X = df[features]
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump({
    "model": model,
    "le_gender": le_gender,
    "le_smoking": le_smoking
}, "models/diabetes_model.pkl")
print("Model saved to models/diabetes_model.pkl")
