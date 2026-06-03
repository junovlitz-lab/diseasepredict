import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os


def train(data_path, target_col, out_path, random_state=42):
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {data_path}")

    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    encoders = {}
    # Encode object/category columns
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = X[col].fillna('NA')
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Fill numeric NaNs with median
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=random_state, stratify=stratify)

    model = RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    report = classification_report(y_test, y_pred, zero_division=0)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'features': list(X.columns),
    }, out_path)

    print(f"Saved model bundle to {out_path}")
    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")
    print(report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    train(args.data, args.target, args.out)
