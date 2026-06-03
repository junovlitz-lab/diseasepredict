import logging
import os

import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, abort

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

bundle = joblib.load("models/diabetes_model.pkl")
model = bundle["model"]
le_gender = bundle["le_gender"]
le_smoking = bundle["le_smoking"]

# try to load additional models if present
heart_bundle = None
stroke_bundle = None
if os.path.exists("models/heart_model.pkl"):
    heart_bundle = joblib.load("models/heart_model.pkl")
if os.path.exists("models/stroke_model.pkl"):
    stroke_bundle = joblib.load("models/stroke_model.pkl")

def safe_transform(le, value, fallback=0):
    try:
        return le.transform([value])[0]
    except ValueError:
        return fallback


def safe_encode(encoders, feature, value):
    """Encode a value with a LabelEncoder if available, otherwise return a numeric cast or fallback 0."""
    if feature in encoders:
        le = encoders[feature]
        try:
            return int(le.transform([value])[0])
        except Exception:
            return 0
    # numeric fallback
    try:
        return float(value)
    except Exception:
        return 0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        app.logger.warning("Predict request missing JSON payload")
        return jsonify({"error": "Missing or invalid JSON payload."}), 400

    disease = data.get("disease", "diabetes")
    app.logger.info("Predict request: %s", {"disease": disease, "keys": list(data.keys())})
    try:
        # Currently only a trained diabetes model is bundled.
        if disease == "diabetes":
            gender = safe_transform(le_gender, data["gender"])
            smoking = safe_transform(le_smoking, data["smoking_history"])
            features = np.array([[
                float(data.get("age", 50)),
                gender,
                float(data.get("bmi", 25)),
                int(data.get("hypertension", 0)),
                int(data.get("heart_disease", 0)),
                smoking,
                float(data.get("HbA1c_level", 5.5)),
                float(data.get("blood_glucose_level", 100))
            ]])
            prob = model.predict_proba(features)[0][1]
            pct = round(prob * 100, 1)
            if pct < 30:
                label = "Low"
            elif pct < 60:
                label = "Medium"
            else:
                label = "High"
            return jsonify({"risk_pct": pct, "label": label})
        elif disease in ("heart", "stroke"):
            # select bundle
            bundle_map = {"heart": heart_bundle, "stroke": stroke_bundle}
            b = bundle_map.get(disease)
            if not b:
                return jsonify({"error": f"No model available for '{disease}'."}), 400

            features = b.get("features", [])
            encoders = b.get("encoders", {})
            scaler = b.get("scaler", None)
            model_local = b.get("model")

            row = []
            for feat in features:
                # prefer direct key, otherwise try common fallbacks
                val = data.get(feat, data.get(feat.lower(), data.get(feat.replace('-', '_'), 0)))
                enc_val = safe_encode(encoders, feat, val)
                row.append(enc_val)

            X = np.array([row])
            if scaler is not None:
                try:
                    X = scaler.transform(X)
                except Exception:
                    pass

            try:
                prob = model_local.predict_proba(X)[0][1]
            except Exception:
                prob = float(model_local.predict(X)[0])

            pct = round(prob * 100, 1)
            if pct < 30:
                label = "Low"
            elif pct < 60:
                label = "Medium"
            else:
                label = "High"
            return jsonify({"risk_pct": pct, "label": label})
        else:
            return jsonify({"error": f"Unknown disease '{disease}'"}), 400
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 400


@app.route('/data/<path:filename>')
def data_file(filename):
    # Serve files from the data/ directory for download/viewing.
    try:
        return send_from_directory('data', filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
