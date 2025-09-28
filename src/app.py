# src/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from datetime import datetime

MODEL_PATH = "models/model.joblib"
INFERENCE_LOG = "models/inference_log.csv"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

# features must match training features order
# We'll reconstruct the input DataFrame similarly to training:
FEATURES = model.named_steps['preprocessor'].transformers_[0][2] + \
           model.named_steps['preprocessor'].transformers_[1][2] + \
           list(model.named_steps['preprocessor'].transformers_[-1][2] if hasattr(model.named_steps['preprocessor'], 'transformers_') else [])

# Simpler: accept JSON with exact keys used in preprocess_train.features
@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json
    df = pd.DataFrame([payload])
    
    # get expected genre columns from the training pipeline
    # Suppose you saved the list of genre features during training, e.g. in model metadata
    expected_genres = ["genre_Documentary", "genre_Comedy Romance", "genre_Comedy Drama",
                       "genre_Drama Romance", "genre_Comedy", "genre_Comedy Drama Romance",
                       "genre_Horror Thriller", "genre_Action", "genre_Adventure", "genre_Drama"]
    for col in expected_genres:
        if col not in df.columns:
            df[col] = 0

    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0,1])
    # log inference
    os.makedirs(os.path.dirname(INFERENCE_LOG), exist_ok=True)
    log_row = {**payload, "prediction": pred, "prob": prob, "ts": datetime.utcnow().isoformat()}
    pd.DataFrame([log_row]).to_csv(INFERENCE_LOG, mode="a", header=not os.path.exists(INFERENCE_LOG), index=False)
    return jsonify({"prediction": pred, "probability": prob})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
