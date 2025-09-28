# src/preprocess_train.py
import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import json
import os
from datetime import datetime

DATA_PATH = "data/movie_dataset.csv"
MODEL_OUT = "models/model.joblib"

def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    # keep only necessary cols; be defensive if columns missing
    for col in ["budget", "revenue", "runtime", "original_language", "genres", "release_date"]:
        if col not in df.columns:
            df[col] = np.nan

    # convert numeric
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(df["runtime"].median())
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)

    # parse release_date to year/month
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year.fillna(0).astype(int)
    df["release_month"] = df["release_date"].dt.month.fillna(0).astype(int)

    # genres: try to extract top 5 genres — dataset might store as "Action|Adventure" or "['Action', 'Adventure']"
    def parse_genres(x):
        if pd.isna(x): 
            return []
        if isinstance(x, str):
            if "|" in x:
                return [g.strip() for g in x.split("|") if g.strip()]
            try:
                # try json-like
                parsed = json.loads(x.replace("'", '"'))
                if isinstance(parsed, list):
                    return [g.get('name') if isinstance(g, dict) else g for g in parsed]
            except Exception:
                # fallback comma separated
                return [g.strip() for g in x.split(",") if g.strip()]
        return []
    df["genres_list"] = df["genres"].apply(parse_genres)

    return df

def featurize(df):
    # create binary features for top N genres
    all_genres = pd.Series([g for sub in df["genres_list"] for g in sub])
    top_genres = all_genres.value_counts().head(8).index.tolist()
    for g in top_genres:
        df[f"genre_{g}"] = df["genres_list"].apply(lambda lst: int(g in lst))
    # keep features
    features = ["budget", "runtime", "release_year", "release_month", "original_language"] + [f"genre_{g}" for g in top_genres]
    # fill missing language
    df["original_language"] = df["original_language"].fillna("unknown")
    return df, features

def make_label(df):
    median_rev = df["revenue"].median()
    df["hit"] = (df["revenue"] >= median_rev).astype(int)
    return df, median_rev

def train():
    df = load_and_clean()
    df, features = featurize(df)
    df, median_rev = make_label(df)

    X = df[features]
    y = df["hit"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # pipeline: numeric imputer+scaler, categorical encoder
    numeric_feats = ["budget", "runtime", "release_year", "release_month"]
    categorical_feats = ["original_language"]
    pass_through_feats = [c for c in X.columns if c.startswith("genre_")]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_feats),
        ("cat", categorical_transformer, categorical_feats),
    ], remainder="passthrough")  # passthrough genre binary columns

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    mlflow.set_experiment("movie-hit-classifier")
    with mlflow.start_run():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print("Accuracy:", acc, "AUC:", auc)
        print(classification_report(y_test, preds))

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("roc_auc", float(auc))
        # save model locally and log artifact
        os.makedirs("../models", exist_ok=True)
        joblib.dump(clf, MODEL_OUT)
        mlflow.log_artifact(MODEL_OUT, artifact_path="model")
        # save metadata
        mlflow.log_param("median_revenue", float(median_rev))
        mlflow.sklearn.log_model(clf, "sklearn-model")

    print("Model saved:", MODEL_OUT)

if __name__ == "__main__":
    train()
