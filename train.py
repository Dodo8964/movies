import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import mlflow
import mlflow.sklearn

def load_and_preprocess():
    ds = fetch_ucirepo(id=235)
    df = ds.data.features.copy()
    # combine date & time
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    # drop raw Date, Time
    df = df.drop(columns=['Date', 'Time'])
    # convert numeric columns
    for c in ['Global_active_power', 'Global_reactive_power', 'Voltage',
              'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # set index
    df = df.set_index('datetime')
    # handle missing: simple forward fill then backfill
    df = df.ffill().bfill()
    # downsample to hourly: take mean
    df_hour = df.resample('H').mean()
    # create lag features and rolling stats
    for lag in [1, 2, 24]:
        df_hour[f'lag_{lag}'] = df_hour['Global_active_power'].shift(lag)
    df_hour['rolling_mean_3'] = df_hour['Global_active_power'].rolling(3).mean()
    df_hour['rolling_mean_24'] = df_hour['Global_active_power'].rolling(24).mean()
    df_hour = df_hour.dropna()

    # target is next hour
    df_hour['target'] = df_hour['Global_active_power'].shift(-1)
    df_hour = df_hour.dropna()

    features = df_hour.drop(columns=['target'])
    target = df_hour['target']
    return features, target

def main():
    X, y = load_and_preprocess()
    # train/test split by time (here simple)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # model pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=42))
    ])

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        pipe.fit(X_train, y_train)
        preds_val = pipe.predict(X_val)
        preds_test = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds_test)
        rmse = mean_squared_error(y_test, preds_test, squared=False)
        print("Test MAE:", mae, "RMSE:", rmse)

        # save pipeline
        joblib.dump(pipe, "model.joblib")
        mlflow.log_artifact("model.joblib")

if __name__ == "__main__":
    main()
