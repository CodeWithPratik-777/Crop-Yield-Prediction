# Minimal baseline training script (XGBoost)
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

def load_data(path):
    df = pd.read_csv(path)
    # Example: drop id/date and target separation
    y = df['yield'].values
    X = df.drop(columns=['field_id', 'season', 'yield'], errors='ignore').values
    return X, y

def train(path, model_out='model.joblib'):
    X, y = load_data(path)
    # simple time train-test split (replace with proper time-based split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8
    )
    model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_test, y_test)], verbose=20)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")
    joblib.dump(model, model_out)
    print(f"Saved model to {model_out}")

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/feature_table.csv'
    train(path)