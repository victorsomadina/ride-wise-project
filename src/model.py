from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost

MODELS_DIR = Path('../models')
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_parquet('../data/model_input.parquet')

X = df.drop(columns='churn')
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ride-wise-project")

with mlflow.start_run(run_name="logistic-regression"):
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("features", X.columns.tolist())
    mlflow.log_metric('roc_auc', auc)
    mlflow.sklearn.log_model(model)
    joblib.dump(model, MODELS_DIR / 'logistic_regression.pkl')

    print(f"Logistic Regression AUC: {auc}")

with mlflow.start_run(run_name="random-forest"):
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3],
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_distributions=param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("features", X.columns.tolist())
    mlflow.log_params(search.best_params_)
    mlflow.log_metric('roc_auc', auc)
    mlflow.sklearn.log_model(model)
    joblib.dump(model, MODELS_DIR / 'random_forest_search.pkl')

    print(f"Random Forest AUC: {auc}")

with mlflow.start_run(run_name="random-forest-fixed"):
    rf_params = {
        "n_estimators": 500,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_depth": None,
    }
    model = RandomForestClassifier(random_state=42, class_weight="balanced", **rf_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("features", X.columns.tolist())
    mlflow.log_params(rf_params)
    mlflow.log_metric('roc_auc', auc)
    mlflow.sklearn.log_model(model)

    feature_importance = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .rename('importance')
        .rename_axis('feature')
        .reset_index()
    )
    feature_importance.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')
    joblib.dump(model, MODELS_DIR / 'random_forest_fixed.pkl')

    print(f"Random Forest (fixed params) AUC: {auc}")
    print(feature_importance)

with mlflow.start_run(run_name="xgboost"):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    mlflow.log_param("model", "XGBClassifier")
    mlflow.log_param("features", X.columns.tolist())
    mlflow.log_metric('roc_auc', auc)
    mlflow.xgboost.log_model(model)

    feature_importance = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .rename('importance')
        .rename_axis('feature')
        .reset_index()
    )
    feature_importance.to_csv('feature_importance_xgboost.csv', index=False)
    mlflow.log_artifact('feature_importance_xgboost.csv')
    joblib.dump(model, MODELS_DIR / 'xgboost.pkl')

    print(f"XGBoost AUC: {auc}")
    print(feature_importance)
