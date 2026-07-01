import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_pipeline import load_raw_data, build_preprocessor, numeric_features, categorical_features, target_col


def get_train_test_data(test_size: float = 0.2, random_state: int = 42):
    df = load_raw_data()
    X = df[numeric_features + categorical_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_models(preprocessor):
    # Logistic Regression pipeline
    lr = LogisticRegression(max_iter=1000, solver="liblinear")

    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", lr)
    ])

    # Random Forest pipeline
    rf = RandomForestClassifier(random_state=42)

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", rf)
    ])

    return lr_pipeline, rf_pipeline


def hyperparameter_tuning(model_pipeline, param_grid, X_train, y_train, cv_splits=5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def main():
    X_train, X_test, y_train, y_test = get_train_test_data()
    preprocessor = build_preprocessor()

    lr_pipeline, rf_pipeline = build_models(preprocessor)

    # Hyperparameter grids
    lr_param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l1", "l2"]
    }

    rf_param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5]
    }

    # Tune Logistic Regression
    lr_grid = hyperparameter_tuning(lr_pipeline, lr_param_grid, X_train, y_train)
    lr_best = lr_grid.best_estimator_
    lr_metrics = evaluate_model(lr_best, X_test, y_test)

    print("Logistic Regression best params:", lr_grid.best_params_)
    print("Logistic Regression metrics:", lr_metrics)

    # Tune Random Forest
    rf_grid = hyperparameter_tuning(rf_pipeline, rf_param_grid, X_train, y_train)
    rf_best = rf_grid.best_estimator_
    rf_metrics = evaluate_model(rf_best, X_test, y_test)

    print("Random Forest best params:", rf_grid.best_params_)
    print("Random Forest metrics:", rf_metrics)


if __name__ == "__main__":
    main()
