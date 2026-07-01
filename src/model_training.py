import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve)
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
    
def log_metrics_to_mlflow(metrics: dict):
    for name, value in metrics.items():
        mlflow.log_metric(name, value)


def log_confusion_matrix(y_test, y_pred, prefix: str):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{prefix} Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

    fname = f"{prefix.lower()}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

    mlflow.log_artifact(fname)
    os.remove(fname)


def log_roc_curve(y_test, y_proba, prefix: str):
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"{prefix} ROC curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend(loc="lower right")

    fname = f"{prefix.lower()}_roc_curve.png"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

    mlflow.log_artifact(fname)
    os.remove(fname)


def main():
    # Set or create experiment
    mlflow.set_experiment("heart-disease-uci-experiments")

    X_train, X_test, y_train, y_test = get_train_test_data()
    preprocessor = build_preprocessor()

    lr_pipeline, rf_pipeline = build_models(preprocessor)

    lr_param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l1", "l2"]
    }

    rf_param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5]
    }

    # Logistic Regression run
    with mlflow.start_run(run_name="LogisticRegression"):
        lr_grid = hyperparameter_tuning(lr_pipeline, lr_param_grid, X_train, y_train)
        lr_best = lr_grid.best_estimator_
        lr_metrics = evaluate_model(lr_best, X_test, y_test)

        print("Logistic Regression best params:", lr_grid.best_params_)
        print("Logistic Regression metrics:", lr_metrics)

        # Log params and metrics
        mlflow.log_params(lr_grid.best_params_)
        log_metrics_to_mlflow(lr_metrics)
        mlflow.log_metric("cv_roc_auc", lr_grid.best_score_)

        # Log plots
        y_pred_lr = lr_best.predict(X_test)
        y_proba_lr = lr_best.predict_proba(X_test)[:, 1]
        log_confusion_matrix(y_test, y_pred_lr, prefix="LR")
        log_roc_curve(y_test, y_proba_lr, prefix="LR")

        # Log model
        mlflow.sklearn.log_model(
        	lr_best,
        	name="model_lr",
        	serialization_format="pickle"
        )

    # Random Forest run
    with mlflow.start_run(run_name="RandomForest"):
        rf_grid = hyperparameter_tuning(rf_pipeline, rf_param_grid, X_train, y_train)
        rf_best = rf_grid.best_estimator_
        rf_metrics = evaluate_model(rf_best, X_test, y_test)

        print("Random Forest best params:", rf_grid.best_params_)
        print("Random Forest metrics:", rf_metrics)

        mlflow.log_params(rf_grid.best_params_)
        log_metrics_to_mlflow(rf_metrics)
        mlflow.log_metric("cv_roc_auc", rf_grid.best_score_)

        y_pred_rf = rf_best.predict(X_test)
        y_proba_rf = rf_best.predict_proba(X_test)[:, 1]
        log_confusion_matrix(y_test, y_pred_rf, prefix="RF")
        log_roc_curve(y_test, y_proba_rf, prefix="RF")

        mlflow.sklearn.log_model(
        	rf_best,
        	name="model_rf",
        	serialization_format="pickle"
        )
        
    # Save final RF pipeline (preprocessor + model) for API usage
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_best, "models/final_model_rf.pkl")
    print("Saved final Random Forest pipeline to models/final_model_rf.pkl")

if __name__ == "__main__":
    main()
