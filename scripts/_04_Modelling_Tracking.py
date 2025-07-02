import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

class CreditRiskTrainer:
    """
    Trains and evaluates a credit risk classification model, with MLflow tracking support.
    
    Attributes:
        data_path (str): Path to CSV file containing engineered features + 'is_high_risk' column.
                    - This should be the WOE transformed dataset with the target variable. 
                    - is_high_risk is represented by vd column in the dataset.
        experiment_name (str): MLflow experiment name.
        model (object): Best trained model after tuning.
        grid_search (object): Fitted GridSearchCV object.
        mdl_dir (str): Directory to save trained models.
    """

    def __init__(self, data_path, experiment_name = "credit-risk-boosted", mdl_dir = "models"):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.model = None
        self.grid_search = None
        self.mdl_dir = mdl_dir

    def load_data(self):
        """
        Loads dataset and performs stratified train-test split.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        rel_path = os.path.relpath(self.data_path)
        print(f"Loading data from: {rel_path}")
        df = pd.read_csv(self.data_path)
        drop_cols = ["vd", "FraudResult"]
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df["vd"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        print("\nData split into training and testing sets.")
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates a trained model and returns key metrics.

        Returns:
            dict: Dictionary of performance metrics
        """
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs)
        }

        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k:<10}: {v:.4f}")
        return metrics
    
    def compare_models(self):
        """
        Trains and compares multiple classification models on the same data.
        Logs each to MLflow and prints ROC-AUC scores for selection.

        Models Compared:
            - Logistic Regression
            - Decision Tree
            - Random Forest
            - Gradient Boosting Machine
        """
        print("\nStarting model comparison...")
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Impute missing values
        X_train = X_train.fillna(X_train.median(numeric_only=True))
        X_test = X_test.fillna(X_train.median(numeric_only=True))
        
        models = {
            "LogisticRegression": LogisticRegression(solver="liblinear", max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(max_depth=5),
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "GBM": GradientBoostingClassifier(n_estimators=100, max_depth=3)
        }

        mlflow.set_experiment(self.experiment_name)
        model_scores = {}

        print("\nComparing models...\n")
        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:, 1]

                metrics = {
                    "accuracy": accuracy_score(y_test, preds),
                    "precision": precision_score(y_test, preds, zero_division=0),
                    "recall": recall_score(y_test, preds, zero_division=0),
                    "f1": f1_score(y_test, preds),
                    "roc_auc": roc_auc_score(y_test, probs)
                }

                mlflow.log_params({"model": name})
                mlflow.log_metrics(metrics)

                # Log input example + inferred signature
                X_train = X_train.astype({col: "float64" for col in X_train.select_dtypes("int").columns})
                signature = mlflow.models.infer_signature(X_train, model.predict_proba(X_train))
                mlflow.sklearn.log_model(
                    model,
                    name=f"{name}_model",
                    input_example=X_train.head(1),
                    signature=signature
)
                model_scores[name] = metrics["roc_auc"]
                print(f"{name} ROC-AUC: {metrics['roc_auc']:.4f}")

        best_model = max(model_scores, key=model_scores.get)
        self.best_model_name = best_model  # store name for future tuning
        print(f"\nBest Model: {best_model} with ROC-AUC = {model_scores[best_model]:.4f}")


    def train_with_tracking(self):
        """
        Tunes the best model architecture selected via compare_models().
        Falls back to GradientBoosting if none selected.
        Logs metrics, parameters, and trained model to MLflow.
        """
        print("\nStarting model training with MLflow tracking...")
        X_train, X_test, y_train, y_test = self.load_data()

        # Impute missing values
        X_train = X_train.fillna(X_train.median(numeric_only=True))
        X_test = X_test.fillna(X_train.median(numeric_only=True))

        # Pick model type based on previous comparison
        model_name = getattr(self, "best_model_name", "GBM")
        print(f"Tuning model: {model_name}")

        base_models = {
            "LogisticRegression": LogisticRegression(solver="liblinear", max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
            "GBM": GradientBoostingClassifier()
        }

        param_grids = {
            "LogisticRegression": {
                "C": [0.1, 1],
                "penalty": ["l1", "l2"]
            },
            "DecisionTree": {
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5]
            },
            "RandomForest": {
                "n_estimators": [100, 150],
                "max_depth": [4, 6]
            },
            "GBM": {
                "n_estimators": [100],
                "max_depth": [3, 4],
                "learning_rate": [0.01, 0.1]
            }
        }

        clf = base_models[model_name]
        param_grid = param_grids[model_name]

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=f"Tuned-{model_name}"):
            self.grid_search = GridSearchCV(clf, param_grid, scoring="roc_auc", cv=3)
            self.grid_search.fit(X_train, y_train)
            self.model = self.grid_search.best_estimator_

            print(f"\nBest Parameters Found: {self.grid_search.best_params_}")
            metrics = self.evaluate_model(self.model, X_test, y_test)

            # Save feature list for API alignment
            feature_list_path = os.path.join(self.mdl_dir, "features_list.pkl")
            feature_list_relative = os.path.relpath(feature_list_path, os.getcwd())
            joblib.dump(X_train.columns.tolist(), feature_list_path)
            print(f"\nFeature list saved to: {feature_list_relative}")

            mlflow.log_params(self.grid_search.best_params_)
            mlflow.log_metrics(metrics)
            
            # Log input example + inferred signature
            X_train = X_train.astype({col: "float64" for col in X_train.select_dtypes("int").columns})
            signature = mlflow.models.infer_signature(X_train, self.model.predict_proba(X_train))
            mlflow.sklearn.log_model(
                self.model,
                name = "model",
                input_example = X_train.head(1),
                signature = signature)
            
            print(f"\nTraining completed for {model_name} and logged to MLflow.")

    def get_best_model(self):
        """
        Returns the best estimator after GridSearchCV tuning.

        Returns:
            object: Trained model
        """
        return self.model

    def plot_curves(self):
        """
        Plots ROC and Precision-Recall curves for the given model.
        """
        if self.model is None:
            print("No model trained.")
            return
        _, X_test, _, y_test = self.load_data()

        probs = self.model.predict_proba(X_test)[:, 1]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label="ROC Curve", color="navy")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        model_name = self.model.__class__.__name__
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.show()

        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_test, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, label="Precision-Recall Curve", color="darkgreen")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.legend()
        plt.show()

    def save_model(self, filename=None):
        """
        Saves the trained model to a file with model-type-aware naming.

        Args:
            filename (str): Optional custom file name. If not provided, auto-generates based on model type.
        """
        if self.model:
            # Auto-generate file name if not provided
            model_type = self.model.__class__.__name__
            filename = filename or f"{model_type.lower()}_credit_model_boosted.pkl"

            if not os.path.exists(self.mdl_dir):
                os.makedirs(self.mdl_dir)

            filepath = os.path.join(self.mdl_dir, filename)
            joblib.dump(self.model, filepath)

            relative_path = os.path.relpath(filepath, os.getcwd())
            print(f"\nModel saved to: {relative_path}")
        else:
            print("\nNo model trained yet.")

    def load_model(self, filename=None):
        """
        Loads a trained model from the model directory.
        If filename not provided, tries to detect model type from known names.

        Args:
            filename (str): Optional filename to load. If None, attempts auto-load.
        """
        if not self.mdl_dir:
            print(f"\nModel directory {self.mdl_dir} is not set.")
            return

        # Try to auto-locate a saved model
        if filename is None:
            known_models = ["gradientboostingclassifier", "randomforestclassifier", "logisticregression", "decisiontreeclassifier"]
            for model_id in known_models:
                candidate = os.path.join(self.mdl_dir, f"{model_id}_credit_model_boosted.pkl")
                if os.path.exists(candidate):
                    filename = f"{model_id}_credit_model_boosted.pkl"
                    break

        if filename is None:
            print("\nNo model filename provided and no known model found in directory.")
            return

        filepath = os.path.join(self.mdl_dir, filename)
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            relative_path = os.path.relpath(filepath, os.getcwd())
            print(f"\nModel loaded from: {relative_path}")
        else:
            print(f"\nModel file not found at: {filepath}")
