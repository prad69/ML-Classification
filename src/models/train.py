"""Train machine learning models for loan approval prediction."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

RANDOM_STATE = 42


def load_processed_data(dataset_name='kaggle'):
    """Load preprocessed data."""
    if dataset_name == 'kaggle':
        df = pd.read_csv(PROCESSED_DATA_DIR / "kaggle_loan_processed.csv")
    else:
        df = pd.read_csv(PROCESSED_DATA_DIR / "german_credit_processed.csv")

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model."""
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}...")
    print(f"{'=' * 60}")

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"CV F1-Score:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

    return {
        'model': model,
        'model_name': model_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std()
    }


def train_all_models(dataset_name='kaggle'):
    """Train and compare multiple models."""
    print(f"\n{'*' * 60}")
    print(f"Training models on {dataset_name.upper()} dataset")
    print(f"{'*' * 60}")

    # Load data
    X, y = load_processed_data(dataset_name)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain size: {X_train.shape}")
    print(f"Test size:  {X_test.shape}")

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss')
    }

    # Train and evaluate all models
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)

    # Compare results
    print(f"\n{'=' * 60}")
    print("Model Comparison")
    print(f"{'=' * 60}")
    print(f"{'Model':<25} {'Test Acc':<12} {'F1-Score':<12} {'CV F1':<12}")
    print(f"{'-' * 60}")

    for r in results:
        print(f"{r['model_name']:<25} {r['test_acc']:<12.4f} {r['f1']:<12.4f} {r['cv_f1_mean']:<12.4f}")

    # Select best model based on F1-score
    best_result = max(results, key=lambda x: x['f1'])
    best_model = best_result['model']
    best_name = best_result['model_name']

    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_name} (F1-Score: {best_result['f1']:.4f})")
    print(f"{'=' * 60}")

    return best_model, best_name, X.columns.tolist(), results


def save_model(model, model_name, feature_columns, dataset_name='kaggle'):
    """Save the best model and metadata."""
    model_path = MODEL_DIR / f"{dataset_name}_loan_model.pkl"

    model_data = {
        'model': model,
        'model_name': model_name,
        'feature_columns': feature_columns,
        'dataset': dataset_name
    }

    joblib.dump(model_data, model_path)
    print(f"\n✓ Model saved to: {model_path}")

    return model_path


if __name__ == "__main__":
    # Train on Kaggle dataset (primary)
    best_model, best_name, features, results = train_all_models('kaggle')
    save_model(best_model, best_name, features, 'kaggle')

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best model: {best_name}")
    print(f"Features: {len(features)}")
    print("\nNext steps:")
    print("  1. Implement SHAP explainability")
    print("  2. Build FastAPI backend")
    print("  3. Create React dashboard")
