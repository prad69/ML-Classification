"""Model explainability using SHAP (SHapley Additive exPlanations)."""
import shap
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"


class LoanExplainer:
    """Explain loan approval decisions using SHAP."""

    def __init__(self, model_path=None):
        """Initialize the explainer with a trained model."""
        if model_path is None:
            model_path = MODEL_DIR / "kaggle_loan_model.pkl"

        # Load model
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_columns = model_data['feature_columns']

        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        print(f"✓ Loaded {self.model_name} model")
        print(f"✓ Features: {len(self.feature_columns)}")

    def explain_prediction(self, X: pd.DataFrame, feature_names: List[str] = None) -> Dict:
        """
        Explain a single prediction.

        Args:
            X: Feature DataFrame (single row)
            feature_names: Optional list of feature names for display

        Returns:
            Dictionary with prediction and explanation
        """
        if feature_names is None:
            feature_names = self.feature_columns

        # Get prediction
        prediction = self.model.predict(X)[0]
        prediction_proba = self.model.predict_proba(X)[0]

        # Get SHAP values
        shap_values = self.explainer.shap_values(X)

        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Get base value (expected value)
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

        # Ensure shap_values is 2D array
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)

        # Create feature contributions
        feature_contributions = []
        shap_vals_row = np.array(shap_values[0]).flatten()  # Flatten to 1D
        for i, (feat_name, feat_val) in enumerate(zip(feature_names, X.iloc[0].values)):
            shap_val = shap_vals_row[i].item() if hasattr(shap_vals_row[i], 'item') else float(shap_vals_row[i])
            feature_contributions.append({
                'feature': feat_name,
                'value': float(feat_val),
                'contribution': shap_val,
                'abs_contribution': abs(shap_val)
            })

        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)

        # Create decision explanation
        decision = "APPROVED" if prediction == 1 else "REJECTED"
        confidence = float(prediction_proba[prediction]) * 100

        # Top positive and negative factors
        positive_factors = [f for f in feature_contributions if f['contribution'] > 0][:5]
        negative_factors = [f for f in feature_contributions if f['contribution'] < 0][:5]

        # Helper function to convert to float safely
        def to_float(val):
            if isinstance(val, np.ndarray):
                return float(val.flatten()[0])
            elif hasattr(val, 'item'):
                try:
                    return val.item()
                except:
                    return float(val)
            return float(val)

        explanation = {
            'decision': decision,
            'confidence': confidence,
            'probability_approved': float(prediction_proba[1]) * 100,
            'probability_rejected': float(prediction_proba[0]) * 100,
            'base_value': to_float(base_value),
            'prediction_value': to_float(base_value) + shap_vals_row.sum(),
            'feature_contributions': feature_contributions,
            'top_positive_factors': positive_factors,
            'top_negative_factors': negative_factors,
            'reasoning': self._generate_reasoning(decision, positive_factors, negative_factors)
        }

        return explanation

    def _generate_reasoning(self, decision: str, positive: List[Dict], negative: List[Dict]) -> str:
        """Generate human-readable reasoning for the decision."""
        if decision == "APPROVED":
            reason = f"Loan application {decision.lower()}. "

            if positive:
                reason += "Key supporting factors: "
                factors = [f"{f['feature']} (impact: +{f['contribution']:.3f})" for f in positive[:3]]
                reason += ", ".join(factors) + ". "

            if negative:
                reason += "Despite some concerns: "
                factors = [f"{f['feature']} (impact: {f['contribution']:.3f})" for f in negative[:2]]
                reason += ", ".join(factors) + "."
        else:
            reason = f"Loan application {decision.lower()}. "

            if negative:
                reason += "Primary concerns: "
                factors = [f"{f['feature']} (impact: {f['contribution']:.3f})" for f in negative[:3]]
                reason += ", ".join(factors) + ". "

            if positive:
                reason += "Some positive aspects: "
                factors = [f"{f['feature']} (impact: +{f['contribution']:.3f})" for f in positive[:2]]
                reason += ", ".join(factors) + "."

        return reason

    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            })
            df = df.sort_values('importance', ascending=False)
            return df
        else:
            return None

    def save_explainer(self, path=None):
        """Save the explainer for later use."""
        if path is None:
            path = MODEL_DIR / "loan_explainer.pkl"

        joblib.dump(self, path)
        print(f"✓ Explainer saved to: {path}")


def test_explainer():
    """Test the explainer with sample data."""
    from ..data.preprocessing import KaggleLoanPreprocessor
    import sys
    sys.path.append(str(BASE_DIR))

    # Load and preprocess test data
    from pathlib import Path
    test_data = pd.read_csv(BASE_DIR / "data" / "raw" / "kaggle_loan" / "train.csv").head(5)

    preprocessor = KaggleLoanPreprocessor()
    processed = preprocessor.preprocess(test_data, is_train=True)

    # Create explainer
    explainer = LoanExplainer()

    print("\n" + "=" * 60)
    print("Testing Loan Explainer")
    print("=" * 60)

    # Explain first sample
    X_sample = processed.drop('target', axis=1).iloc[0:1]
    explanation = explainer.explain_prediction(X_sample)

    print(f"\nDecision: {explanation['decision']}")
    print(f"Confidence: {explanation['confidence']:.2f}%")
    print(f"Probability Approved: {explanation['probability_approved']:.2f}%")
    print(f"\nReasoning: {explanation['reasoning']}")

    print(f"\nTop 5 Contributing Features:")
    for feat in explanation['feature_contributions'][:5]:
        print(f"  {feat['feature']:<25} = {feat['value']:<10.2f} (impact: {feat['contribution']:+.4f})")

    # Feature importance
    print(f"\n" + "=" * 60)
    print("Global Feature Importance (Top 10)")
    print("=" * 60)
    importance_df = explainer.get_feature_importance()
    if importance_df is not None:
        print(importance_df.head(10).to_string(index=False))

    # Save explainer
    explainer.save_explainer()


if __name__ == "__main__":
    test_explainer()
