"""Data preprocessing for loan datasets."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


class KaggleLoanPreprocessor:
    """Preprocessor for Kaggle Loan Prediction dataset."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None

    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Preprocess the dataset."""
        df = df.copy()

        # Remove Loan_ID
        if 'Loan_ID' in df.columns:
            df = df.drop('Loan_ID', axis=1)

        # Handle missing values
        # Credit_History: fill with mode (most common)
        if 'Credit_History' in df.columns:
            df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

        # Categorical: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Loan_Status':
                df[col].fillna(df[col].mode()[0], inplace=True)

        # Numerical: fill with median
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if col != 'Loan_Status':
                df[col].fillna(df[col].median(), inplace=True)

        # Feature engineering
        # Total income
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

        # Loan amount to income ratio
        df['LoanAmountToIncome'] = df['LoanAmount'] / (df['TotalIncome'] + 1)

        # Income per dependent
        df['Dependents_num'] = df['Dependents'].replace('3+', '3').astype(int)
        df['IncomePerDependent'] = df['TotalIncome'] / (df['Dependents_num'] + 1)

        # EMI (Estimated Monthly Installment)
        df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1)

        # EMI to income ratio
        df['EMIToIncome'] = df['EMI'] / (df['TotalIncome'] + 1)

        # Encode target
        if 'Loan_Status' in df.columns:
            df['Loan_Status'] = (df['Loan_Status'] == 'Y').astype(int)
            target = df['Loan_Status']
            df = df.drop('Loan_Status', axis=1)
        else:
            target = None

        # Encode categorical variables
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education',
                           'Self_Employed', 'Property_Area']

        for col in categorical_cols:
            if col in df.columns:
                if is_train:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )

        # Store feature columns
        if is_train:
            self.feature_columns = df.columns.tolist()

        # Add target back if it exists
        if target is not None:
            df['target'] = target

        return df


class GermanCreditPreprocessor:
    """Preprocessor for German Credit dataset."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None

    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Preprocess the dataset."""
        df = df.copy()

        # Extract target
        if 'target' in df.columns:
            target = df['target']
            df = df.drop('target', axis=1)
        else:
            target = None

        # Feature engineering
        # Debt to income ratio
        df['DebtToIncome'] = df['credit_amount'] / (df['duration_months'] + 1)

        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100],
                                 labels=['young', 'middle', 'senior', 'old'])

        # Credit amount groups
        df['credit_group'] = pd.cut(df['credit_amount'],
                                    bins=[0, 2000, 5000, 10000, float('inf')],
                                    labels=['low', 'medium', 'high', 'very_high'])

        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        for col in categorical_cols:
            if is_train:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        # Store feature columns
        if is_train:
            self.feature_columns = df.columns.tolist()

        # Add target back if it exists
        if target is not None:
            df['target'] = target

        return df


def preprocess_all_datasets():
    """Preprocess both datasets and save to processed folder."""
    print("Preprocessing datasets...")

    # Kaggle Loan
    print("\n1. Processing Kaggle Loan dataset...")
    kaggle_train = pd.read_csv(RAW_DATA_DIR / "kaggle_loan" / "train.csv")
    kaggle_preprocessor = KaggleLoanPreprocessor()
    kaggle_processed = kaggle_preprocessor.preprocess(kaggle_train, is_train=True)

    kaggle_output = PROCESSED_DATA_DIR / "kaggle_loan_processed.csv"
    kaggle_processed.to_csv(kaggle_output, index=False)
    print(f"  ✓ Saved: {kaggle_output}")
    print(f"  Shape: {kaggle_processed.shape}")
    print(f"  Features: {kaggle_preprocessor.feature_columns}")

    # German Credit
    print("\n2. Processing German Credit dataset...")
    german_df = pd.read_csv(RAW_DATA_DIR / "german_credit" / "german_credit.csv")
    german_preprocessor = GermanCreditPreprocessor()
    german_processed = german_preprocessor.preprocess(german_df, is_train=True)

    german_output = PROCESSED_DATA_DIR / "german_credit_processed.csv"
    german_processed.to_csv(german_output, index=False)
    print(f"  ✓ Saved: {german_output}")
    print(f"  Shape: {german_processed.shape}")

    print("\n✓ Preprocessing complete!")
    return kaggle_processed, german_processed


if __name__ == "__main__":
    preprocess_all_datasets()
