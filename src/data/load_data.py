"""Load and combine loan datasets."""
import pandas as pd
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def load_kaggle_loan_data() -> pd.DataFrame:
    """Load Kaggle Loan Prediction dataset."""
    train_path = RAW_DATA_DIR / "kaggle_loan" / "train.csv"
    df = pd.read_csv(train_path)
    print(f"Kaggle Loan Data loaded: {df.shape}")
    return df


def load_german_credit_data() -> pd.DataFrame:
    """Load German Credit dataset."""
    path = RAW_DATA_DIR / "german_credit" / "german_credit.csv"
    df = pd.read_csv(path)
    print(f"German Credit Data loaded: {df.shape}")
    return df


def get_dataset_info(df: pd.DataFrame, name: str) -> None:
    """Print dataset information."""
    print(f"\n{'=' * 60}")
    print(f"{name} Dataset Info")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nTarget distribution:\n{df.iloc[:, -1].value_counts()}")


if __name__ == "__main__":
    # Load datasets
    kaggle_df = load_kaggle_loan_data()
    german_df = load_german_credit_data()

    # Show info
    get_dataset_info(kaggle_df, "Kaggle Loan")
    get_dataset_info(german_df, "German Credit")
