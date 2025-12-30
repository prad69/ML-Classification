"""Download loan datasets from public sources."""
import os
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Create directories
(RAW_DATA_DIR / "kaggle_loan").mkdir(parents=True, exist_ok=True)
(RAW_DATA_DIR / "german_credit").mkdir(parents=True, exist_ok=True)


def download_german_credit_data():
    """Download German Credit Data from UCI ML Repository."""
    print("Downloading German Credit Data...")

    urls = {
        "german.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        "german.doc": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc",
    }

    for filename, url in urls.items():
        filepath = RAW_DATA_DIR / "german_credit" / filename
        if filepath.exists():
            print(f"  {filename} already exists, skipping...")
            continue

        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ {filename} downloaded successfully")
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")

    # Convert german.data to CSV with proper column names
    column_names = [
        'status_checking_account', 'duration_months', 'credit_history',
        'purpose', 'credit_amount', 'savings_account', 'employment_since',
        'installment_rate', 'personal_status_sex', 'other_debtors',
        'present_residence_since', 'property', 'age', 'other_installment_plans',
        'housing', 'num_existing_credits', 'job', 'num_dependents',
        'telephone', 'foreign_worker', 'target'
    ]

    try:
        df = pd.read_csv(
            RAW_DATA_DIR / "german_credit" / "german.data",
            sep=' ',
            names=column_names,
            header=None
        )
        # Target: 1 = Good, 2 = Bad -> Convert to 1 = Good, 0 = Bad
        df['target'] = (df['target'] == 1).astype(int)

        csv_path = RAW_DATA_DIR / "german_credit" / "german_credit.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Converted to CSV: {csv_path}")
        print(f"  Dataset shape: {df.shape}")
    except Exception as e:
        print(f"  ✗ Error converting to CSV: {e}")


def download_kaggle_loan_data():
    """
    Download Kaggle Loan Prediction dataset.

    Note: This requires Kaggle API credentials.
    Alternative: Download manually from Kaggle website.
    """
    print("\nDownloading Kaggle Loan Prediction Data...")
    print("  Note: This requires Kaggle API credentials")

    try:
        import kaggle

        # Download using Kaggle API
        kaggle.api.dataset_download_files(
            'altruistdelhite04/loan-prediction-problem-dataset',
            path=str(RAW_DATA_DIR / "kaggle_loan"),
            unzip=True
        )
        print("  ✓ Kaggle dataset downloaded successfully")

        # List downloaded files
        files = list((RAW_DATA_DIR / "kaggle_loan").glob("*.csv"))
        for f in files:
            df = pd.read_csv(f)
            print(f"  ✓ {f.name}: {df.shape}")

    except ImportError:
        print("  ⚠ Kaggle package not installed")
        print("  Install with: pip install kaggle")
        print("\n  OR download manually:")
        print("  1. Visit: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset")
        print("  2. Download the dataset")
        print("  3. Extract to: data/raw/kaggle_loan/")
    except Exception as e:
        print(f"  ⚠ Could not download via API: {e}")
        print("\n  Please download manually:")
        print("  1. Visit: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset")
        print("  2. Download the dataset")
        print("  3. Extract to: data/raw/kaggle_loan/")


def create_sample_kaggle_data():
    """Create a sample Kaggle-like dataset for development if API download fails."""
    print("\nCreating sample Kaggle loan dataset for development...")

    sample_data = {
        'Loan_ID': ['LP001002', 'LP001003', 'LP001005', 'LP001006', 'LP001008'],
        'Gender': ['Male', 'Male', 'Male', 'Male', 'Male'],
        'Married': ['No', 'Yes', 'Yes', 'Yes', 'No'],
        'Dependents': ['0', '1', '0', '0', '0'],
        'Education': ['Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate'],
        'Self_Employed': ['No', 'No', 'Yes', 'No', 'No'],
        'ApplicantIncome': [5849, 4583, 3000, 2583, 6000],
        'CoapplicantIncome': [0, 1508, 0, 2358, 0],
        'LoanAmount': [128, 128, 66, 120, 141],
        'Loan_Amount_Term': [360, 360, 360, 360, 360],
        'Credit_History': [1.0, 1.0, 1.0, 1.0, 1.0],
        'Property_Area': ['Urban', 'Rural', 'Urban', 'Urban', 'Urban'],
        'Loan_Status': ['Y', 'N', 'Y', 'Y', 'Y']
    }

    df = pd.DataFrame(sample_data)
    sample_path = RAW_DATA_DIR / "kaggle_loan" / "train_sample.csv"
    df.to_csv(sample_path, index=False)
    print(f"  ✓ Sample dataset created: {sample_path}")
    print(f"  (Replace with real data when available)")


if __name__ == "__main__":
    print("=" * 60)
    print("Loan Dataset Download Script")
    print("=" * 60)

    # Download German Credit Data (publicly available)
    download_german_credit_data()

    # Attempt to download Kaggle data
    download_kaggle_loan_data()

    # Check if Kaggle data exists, if not create sample
    kaggle_files = list((RAW_DATA_DIR / "kaggle_loan").glob("train*.csv"))
    if not kaggle_files:
        create_sample_kaggle_data()

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nData location: {RAW_DATA_DIR}")
    print("\nNext steps:")
    print("  1. Review data in notebooks/01_eda.ipynb")
    print("  2. Run preprocessing: python src/data/preprocessing.py")
