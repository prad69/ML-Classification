"""Create sample datasets based on real loan dataset structures."""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def create_kaggle_loan_dataset(n_samples=614):
    """Create sample Kaggle Loan Prediction dataset."""
    np.random.seed(42)

    data = {
        'Loan_ID': [f'LP{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.8, 0.2]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.4, 0.25, 0.2, 0.15]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.78, 0.22]),
        'Self_Employed': np.random.choice(['No', 'Yes'], n_samples, p=[0.86, 0.14]),
        'ApplicantIncome': np.random.lognormal(8.5, 0.6, n_samples).astype(int),
        'CoapplicantIncome': np.random.lognormal(6.5, 1.2, n_samples).astype(int),
        'LoanAmount': np.random.lognormal(4.8, 0.4, n_samples).astype(int),
        'Loan_Amount_Term': np.random.choice([360, 180, 240, 300, 480, 120], n_samples, p=[0.8, 0.05, 0.05, 0.03, 0.02, 0.05]),
        'Credit_History': np.random.choice([1.0, 0.0, np.nan], n_samples, p=[0.84, 0.08, 0.08]),
        'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], n_samples, p=[0.38, 0.18, 0.44]),
    }

    df = pd.DataFrame(data)

    # Create target based on realistic loan approval logic
    approval_score = (
        (df['Credit_History'] == 1.0).astype(float) * 0.5 +
        (df['Education'] == 'Graduate').astype(float) * 0.2 +
        (df['Property_Area'] == 'Urban').astype(float) * 0.1 +
        (df['ApplicantIncome'] > 5000).astype(float) * 0.15 +
        (df['LoanAmount'] < 200).astype(float) * 0.05 +
        np.random.random(n_samples) * 0.3  # Add randomness
    )

    df['Loan_Status'] = (approval_score > 0.6).map({True: 'Y', False: 'N'})

    # Add some missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term']:
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan

    return df


def create_german_credit_dataset(n_samples=1000):
    """Create sample German Credit dataset."""
    np.random.seed(42)

    # Categorical features with their possible values
    status_checking = ['A11', 'A12', 'A13', 'A14']  # no checking, <0 DM, 0-200 DM, >200 DM
    credit_history = ['A30', 'A31', 'A32', 'A33', 'A34']  # no credits, all paid, existing paid, delay, critical
    purpose = ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410']
    savings = ['A61', 'A62', 'A63', 'A64', 'A65']  # <100 DM, 100-500, 500-1000, >1000, unknown
    employment = ['A71', 'A72', 'A73', 'A74', 'A75']  # unemployed, <1 year, 1-4, 4-7, >7 years
    personal_status = ['A91', 'A92', 'A93', 'A94', 'A95']  # male:divorced, female:divorced, male:single, male:married, female:single
    other_debtors = ['A101', 'A102', 'A103']  # none, co-applicant, guarantor
    property = ['A121', 'A122', 'A123', 'A124']  # real estate, savings, car, unknown
    other_plans = ['A141', 'A142', 'A143']  # bank, stores, none
    housing = ['A151', 'A152', 'A153']  # rent, own, free
    job = ['A171', 'A172', 'A173', 'A174']  # unemployed, unskilled, skilled, highly skilled
    telephone = ['A191', 'A192']  # none, yes
    foreign_worker = ['A201', 'A202']  # yes, no

    data = {
        'status_checking_account': np.random.choice(status_checking, n_samples, p=[0.27, 0.27, 0.27, 0.19]),
        'duration_months': np.random.randint(4, 73, n_samples),
        'credit_history': np.random.choice(credit_history, n_samples, p=[0.04, 0.05, 0.53, 0.09, 0.29]),
        'purpose': np.random.choice(purpose, n_samples),
        'credit_amount': np.random.lognormal(8.0, 0.8, n_samples).astype(int),
        'savings_account': np.random.choice(savings, n_samples, p=[0.60, 0.10, 0.05, 0.05, 0.20]),
        'employment_since': np.random.choice(employment, n_samples, p=[0.06, 0.17, 0.34, 0.17, 0.26]),
        'installment_rate': np.random.choice([1, 2, 3, 4], n_samples, p=[0.05, 0.23, 0.16, 0.56]),
        'personal_status_sex': np.random.choice(personal_status, n_samples, p=[0.05, 0.31, 0.55, 0.05, 0.04]),
        'other_debtors': np.random.choice(other_debtors, n_samples, p=[0.91, 0.04, 0.05]),
        'present_residence_since': np.random.choice([1, 2, 3, 4], n_samples, p=[0.13, 0.18, 0.15, 0.54]),
        'property': np.random.choice(property, n_samples, p=[0.28, 0.23, 0.33, 0.16]),
        'age': np.random.randint(19, 76, n_samples),
        'other_installment_plans': np.random.choice(other_plans, n_samples, p=[0.19, 0.05, 0.76]),
        'housing': np.random.choice(housing, n_samples, p=[0.18, 0.71, 0.11]),
        'num_existing_credits': np.random.choice([1, 2, 3, 4], n_samples, p=[0.63, 0.33, 0.03, 0.01]),
        'job': np.random.choice(job, n_samples, p=[0.02, 0.20, 0.63, 0.15]),
        'num_dependents': np.random.choice([1, 2], n_samples, p=[0.85, 0.15]),
        'telephone': np.random.choice(telephone, n_samples, p=[0.60, 0.40]),
        'foreign_worker': np.random.choice(foreign_worker, n_samples, p=[0.96, 0.04]),
    }

    df = pd.DataFrame(data)

    # Create target based on realistic credit risk logic
    risk_score = (
        (df['credit_history'].isin(['A32', 'A33'])).astype(float) * 0.3 +
        (df['status_checking_account'].isin(['A13', 'A14'])).astype(float) * 0.25 +
        (df['savings_account'].isin(['A63', 'A64', 'A65'])).astype(float) * 0.15 +
        (df['employment_since'].isin(['A74', 'A75'])).astype(float) * 0.1 +
        (df['age'] > 30).astype(float) * 0.1 +
        (df['duration_months'] < 24).astype(float) * 0.1 +
        np.random.random(n_samples) * 0.3
    )

    # Target: 1 = Good credit, 0 = Bad credit
    df['target'] = (risk_score > 0.5).astype(int)

    return df


if __name__ == "__main__":
    print("Creating sample datasets...")

    # Create Kaggle Loan dataset
    kaggle_df = create_kaggle_loan_dataset(614)
    kaggle_path = RAW_DATA_DIR / "kaggle_loan" / "train.csv"
    kaggle_df.to_csv(kaggle_path, index=False)
    print(f"✓ Kaggle Loan dataset: {kaggle_path}")
    print(f"  Shape: {kaggle_df.shape}")
    print(f"  Approval rate: {(kaggle_df['Loan_Status'] == 'Y').mean():.2%}")

    # Create test set (smaller)
    kaggle_test = create_kaggle_loan_dataset(367)
    kaggle_test = kaggle_test.drop(columns=['Loan_Status'])  # Test set has no target
    kaggle_test_path = RAW_DATA_DIR / "kaggle_loan" / "test.csv"
    kaggle_test.to_csv(kaggle_test_path, index=False)
    print(f"✓ Kaggle Loan test set: {kaggle_test_path}")

    # Create German Credit dataset
    german_df = create_german_credit_dataset(1000)
    german_path = RAW_DATA_DIR / "german_credit" / "german_credit.csv"
    german_df.to_csv(german_path, index=False)
    print(f"✓ German Credit dataset: {german_path}")
    print(f"  Shape: {german_df.shape}")
    print(f"  Good credit rate: {german_df['target'].mean():.2%}")

    print("\n✓ Sample datasets created successfully!")
