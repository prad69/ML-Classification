"""Configuration settings for the loan approval application."""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "loan_model.pkl"
EXPLAINER_PATH = MODEL_DIR / "shap_explainer.pkl"

# Database
DATABASE_URL = "sqlite:///./loan_applications.db"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Loan Approval API"
API_VERSION = "1.0.0"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature columns (will be populated after data analysis)
FEATURE_COLUMNS = []
TARGET_COLUMN = "loan_status"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
