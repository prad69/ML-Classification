"""FastAPI application for loan approval predictions."""
import sys
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
import numpy as np

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.api.database import init_db, get_db, LoanApplication
from src.api.schemas import (
    LoanApplicationRequest,
    PredictionResponse,
    LoanApplicationResponse,
    HealthResponse,
    FeatureContribution
)
from src.models.explainer import LoanExplainer
from src.data.preprocessing import KaggleLoanPreprocessor
from config import API_TITLE, API_VERSION

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Machine Learning-powered loan approval API with explainability"
)

# CORS middleware (allow React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
explainer: LoanExplainer = None
preprocessor: KaggleLoanPreprocessor = None


@app.on_event("startup")
async def startup_event():
    """Initialize database and load model on startup."""
    global explainer, preprocessor

    # Initialize database
    init_db()

    # Load model and explainer
    try:
        explainer = LoanExplainer()

        # Load and fit preprocessor with training data
        preprocessor = KaggleLoanPreprocessor()
        train_data = pd.read_csv(BASE_DIR / "data" / "raw" / "kaggle_loan" / "train.csv")
        _ = preprocessor.preprocess(train_data, is_train=True)  # Fit the preprocessor

        print("✓ Model and explainer loaded successfully")
        print("✓ Preprocessor fitted")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": explainer is not None,
        "model_name": explainer.model_name if explainer else "Not loaded"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_approval(
    application: LoanApplicationRequest,
    db: Session = Depends(get_db)
):
    """
    Predict loan approval and provide explanation.

    - **gender**: Male or Female
    - **married**: Yes or No
    - **dependents**: 0, 1, 2, or 3+
    - **education**: Graduate or Not Graduate
    - **self_employed**: Yes or No
    - **applicant_income**: Applicant's income
    - **coapplicant_income**: Coapplicant's income
    - **loan_amount**: Requested loan amount
    - **loan_amount_term**: Loan term in months
    - **credit_history**: 1.0 for good credit, 0.0 for bad credit
    - **property_area**: Urban, Rural, or Semiurban
    """
    try:
        # Convert request to DataFrame
        data = {
            'Gender': [application.gender],
            'Married': [application.married],
            'Dependents': [application.dependents],
            'Education': [application.education],
            'Self_Employed': [application.self_employed],
            'ApplicantIncome': [application.applicant_income],
            'CoapplicantIncome': [application.coapplicant_income],
            'LoanAmount': [application.loan_amount],
            'Loan_Amount_Term': [application.loan_amount_term],
            'Credit_History': [application.credit_history],
            'Property_Area': [application.property_area],
        }
        df = pd.DataFrame(data)

        # Preprocess
        processed_df = preprocessor.preprocess(df, is_train=False)

        # Get prediction and explanation
        explanation = explainer.explain_prediction(processed_df)

        # Save to database
        db_application = LoanApplication(
            gender=application.gender,
            married=application.married,
            dependents=application.dependents,
            education=application.education,
            self_employed=application.self_employed,
            applicant_income=application.applicant_income,
            coapplicant_income=application.coapplicant_income,
            loan_amount=application.loan_amount,
            loan_amount_term=application.loan_amount_term,
            credit_history=application.credit_history,
            property_area=application.property_area,
            prediction=explanation['decision'],
            confidence=explanation['confidence'],
            probability_approved=explanation['probability_approved'],
            probability_rejected=explanation['probability_rejected'],
            explanation=explanation
        )
        db.add(db_application)
        db.commit()
        db.refresh(db_application)

        # Format response
        response = PredictionResponse(
            application_id=db_application.id,
            decision=explanation['decision'],
            confidence=explanation['confidence'],
            probability_approved=explanation['probability_approved'],
            probability_rejected=explanation['probability_rejected'],
            reasoning=explanation['reasoning'],
            feature_contributions=[FeatureContribution(**fc) for fc in explanation['feature_contributions']],
            top_positive_factors=[FeatureContribution(**fc) for fc in explanation['top_positive_factors']],
            top_negative_factors=[FeatureContribution(**fc) for fc in explanation['top_negative_factors']],
            created_at=db_application.created_at
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/applications", response_model=List[LoanApplicationResponse])
async def get_applications(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all loan applications."""
    applications = db.query(LoanApplication).offset(skip).limit(limit).all()
    return applications


@app.get("/applications/{application_id}", response_model=PredictionResponse)
async def get_application(
    application_id: int,
    db: Session = Depends(get_db)
):
    """Get specific loan application with full explanation."""
    application = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()

    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    # Reconstruct full response from stored data
    explanation = application.explanation

    response = PredictionResponse(
        application_id=application.id,
        decision=explanation['decision'],
        confidence=explanation['confidence'],
        probability_approved=explanation['probability_approved'],
        probability_rejected=explanation['probability_rejected'],
        reasoning=explanation['reasoning'],
        feature_contributions=[FeatureContribution(**fc) for fc in explanation['feature_contributions']],
        top_positive_factors=[FeatureContribution(**fc) for fc in explanation['top_positive_factors']],
        top_negative_factors=[FeatureContribution(**fc) for fc in explanation['top_negative_factors']],
        created_at=application.created_at
    )

    return response


@app.delete("/applications/{application_id}")
async def delete_application(
    application_id: int,
    db: Session = Depends(get_db)
):
    """Delete a loan application."""
    application = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()

    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    db.delete(application)
    db.commit()

    return {"message": f"Application {application_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
