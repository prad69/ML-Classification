"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class LoanApplicationRequest(BaseModel):
    """Loan application input."""
    gender: str = Field(..., description="Gender: Male or Female")
    married: str = Field(..., description="Married: Yes or No")
    dependents: str = Field(..., description="Dependents: 0, 1, 2, 3+")
    education: str = Field(..., description="Education: Graduate or Not Graduate")
    self_employed: str = Field(..., description="Self Employed: Yes or No")
    applicant_income: float = Field(..., description="Applicant Income", gt=0)
    coapplicant_income: float = Field(..., description="Coapplicant Income", ge=0)
    loan_amount: float = Field(..., description="Loan Amount", gt=0)
    loan_amount_term: float = Field(..., description="Loan Amount Term in months", gt=0)
    credit_history: float = Field(..., description="Credit History: 1.0 (good) or 0.0 (bad)")
    property_area: str = Field(..., description="Property Area: Urban, Rural, or Semiurban")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "married": "Yes",
                "dependents": "0",
                "education": "Graduate",
                "self_employed": "No",
                "applicant_income": 5849,
                "coapplicant_income": 0,
                "loan_amount": 128,
                "loan_amount_term": 360,
                "credit_history": 1.0,
                "property_area": "Urban"
            }
        }


class FeatureContribution(BaseModel):
    """Feature contribution to prediction."""
    feature: str
    value: float
    contribution: float
    abs_contribution: float


class PredictionResponse(BaseModel):
    """Prediction response with explanation."""
    application_id: int
    decision: str
    confidence: float
    probability_approved: float
    probability_rejected: float
    reasoning: str
    feature_contributions: List[FeatureContribution]
    top_positive_factors: List[FeatureContribution]
    top_negative_factors: List[FeatureContribution]
    created_at: datetime


class LoanApplicationResponse(BaseModel):
    """Loan application record."""
    id: int
    created_at: datetime
    gender: str
    married: str
    dependents: str
    education: str
    self_employed: str
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_amount_term: float
    credit_history: float
    property_area: str
    prediction: str
    confidence: float
    probability_approved: float
    probability_rejected: float

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
