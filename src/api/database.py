"""Database models and setup for loan applications."""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from config import DATABASE_URL

Base = declarative_base()


class LoanApplication(Base):
    """Loan application record."""
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Applicant information
    gender = Column(String)
    married = Column(String)
    dependents = Column(String)
    education = Column(String)
    self_employed = Column(String)

    # Financial information
    applicant_income = Column(Float)
    coapplicant_income = Column(Float)
    loan_amount = Column(Float)
    loan_amount_term = Column(Float)
    credit_history = Column(Float)
    property_area = Column(String)

    # Prediction results
    prediction = Column(String)  # APPROVED or REJECTED
    confidence = Column(Float)
    probability_approved = Column(Float)
    probability_rejected = Column(Float)

    # Explanation (stored as JSON)
    explanation = Column(JSON)


# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database."""
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized")


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
