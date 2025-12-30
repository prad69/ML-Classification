# 🏦 ML-Powered Loan Approval System

An end-to-end machine learning application that predicts loan approval decisions with **explainable AI** (XAI) using SHAP values. The system includes a FastAPI backend and React dashboard to visualize why loans are approved or rejected.

## 🌟 Features

- **Machine Learning Models**: Trained on Kaggle Loan Prediction and German Credit datasets
- **Model Explainability**: SHAP (SHapley Additive exPlanations) for transparent decision-making
- **REST API**: FastAPI backend with automatic OpenAPI documentation
- **Interactive Dashboard**: React TypeScript frontend with real-time predictions
- **Visualization**: Feature contribution charts showing positive and negative factors
- **Application History**: Track all past loan applications
- **Database**: SQLite storage for all predictions and explanations

## 📊 Model Performance

**Best Model: Random Forest Classifier**
- **Test Accuracy**: 92.68%
- **F1-Score**: 95.89%
- **Cross-Validation F1**: 93.50% (±1.32%)
- **Precision**: 92.11%
- **Recall**: 100.00%

### Feature Importance (Top 5)
1. **Credit_History** (20.33%) - Most critical factor
2. **LoanAmountToIncome** (8.84%)
3. **IncomePerDependent** (8.78%)
4. **ApplicantIncome** (8.69%)
5. **EMIToIncome** (8.19%)

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │  (Port 3000)
│   - Loan Form   │
│   - Results     │
│   - Charts      │
└────────┬────────┘
         │
         ↓ HTTP/REST
┌────────────────────┐
│   FastAPI Backend  │  (Port 8000)
│   - /predict       │
│   - /applications  │
└────────┬───────────┘
         │
    ┌────┴─────┬──────────┐
    ↓          ↓          ↓
┌────────┐ ┌────────┐ ┌─────────┐
│  Model │ │  SHAP  │ │ SQLite  │
│  .pkl  │ │Explain │ │   DB    │
└────────┘ └────────┘ └─────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ML-Classification
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Create sample datasets** (if real datasets not available)
```bash
python src/data/create_sample_data.py
```

4. **Preprocess data**
```bash
python src/data/preprocessing.py
```

5. **Train the model**
```bash
python src/models/train.py
```

6. **Install frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

### Running the Application

#### Option 1: Run Backend and Frontend Separately

**Terminal 1 - Backend:**
```bash
python src/api/main.py
```
Backend will start at http://localhost:8000

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```
Frontend will start at http://localhost:3000

#### Option 2: Using the Run Script (Recommended)
```bash
# Create run script
cat > run.sh << 'EOF'
#!/bin/bash
echo "Starting Loan Approval System..."

# Start backend in background
echo "Starting FastAPI backend..."
python src/api/main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting React frontend..."
cd frontend
npm start &
FRONTEND_PID=$!

echo "✓ System running!"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop..."

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
EOF

chmod +x run.sh
./run.sh
```

## 📖 API Documentation

Once the backend is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Key Endpoints

#### POST `/predict`
Submit a loan application and get approval decision with explanation.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "application_id": 1,
  "decision": "APPROVED",
  "confidence": 88.0,
  "probability_approved": 88.0,
  "probability_rejected": 12.0,
  "reasoning": "Loan application approved. Key supporting factors: ...",
  "feature_contributions": [...],
  "top_positive_factors": [...],
  "top_negative_factors": [...],
  "created_at": "2024-01-01T12:00:00"
}
```

#### GET `/applications`
Retrieve all loan applications with pagination.

#### GET `/applications/{id}`
Get detailed explanation for a specific application.

## 🎨 Dashboard Features

### 1. Loan Application Form
- Personal information inputs
- Financial details
- Real-time validation
- Clean, intuitive UI

### 2. Prediction Results
- **Approval/Rejection Badge**: Clear visual indicator
- **Confidence Score**: Model confidence percentage
- **Probability Bars**: Visual representation of approval/rejection probabilities
- **Reasoning**: Plain English explanation of the decision

### 3. Feature Contributions Visualization
- **Horizontal Bar Chart**: Shows top 10 feature impacts
- **Color Coding**:
  - 🟢 Green = Positive contribution (helps approval)
  - 🔴 Red = Negative contribution (hurts approval)
- **Interactive**: Hover for details

### 4. Factors Breakdown
- **Positive Factors**: Features that supported approval
- **Negative Factors**: Features that opposed approval
- Shows actual SHAP values for transparency

### 5. Application History
- Sidebar with recent applications
- Quick view of past decisions
- Filterable and searchable (future enhancement)

## 🧪 Testing the System

### Test with Sample Applications

**High Approval Probability:**
```
Gender: Male
Married: Yes
Dependents: 0
Education: Graduate
Self Employed: No
Applicant Income: $8000
Coapplicant Income: $2000
Loan Amount: $100k
Loan Term: 360 months
Credit History: Good (1.0)
Property Area: Urban
```

**Low Approval Probability:**
```
Gender: Female
Married: No
Dependents: 3+
Education: Not Graduate
Self Employed: Yes
Applicant Income: $2000
Coapplicant Income: $0
Loan Amount: $300k
Loan Term: 120 months
Credit History: Bad (0.0)
Property Area: Rural
```

## 📁 Project Structure

```
ML-Classification/
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── kaggle_loan/
│   │   └── german_credit/
│   └── processed/              # Preprocessed data
├── src/
│   ├── data/
│   │   ├── create_sample_data.py
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── train.py            # Model training
│   │   └── explainer.py        # SHAP explainability
│   └── api/
│       ├── main.py             # FastAPI app
│       ├── database.py         # SQLAlchemy models
│       └── schemas.py          # Pydantic schemas
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LoanForm.tsx
│   │   │   ├── PredictionResult.tsx
│   │   │   └── ApplicationHistory.tsx
│   │   ├── App.tsx
│   │   └── App.css
│   └── package.json
├── models/
│   ├── kaggle_loan_model.pkl   # Trained model
│   └── loan_explainer.pkl      # SHAP explainer
├── config.py
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Backend Configuration (`config.py`)
- API host and port
- Database URL
- Model paths
- Feature columns

### Frontend Configuration
Create `.env` file in `frontend/`:
```
REACT_APP_API_URL=http://localhost:8000
```

## 🎯 How It Works

### 1. Data Collection & Preprocessing
- Loads Kaggle Loan and German Credit datasets
- Handles missing values (median for numerical, mode for categorical)
- Feature engineering:
  - Total Income = Applicant + Coapplicant
  - Loan-to-Income Ratio
  - EMI (Estimated Monthly Installment)
  - Income per Dependent
- Label encoding for categorical features

### 2. Model Training
- Trains 4 models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Uses 5-fold cross-validation
- Selects best model based on F1-score
- Saves model with metadata

### 3. SHAP Explainability
- Creates TreeExplainer for Random Forest
- Calculates SHAP values for each prediction
- Generates:
  - Feature contributions (positive/negative)
  - Global feature importance
  - Plain English reasoning

### 4. API Serving
- FastAPI loads model and explainer
- Receives loan application JSON
- Preprocesses input data
- Makes prediction
- Generates SHAP explanation
- Stores in database
- Returns detailed response

### 5. Dashboard Visualization
- React receives API response
- Displays decision with confidence
- Shows feature contributions as charts
- Lists positive/negative factors
- Provides reasoning text

## 🚨 Troubleshooting

### Backend Issues

**Error: "ModuleNotFoundError: No module named 'xxx'"**
```bash
pip install -r requirements.txt
```

**Error: "Model file not found"**
```bash
python src/models/train.py
```

**Port 8000 already in use:**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

### Frontend Issues

**Error: "Cannot find module 'axios'"**
```bash
cd frontend
npm install
```

**CORS errors:**
- Ensure backend is running
- Check REACT_APP_API_URL in `.env`
- Verify CORS middleware in `src/api/main.py`

## 📊 Datasets

### Kaggle Loan Prediction Dataset
- **Size**: 614 records
- **Features**: 12 features including gender, income, credit history
- **Target**: Loan_Status (Y/N)

### German Credit Dataset
- **Size**: 1000 records
- **Features**: 20 attributes including credit history, purpose, employment
- **Target**: Credit risk (Good/Bad)

## 🔮 Future Enhancements

- [ ] Add authentication and user management
- [ ] Implement model retraining pipeline
- [ ] Add more ML models (Neural Networks, SVM)
- [ ] Enhanced visualizations (LIME, force plots)
- [ ] Export reports as PDF
- [ ] A/B testing for models
- [ ] Real-time model monitoring
- [ ] Deployment to cloud (AWS, GCP, Azure)
- [ ] Docker containerization
- [ ] CI/CD pipeline

## 📝 License

This project is for educational purposes.

## 👥 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Built with ❤️ using Python, FastAPI, React, and SHAP**
