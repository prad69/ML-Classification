import React, { useState } from 'react';
import axios from 'axios';
import './LoanForm.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface LoanFormProps {
  onPredictionReceived: (data: any) => void;
}

const LoanForm: React.FC<LoanFormProps> = ({ onPredictionReceived }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    gender: 'Male',
    married: 'Yes',
    dependents: '0',
    education: 'Graduate',
    self_employed: 'No',
    applicant_income: 5000,
    coapplicant_income: 0,
    loan_amount: 128,
    loan_amount_term: 360,
    credit_history: 1.0,
    property_area: 'Urban'
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: ['applicant_income', 'coapplicant_income', 'loan_amount', 'loan_amount_term', 'credit_history'].includes(name)
        ? parseFloat(value)
        : value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      onPredictionReceived(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="loan-form-container">
      <h2>Loan Application Form</h2>

      {error && <div className="error-message">{error}</div>}

      <form onSubmit={handleSubmit} className="loan-form">
        <div className="form-section">
          <h3>Personal Information</h3>

          <div className="form-group">
            <label>Gender</label>
            <select name="gender" value={formData.gender} onChange={handleChange}>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>

          <div className="form-group">
            <label>Married</label>
            <select name="married" value={formData.married} onChange={handleChange}>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>

          <div className="form-group">
            <label>Dependents</label>
            <select name="dependents" value={formData.dependents} onChange={handleChange}>
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3+">3+</option>
            </select>
          </div>

          <div className="form-group">
            <label>Education</label>
            <select name="education" value={formData.education} onChange={handleChange}>
              <option value="Graduate">Graduate</option>
              <option value="Not Graduate">Not Graduate</option>
            </select>
          </div>

          <div className="form-group">
            <label>Self Employed</label>
            <select name="self_employed" value={formData.self_employed} onChange={handleChange}>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>
        </div>

        <div className="form-section">
          <h3>Financial Information</h3>

          <div className="form-group">
            <label>Applicant Income ($)</label>
            <input
              type="number"
              name="applicant_income"
              value={formData.applicant_income}
              onChange={handleChange}
              min="0"
              step="100"
              required
            />
          </div>

          <div className="form-group">
            <label>Coapplicant Income ($)</label>
            <input
              type="number"
              name="coapplicant_income"
              value={formData.coapplicant_income}
              onChange={handleChange}
              min="0"
              step="100"
            />
          </div>

          <div className="form-group">
            <label>Loan Amount ($1000s)</label>
            <input
              type="number"
              name="loan_amount"
              value={formData.loan_amount}
              onChange={handleChange}
              min="1"
              step="1"
              required
            />
          </div>

          <div className="form-group">
            <label>Loan Term (months)</label>
            <select name="loan_amount_term" value={formData.loan_amount_term} onChange={handleChange}>
              <option value="120">120</option>
              <option value="180">180</option>
              <option value="240">240</option>
              <option value="300">300</option>
              <option value="360">360</option>
              <option value="480">480</option>
            </select>
          </div>

          <div className="form-group">
            <label>Credit History</label>
            <select name="credit_history" value={formData.credit_history} onChange={handleChange}>
              <option value="1.0">Good (1.0)</option>
              <option value="0.0">Bad (0.0)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Property Area</label>
            <select name="property_area" value={formData.property_area} onChange={handleChange}>
              <option value="Urban">Urban</option>
              <option value="Rural">Rural</option>
              <option value="Semiurban">Semiurban</option>
            </select>
          </div>
        </div>

        <button type="submit" className="btn-submit" disabled={loading}>
          {loading ? 'Processing...' : 'Submit Application'}
        </button>
      </form>
    </div>
  );
};

export default LoanForm;
