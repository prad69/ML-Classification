import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ApplicationHistory.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface Application {
  id: number;
  created_at: string;
  prediction: string;
  confidence: number;
  applicant_income: number;
  loan_amount: number;
}

const ApplicationHistory: React.FC = () => {
  const [applications, setApplications] = useState<Application[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchApplications();
  }, []);

  const fetchApplications = async () => {
    try {
      const response = await axios.get(`${API_URL}/applications?limit=50`);
      setApplications(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load applications');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="history-loading">Loading...</div>;
  }

  if (error) {
    return <div className="history-error">{error}</div>;
  }

  if (applications.length === 0) {
    return (
      <div className="history-container">
        <h3>Application History</h3>
        <p className="no-applications">No applications yet</p>
      </div>
    );
  }

  return (
    <div className="history-container">
      <h3>Recent Applications</h3>
      <div className="history-list">
        {applications.map(app => (
          <div key={app.id} className={`history-item ${app.prediction.toLowerCase()}`}>
            <div className="history-header">
              <span className={`status-badge ${app.prediction.toLowerCase()}`}>
                {app.prediction}
              </span>
              <span className="application-id">#{app.id}</span>
            </div>
            <div className="history-details">
              <div className="history-row">
                <span>Income:</span>
                <span>${app.applicant_income}</span>
              </div>
              <div className="history-row">
                <span>Loan:</span>
                <span>${app.loan_amount}k</span>
              </div>
              <div className="history-row">
                <span>Confidence:</span>
                <span>{app.confidence.toFixed(1)}%</span>
              </div>
              <div className="history-date">
                {new Date(app.created_at).toLocaleString()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ApplicationHistory;
