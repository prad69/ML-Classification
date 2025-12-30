import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import './PredictionResult.css';

interface FeatureContribution {
  feature: string;
  value: number;
  contribution: number;
  abs_contribution: number;
}

interface PredictionResultProps {
  prediction: {
    application_id: number;
    decision: string;
    confidence: number;
    probability_approved: number;
    probability_rejected: number;
    reasoning: string;
    feature_contributions: FeatureContribution[];
    top_positive_factors: FeatureContribution[];
    top_negative_factors: FeatureContribution[];
    created_at: string;
  };
}

const PredictionResult: React.FC<PredictionResultProps> = ({ prediction }) => {
  const isApproved = prediction.decision === 'APPROVED';

  // Prepare data for chart - top 10 features
  const chartData = prediction.feature_contributions.slice(0, 10).map(fc => ({
    name: fc.feature,
    contribution: fc.contribution,
    value: fc.value
  }));

  return (
    <div className="prediction-result">
      <div className={`result-header ${isApproved ? 'approved' : 'rejected'}`}>
        <h2>{isApproved ? '✓ Loan Approved' : '✗ Loan Rejected'}</h2>
        <div className="confidence">
          Confidence: {prediction.confidence.toFixed(2)}%
        </div>
      </div>

      <div className="probabilities">
        <div className="prob-bar">
          <div className="prob-label">Approval Probability</div>
          <div className="prob-value">{prediction.probability_approved.toFixed(2)}%</div>
          <div className="prob-bar-fill" style={{ width: `${prediction.probability_approved}%`, backgroundColor: '#4caf50' }} />
        </div>
        <div className="prob-bar">
          <div className="prob-label">Rejection Probability</div>
          <div className="prob-value">{prediction.probability_rejected.toFixed(2)}%</div>
          <div className="prob-bar-fill" style={{ width: `${prediction.probability_rejected}%`, backgroundColor: '#f44336' }} />
        </div>
      </div>

      <div className="reasoning-section">
        <h3>Decision Reasoning</h3>
        <p>{prediction.reasoning}</p>
      </div>

      <div className="factors-section">
        <div className="factors-column">
          <h3>Positive Factors (+)</h3>
          <ul>
            {prediction.top_positive_factors.map((factor, idx) => (
              <li key={idx} className="factor-item positive">
                <span className="factor-name">{factor.feature}</span>
                <span className="factor-value">+{factor.contribution.toFixed(4)}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="factors-column">
          <h3>Negative Factors (-)</h3>
          <ul>
            {prediction.top_negative_factors.map((factor, idx) => (
              <li key={idx} className="factor-item negative">
                <span className="factor-name">{factor.feature}</span>
                <span className="factor-value">{factor.contribution.toFixed(4)}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="chart-section">
        <h3>Feature Contributions (Top 10)</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={chartData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" width={150} />
            <Tooltip />
            <Legend />
            <Bar dataKey="contribution" name="Contribution">
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.contribution > 0 ? '#4caf50' : '#f44336'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="application-id">
        Application ID: {prediction.application_id}
      </div>
    </div>
  );
};

export default PredictionResult;
