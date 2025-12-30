import React, { useState } from 'react';
import './App.css';
import LoanForm from './components/LoanForm';
import PredictionResult from './components/PredictionResult';
import ApplicationHistory from './components/ApplicationHistory';

interface PredictionData {
  application_id: number;
  decision: string;
  confidence: number;
  probability_approved: number;
  probability_rejected: number;
  reasoning: string;
  feature_contributions: any[];
  top_positive_factors: any[];
  top_negative_factors: any[];
  created_at: string;
}

function App() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [refreshHistory, setRefreshHistory] = useState(0);

  const handlePredictionReceived = (data: PredictionData) => {
    setPrediction(data);
    setRefreshHistory(prev => prev + 1);
  };

  const handleNewApplication = () => {
    setPrediction(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏦 Loan Approval System</h1>
        <p>AI-powered loan decision with explainability</p>
      </header>

      <div className="App-content">
        <div className="App-main">
          {!prediction ? (
            <LoanForm onPredictionReceived={handlePredictionReceived} />
          ) : (
            <div>
              <PredictionResult prediction={prediction} />
              <button
                className="btn-new-application"
                onClick={handleNewApplication}
              >
                New Application
              </button>
            </div>
          )}
        </div>

        <div className="App-sidebar">
          <ApplicationHistory key={refreshHistory} />
        </div>
      </div>
    </div>
  );
}

export default App;
