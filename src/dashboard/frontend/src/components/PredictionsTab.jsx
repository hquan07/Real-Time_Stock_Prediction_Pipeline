import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Target, TrendingUp, AlertTriangle } from 'lucide-react';
import useStore from '../store/useStore';
import FeatureImportanceChart from './FeatureImportanceChart';
import MLPredictionChart from './MLPredictionChart';

const PredictionsTab = () => {
  const [tickers, setTickers] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState('AAPL');
  const [predictions, setPredictions] = useState({ historical: [], future: [] });
  const [loading, setLoading] = useState(true);
  
  const { featureImportance, fetchMLData } = useStore();

  // Fetch Tickers
  useEffect(() => {
    axios.get('http://localhost:8000/api/tickers')
      .then(res => setTickers(res.data))
      .catch(err => console.error(err));
  }, []);

  const fetchPredictions = () => {
    if (!selectedTicker) return;
    
    // We also fetch feature importance using the store
    fetchMLData(selectedTicker);
    
    axios.get(`http://localhost:8000/api/predictions/${selectedTicker}?model=rf`)
      .then(res => {
        setPredictions(res.data);
        if (loading) setLoading(false);
      })
      .catch(err => {
        console.error(err);
        if (loading) setLoading(false);
      });
  };

  useEffect(() => {
    setLoading(true);
    fetchPredictions();
    
    // Auto-refresh every 10 seconds for real-time data
    const interval = setInterval(() => {
      fetchPredictions();
    }, 10000);
    
    return () => clearInterval(interval);
  }, [selectedTicker, fetchMLData]);

  const historical = predictions.historical || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Control Panel */}
      <div className="glass-card" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <h3 style={{ margin: 0, fontSize: '16px' }}>Select Ticker:</h3>
        <select 
          value={selectedTicker}
          onChange={(e) => setSelectedTicker(e.target.value)}
          style={{ 
            background: 'rgba(0,0,0,0.3)', color: '#fff', 
            border: '1px solid rgba(255,255,255,0.1)', 
            padding: '8px 16px', borderRadius: '8px', 
            outline: 'none', fontSize: '16px'
          }}
        >
          {tickers.map(t => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px', color: '#f59e0b', fontSize: '14px', background: 'rgba(245, 158, 11, 0.1)', padding: '8px 12px', borderRadius: '8px' }}>
          <AlertTriangle size={16} />
          <span>ML Predictions are for informational purposes only. Do not trade solely based on these.</span>
        </div>
      </div>

      {loading ? (
        <div className="glass-card" style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ color: '#6366f1', fontSize: '20px', fontWeight: 500, animation: 'pulse 1.5s infinite' }}>Loading ML Models...</div>
        </div>
      ) : (
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
          {/* Prediction vs Actual Chart */}
          <div style={{ flex: '1 1 100%', minWidth: '400px' }}>
            <MLPredictionChart data={historical} />
          </div>
          
          {/* Feature Importance Chart */}
          <div style={{ flex: '1 1 100%', minWidth: '400px' }}>
            <FeatureImportanceChart data={featureImportance} ticker={selectedTicker} />
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionsTab;
