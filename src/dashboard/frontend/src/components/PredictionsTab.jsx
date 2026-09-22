import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Target, TrendingUp, AlertTriangle } from 'lucide-react';

const PredictionsTab = () => {
  const [tickers, setTickers] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState('AAPL');
  const [predictions, setPredictions] = useState({ historical: [], future: [] });
  const [loading, setLoading] = useState(true);

  // Fetch Tickers
  useEffect(() => {
    axios.get('http://localhost:8000/api/tickers')
      .then(res => setTickers(res.data))
      .catch(err => console.error(err));
  }, []);

  const fetchPredictions = () => {
    if (!selectedTicker) return;
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
  }, [selectedTicker]);

  const historical = predictions.historical || [];
  const dates = historical.map(d => d.prediction_date);

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
          <div className="glass-card" style={{ flex: '1 1 100%', minWidth: '400px' }}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: '18px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Target size={20} color="#ec4899" />
              Model Accuracy: Prediction vs Actual
            </h3>
            <div style={{ color: '#94a3b8', fontSize: '12px', marginBottom: '16px' }}>RandomForest model evaluation on historical test data</div>
            
            {historical.length === 0 ? (
              <div style={{ height: '350px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>No prediction data available for {selectedTicker}</div>
            ) : (
              <Plot
                data={[
                  { x: dates, y: historical.map(d => d.actual_close), type: 'scatter', mode: 'lines+markers', name: 'Actual Price', line: { color: '#94a3b8', width: 2 }, marker: { size: 6 } },
                  { x: dates, y: historical.map(d => d.predicted_close), type: 'scatter', mode: 'lines+markers', name: 'Predicted Price', line: { color: '#ec4899', width: 2 }, marker: { size: 6 } },
                  { 
                    x: dates.concat(dates.slice().reverse()), 
                    y: historical.map(d => d.confidence_upper).concat(historical.map(d => d.confidence_lower).reverse()), 
                    type: 'scatter', fill: 'toself', fillcolor: 'rgba(236, 72, 153, 0.1)', 
                    line: { color: 'transparent' }, name: '95% Confidence Interval', showlegend: true 
                  }
                ]}
                layout={{
                  autosize: true, margin: { t: 10, r: 10, b: 40, l: 40 },
                  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8' },
                  xaxis: { gridcolor: 'rgba(255,255,255,0.05)' }, yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
                  legend: { orientation: 'h', y: 1.1, x: 0 }, hovermode: 'x unified'
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '400px' }}
                config={{ displayModeBar: false }}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionsTab;
