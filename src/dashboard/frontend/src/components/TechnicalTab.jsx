import React, { useState, useEffect } from 'react';
import axios from 'axios';
import TechnicalIndicators from './TechnicalIndicators';

const TechnicalTab = () => {
  const [tickers, setTickers] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState('AAPL');
  const [stockData, setStockData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch Tickers
  useEffect(() => {
    axios.get('http://localhost:8000/api/tickers')
      .then(res => setTickers(res.data))
      .catch(err => console.error(err));
  }, []);

  // Fetch Stock Data
  useEffect(() => {
    setLoading(true);
    axios.get(`http://localhost:8000/api/stock/${selectedTicker}?period=6M`)
      .then(res => {
        setStockData(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, [selectedTicker]);

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
      </div>

      {loading ? (
        <div className="glass-card" style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ color: '#6366f1', fontSize: '20px', fontWeight: 500, animation: 'pulse 1.5s infinite' }}>Loading Data...</div>
        </div>
      ) : (
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
          <TechnicalIndicators data={stockData} />
        </div>
      )}
    </div>
  );
};

export default TechnicalTab;
