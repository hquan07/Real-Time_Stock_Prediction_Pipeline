import React, { useState, useEffect } from 'react';
import axios from 'axios';
import StatCard from './StatCard';
import PriceChart from './PriceChart';

const DashboardTab = () => {
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
  const fetchStockData = () => {
    if (!selectedTicker) return;
    axios.get(`http://localhost:8000/api/stock/${selectedTicker}?period=6M`)
      .then(res => {
        setStockData(res.data);
        if (loading) setLoading(false);
      })
      .catch(err => {
        console.error(err);
        if (loading) setLoading(false);
      });
  };

  useEffect(() => {
    setLoading(true);
    fetchStockData();
    
    // Auto-refresh every 10 seconds for real-time data
    const interval = setInterval(() => {
      fetchStockData();
    }, 10000);
    
    return () => clearInterval(interval);
  }, [selectedTicker]);

  // Calculate stats from latest data
  const latest = stockData.length > 0 ? stockData[stockData.length - 1] : null;
  const previous = stockData.length > 1 ? stockData[stockData.length - 2] : null;
  
  const currentPrice = latest ? latest.close : 0;
  const dailyChange = latest && previous ? ((latest.close - previous.close) / previous.close) * 100 : 0;
  const currentVolume = latest ? latest.volume : 0;
  const currentRSI = latest ? latest.rsi : 0;

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

      {/* Metrics Row */}
      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        <StatCard title="Current Price" value={currentPrice.toFixed(2)} prefix="$" change={dailyChange} />
        <StatCard title="Trading Volume" value={(currentVolume / 1000000).toFixed(2)} suffix="M" />
        <StatCard title="RSI (14d)" value={currentRSI.toFixed(2)} change={currentRSI - 50} />
      </div>

      {/* Main Chart Area */}
      <div style={{ height: '600px' }}>
        {loading ? (
          <div className="glass-card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ color: '#6366f1', fontSize: '20px', fontWeight: 500, animation: 'pulse 1.5s infinite' }}>Loading Data...</div>
          </div>
        ) : (
          <PriceChart data={stockData} ticker={selectedTicker} />
        )}
      </div>
    </div>
  );
};

export default DashboardTab;
