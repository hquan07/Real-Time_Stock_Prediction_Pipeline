import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

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

  const dates = stockData.map(d => d.date);

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
          {/* RSI Chart */}
          <div className="glass-card" style={{ flex: '1 1 500px', minWidth: '400px' }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: 600 }}>RSI (Relative Strength Index)</h3>
            <div style={{ color: '#94a3b8', fontSize: '12px', marginBottom: '16px' }}>RSI &gt; 70: Overbought | RSI &lt; 30: Oversold</div>
            <Plot
              data={[
                { x: dates, y: stockData.map(d => d.rsi), type: 'scatter', mode: 'lines', name: 'RSI', line: { color: '#8b5cf6' } },
                { x: dates, y: Array(dates.length).fill(70), type: 'scatter', mode: 'lines', name: 'Overbought', line: { color: '#ef4444', dash: 'dash' } },
                { x: dates, y: Array(dates.length).fill(30), type: 'scatter', mode: 'lines', name: 'Oversold', line: { color: '#10b981', dash: 'dash' } },
              ]}
              layout={{
                autosize: true, margin: { t: 10, r: 10, b: 40, l: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8' },
                xaxis: { gridcolor: 'rgba(255,255,255,0.05)' }, yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
                showlegend: false, hovermode: 'x unified'
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '300px' }}
              config={{ displayModeBar: false }}
            />
          </div>

          {/* MACD Chart */}
          <div className="glass-card" style={{ flex: '1 1 500px', minWidth: '400px' }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: 600 }}>MACD</h3>
            <div style={{ color: '#94a3b8', fontSize: '12px', marginBottom: '16px' }}>Moving Average Convergence Divergence</div>
            <Plot
              data={[
                { x: dates, y: stockData.map(d => d.macd), type: 'scatter', mode: 'lines', name: 'MACD', line: { color: '#3b82f6' } },
                { x: dates, y: stockData.map(d => d.macd_signal), type: 'scatter', mode: 'lines', name: 'Signal', line: { color: '#f59e0b' } },
                { x: dates, y: stockData.map(d => d.macd_hist), type: 'bar', name: 'Histogram', marker: { color: stockData.map(d => d.macd_hist >= 0 ? '#10b981' : '#ef4444') } },
              ]}
              layout={{
                autosize: true, margin: { t: 10, r: 10, b: 40, l: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8' },
                xaxis: { gridcolor: 'rgba(255,255,255,0.05)' }, yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
                legend: { orientation: 'h', y: 1.1, x: 0 }, hovermode: 'x unified'
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '300px' }}
              config={{ displayModeBar: false }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default TechnicalTab;
