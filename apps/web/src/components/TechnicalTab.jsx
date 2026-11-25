import React, { useState, useEffect, useCallback } from 'react';
import TechnicalIndicators from './TechnicalIndicators';
import { apiClient } from '../config/api';
import { DataProvenance, EmptyState, ErrorState } from './QueryState';

const TechnicalTab = () => {
  const [tickers, setTickers] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState('');
  const [stockData, setStockData] = useState([]);
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch Tickers
  useEffect(() => {
    apiClient.get('/tickers')
      .then(res => {
        const options = res.data?.data || [];
        setTickers(options);
        setSelectedTicker(current => current || options[0]?.value || '');
        if (options.length === 0) setLoading(false);
      })
      .catch(err => {
        setError(err?.response?.data?.detail || 'Ticker catalog unavailable.');
        setLoading(false);
      });
  }, []);

  const fetchStockData = useCallback(async () => {
    if (!selectedTicker) return;
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.get(`/stock/${selectedTicker}`, {
        params: { period: '6M' },
      });
      setStockData(response.data?.data || []);
      setMeta(response.data?.meta || null);
    } catch (error) {
      setStockData([]);
      setMeta(null);
      setError(error?.response?.data?.detail || 'Technical data unavailable.');
    } finally {
      setLoading(false);
    }
  }, [selectedTicker]);

  // Fetch Stock Data
  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect -- synchronize remote data with ticker selection
    fetchStockData();
  }, [fetchStockData]);

  const handleTickerChange = (event) => {
    setLoading(true);
    setSelectedTicker(event.target.value);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Control Panel */}
      <div className="glass-card" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <h3 style={{ margin: 0, fontSize: '16px' }}>Select Ticker:</h3>
        <select 
          value={selectedTicker}
          onChange={handleTickerChange}
          disabled={tickers.length === 0}
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

      <DataProvenance meta={meta} />

      {loading ? (
        <div className="glass-card" style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ color: '#6366f1', fontSize: '20px', fontWeight: 500, animation: 'pulse 1.5s infinite' }}>Loading Data...</div>
        </div>
      ) : error ? (
        <ErrorState message={error} onRetry={fetchStockData} />
      ) : stockData.length === 0 ? (
        <EmptyState message="No validated technical data is available for this ticker." />
      ) : (
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
          <TechnicalIndicators data={stockData} />
        </div>
      )}
    </div>
  );
};

export default TechnicalTab;
