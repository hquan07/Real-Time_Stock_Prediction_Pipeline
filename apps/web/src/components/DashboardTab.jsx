import React, { useEffect, useState } from 'react';
import useStore from '../store/useStore';
import StatCard from './StatCard';
import PriceChart from './PriceChart';
import VolumeProfile from './VolumeProfile';
import { createWebSocketUrl } from '../config/api';
import { DataProvenance, EmptyState, ErrorState } from './QueryState';

const DashboardTab = () => {
  const { 
    tickers, 
    selectedTicker, 
    setSelectedTicker, 
    historicalData, 
    isLoadingHistorical, 
    isLoadingTickers,
    tickersError,
    historicalError,
    historicalMeta,
    fetchTickers, 
    fetchHistoricalData,
    historicalPeriod,
    fetchMLData
  } = useStore();

  const [liveTick, setLiveTick] = useState(null);
  const [streamState, setStreamState] = useState('connecting');

  // Initial load
  useEffect(() => {
    fetchTickers();
  }, [fetchTickers]);

  // Fetch historical data on ticker/period change
  useEffect(() => {
    if (selectedTicker) {
      fetchHistoricalData(selectedTicker, historicalPeriod);
      fetchMLData(selectedTicker);
    }
  }, [selectedTicker, historicalPeriod, fetchHistoricalData, fetchMLData]);

  // WebSocket for Realtime ticks
  useEffect(() => {
    const ws = new WebSocket(createWebSocketUrl('/api/ws/stream'));

    ws.onopen = () => setStreamState('connected');
    ws.onerror = () => setStreamState('unavailable');
    ws.onclose = () => setStreamState('disconnected');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'tick' && data.ticker === selectedTicker) {
        setLiveTick({ price: data.price, timestamp: data.timestamp });
      }
    };
    
    return () => {
      ws.close();
    };
  }, [selectedTicker]);

  // Calculate stats from latest data
  const latest = historicalData.length > 0 ? historicalData[historicalData.length - 1] : null;
  const previous = historicalData.length > 1 ? historicalData[historicalData.length - 2] : null;
  
  const currentPrice = liveTick?.price ?? latest?.close;
  const dailyChange = latest && previous ? ((latest.close - previous.close) / previous.close) * 100 : undefined;
  const currentVolume = latest?.volume;
  const currentRSI = latest?.rsi;

  const handleTickerChange = (event) => {
    setLiveTick(null);
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
          disabled={isLoadingTickers || tickers.length === 0}
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
        <span style={{ marginLeft: 'auto', color: streamState === 'connected' ? '#86efac' : '#fbbf24', fontSize: '13px' }}>
          Stream: {streamState}
        </span>
      </div>

      {tickersError && <ErrorState message={tickersError} onRetry={fetchTickers} />}
      <DataProvenance meta={historicalMeta} />

      {/* Metrics Row */}
      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        <StatCard title={liveTick ? "Live Price" : "Latest Close"} value={currentPrice !== undefined ? Number(currentPrice).toFixed(2) : 'Unavailable'} prefix={currentPrice !== undefined ? '$' : ''} change={dailyChange} />
        <StatCard title="Trading Volume" value={currentVolume !== undefined ? (Number(currentVolume) / 1000000).toFixed(2) : 'Unavailable'} suffix={currentVolume !== undefined ? 'M' : ''} />
        <StatCard title="RSI (14d)" value={currentRSI !== undefined && currentRSI !== null ? Number(currentRSI).toFixed(2) : 'Unavailable'} />
      </div>

      {isLoadingHistorical && historicalData.length === 0 ? (
        <div className="glass-card" style={{ height: '600px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ color: '#6366f1', fontSize: '20px', fontWeight: 500, animation: 'pulse 1.5s infinite' }}>Loading Data...</div>
        </div>
      ) : historicalError ? (
        <ErrorState message={historicalError} onRetry={() => fetchHistoricalData(selectedTicker, historicalPeriod)} />
      ) : historicalData.length === 0 ? (
        <EmptyState message="No validated market data is available for this ticker and period." />
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
          <div style={{ height: '600px' }}>
            <PriceChart data={historicalData} ticker={selectedTicker} />
          </div>
          <div style={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
            <VolumeProfile data={historicalData} />
          </div>
        </div>
      )}
    </div>
  );
};

export default DashboardTab;
