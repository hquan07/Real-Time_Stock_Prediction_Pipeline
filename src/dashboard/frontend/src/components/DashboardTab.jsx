import React, { useEffect, useState } from 'react';
import useStore from '../store/useStore';
import StatCard from './StatCard';
import PriceChart from './PriceChart';

const DashboardTab = () => {
  const { 
    tickers, 
    selectedTicker, 
    setSelectedTicker, 
    historicalData, 
    isLoadingHistorical, 
    fetchTickers, 
    fetchHistoricalData,
    historicalPeriod
  } = useStore();

  const [realtimeData, setRealtimeData] = useState([]);

  // Initial load
  useEffect(() => {
    fetchTickers();
  }, [fetchTickers]);

  // Fetch historical data on ticker/period change
  useEffect(() => {
    if (selectedTicker) {
      fetchHistoricalData(selectedTicker, historicalPeriod);
      setRealtimeData([]); // Reset realtime append data
    }
  }, [selectedTicker, historicalPeriod, fetchHistoricalData]);

  // WebSocket for Realtime ticks
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/api/ws/stream');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'tick' && data.ticker === selectedTicker) {
        setRealtimeData(prev => {
          const newData = [...prev];
          
          // Get last known RSI/MACD from historical or previous real-time
          let lastRSI = 0;
          let lastMACD = 0;
          
          if (newData.length > 0) {
            lastRSI = newData[newData.length - 1].rsi || 0;
            lastMACD = newData[newData.length - 1].macd || 0;
          } else if (historicalData.length > 0) {
            lastRSI = historicalData[historicalData.length - 1].rsi || 0;
            lastMACD = historicalData[historicalData.length - 1].macd || 0;
          }
          
          newData.push({
            date: new Date(data.timestamp).toISOString(),
            close: data.price,
            open: data.price,
            high: data.price,
            low: data.price,
            volume: 0,
            rsi: lastRSI,
            macd: lastMACD
          });
          
          // Keep only last 100 realtime ticks in memory to avoid bloat
          if (newData.length > 100) newData.shift();
          
          return newData;
        });
      }
    };
    
    return () => {
      ws.close();
    };
  }, [selectedTicker, historicalData]);

  // Combine historical and real-time data for the chart
  const combinedData = [...historicalData, ...realtimeData];

  // Calculate stats from latest data
  const latest = combinedData.length > 0 ? combinedData[combinedData.length - 1] : null;
  const previous = combinedData.length > 1 ? combinedData[combinedData.length - 2] : null;
  
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
        <StatCard title="RSI (14d)" value={currentRSI ? currentRSI.toFixed(2) : "0.00"} change={currentRSI - 50} />
      </div>

      {/* Main Chart Area */}
      <div style={{ height: '600px' }}>
        {isLoadingHistorical && combinedData.length === 0 ? (
          <div className="glass-card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ color: '#6366f1', fontSize: '20px', fontWeight: 500, animation: 'pulse 1.5s infinite' }}>Loading Data...</div>
          </div>
        ) : (
          <PriceChart data={combinedData} ticker={selectedTicker} />
        )}
      </div>
    </div>
  );
};

export default DashboardTab;
