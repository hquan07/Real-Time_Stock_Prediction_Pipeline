import React, { useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import useStore from '../store/useStore';
import FeatureImportanceChart from './FeatureImportanceChart';
import MLPredictionChart from './MLPredictionChart';
import { DataProvenance, EmptyState, ErrorState } from './QueryState';

const PredictionsTab = () => {
  const {
    tickers,
    selectedTicker,
    setSelectedTicker,
    fetchTickers,
    fetchMLData,
    predictionsData,
    featureImportance,
    mlMeta,
    mlError,
    isLoadingML,
  } = useStore();

  useEffect(() => {
    if (tickers.length === 0) fetchTickers();
  }, [fetchTickers, tickers.length]);

  useEffect(() => {
    if (selectedTicker) fetchMLData(selectedTicker);
  }, [fetchMLData, selectedTicker]);

  const historical = predictionsData.historical || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="glass-card" style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        <label htmlFor="prediction-ticker" style={{ fontSize: '16px' }}>Select Ticker:</label>
        <select id="prediction-ticker" value={selectedTicker} onChange={(event) => setSelectedTicker(event.target.value)}>
          {tickers.map((ticker) => <option key={ticker.value} value={ticker.value}>{ticker.label}</option>)}
        </select>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px', color: '#f59e0b' }}>
          <AlertTriangle size={16} />
          <span>Research use only. No BUY/SELL recommendation is published.</span>
        </div>
      </div>

      <DataProvenance meta={mlMeta} />

      {isLoadingML ? (
        <div className="glass-card" style={{ padding: '40px', textAlign: 'center', color: '#6366f1' }}>Loading validated model output…</div>
      ) : mlError ? (
        <ErrorState message={mlError} onRetry={() => fetchMLData(selectedTicker)} />
      ) : historical.length === 0 ? (
        <EmptyState message="No validated prediction report is available for this ticker." />
      ) : (
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 100%', minWidth: '320px' }}>
            <MLPredictionChart data={historical} />
          </div>
          <div style={{ flex: '1 1 100%', minWidth: '320px' }}>
            {featureImportance.length > 0 ? (
              <FeatureImportanceChart data={featureImportance} ticker={selectedTicker} />
            ) : (
              <EmptyState message="Feature importance is unavailable for this model version." />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionsTab;
