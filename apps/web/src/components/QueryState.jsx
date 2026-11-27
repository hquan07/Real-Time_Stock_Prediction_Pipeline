import React from 'react';

export const ErrorState = ({ message, onRetry }) => (
  <div className="glass-card" role="alert" style={{ padding: '32px', textAlign: 'center' }}>
    <div style={{ color: '#fca5a5', marginBottom: '12px' }}>{message}</div>
    {onRetry && <button type="button" onClick={onRetry}>Retry</button>}
  </div>
);

export const EmptyState = ({ message }) => (
  <div className="glass-card" style={{ padding: '32px', textAlign: 'center', color: '#94a3b8' }}>
    {message}
  </div>
);

export const DataProvenance = ({ meta }) => {
  if (!meta) return null;
  return (
    <div className="glass-card" style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', fontSize: '13px', color: '#94a3b8' }}>
      <span>Source: {meta.source || 'Unavailable'}</span>
      <span>As of: {meta.as_of ? new Date(meta.as_of).toLocaleString() : 'Unavailable'}</span>
      <span>Timezone: {meta.timezone || 'UTC'} / {meta.exchange_timezone || 'Unavailable'}</span>
      <span>Market: {meta.market_status || 'unknown'}</span>
      <span style={{ color: meta.stale ? '#fca5a5' : '#86efac' }}>
        Freshness: {meta.stale === null || meta.stale === undefined ? 'Unavailable' : meta.stale ? 'Stale' : 'Current'}
      </span>
      {meta.model_version && <span>Model: {meta.model_version}</span>}
      {meta.research_only && <span style={{ color: '#fbbf24' }}>Research only</span>}
    </div>
  );
};
