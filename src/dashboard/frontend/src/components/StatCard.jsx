import React from 'react';

const StatCard = ({ title, value, change, suffix = "", prefix = "" }) => {
  const isPositive = change >= 0;
  const changeColor = isPositive ? '#10b981' : '#ef4444'; // success or danger
  
  return (
    <div className="glass-card" style={{ flex: 1, minWidth: '200px' }}>
      <div style={{ color: '#94a3b8', fontSize: '14px', marginBottom: '8px' }}>{title}</div>
      <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#fff', marginBottom: '8px' }}>
        {prefix}{value}{suffix}
      </div>
      {change !== undefined && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: changeColor, fontSize: '14px', fontWeight: 500 }}>
          {isPositive ? '▲' : '▼'} {Math.abs(change).toFixed(2)}%
        </div>
      )}
    </div>
  );
};

export default StatCard;
