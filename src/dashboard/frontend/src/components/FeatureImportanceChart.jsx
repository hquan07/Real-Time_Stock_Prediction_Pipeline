import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div style={{
        background: 'rgba(15, 23, 42, 0.9)',
        border: '1px solid rgba(255,255,255,0.1)',
        padding: '12px',
        borderRadius: '8px',
        color: '#fff',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.5)'
      }}>
        <p style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#94a3b8' }}>
          Feature: {data.feature}
        </p>
        <p style={{ margin: '4px 0', color: '#f59e0b', fontSize: '14px', fontWeight: 500 }}>
          Importance: {(data.importance * 100).toFixed(2)}%
        </p>
      </div>
    );
  }
  return null;
};

const FeatureImportanceChart = ({ data, ticker }) => {
  if (!data || data.length === 0) {
    return (
      <div className="glass-card" style={{ padding: '16px', height: '350px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: '#94a3b8' }}>No Feature Importance data available for {ticker}</p>
      </div>
    );
  }

  return (
    <div className="glass-card" style={{ padding: '16px', height: '350px' }}>
      <h4 style={{ margin: '0 0 16px 0', fontSize: '16px', color: '#fff' }}>ML Feature Importance</h4>
      <ResponsiveContainer width="100%" height="90%">
        <BarChart 
          data={data} 
          layout="vertical"
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={true} vertical={false} />
          <XAxis 
            type="number" 
            hide={true} 
          />
          <YAxis 
            dataKey="feature" 
            type="category" 
            stroke="#64748b" 
            tick={{ fill: '#64748b', fontSize: 11 }} 
            axisLine={false}
            tickLine={false}
            width={80}
          />
          <Tooltip content={<CustomTooltip />} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
          <Bar 
            dataKey="importance" 
            fill="#f59e0b" 
            opacity={0.8}
            isAnimationActive={false}
            radius={[0, 4, 4, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FeatureImportanceChart;
