import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ComposedChart, Bar, ReferenceLine
} from 'recharts';

const TechnicalIndicators = ({ data }) => {
  if (!data || data.length === 0) return null;

  // Format date for tooltip
  const formatTime = (timeStr) => {
    const d = new Date(timeStr);
    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${d.getMinutes().toString().padStart(2, '0')}`;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', width: '100%' }}>
      {/* RSI Chart */}
      <div className="glass-card" style={{ width: '100%', height: '250px' }}>
        <h3 style={{ margin: '0 0 8px 0', fontSize: '14px', fontWeight: 600, color: '#94a3b8' }}>RSI (14)</h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis dataKey="date" hide />
            <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
              labelFormatter={formatTime}
            />
            <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
            <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" opacity={0.5} />
            <Line type="monotone" dataKey="rsi" stroke="#8b5cf6" strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* MACD Chart */}
      <div className="glass-card" style={{ width: '100%', height: '250px' }}>
        <h3 style={{ margin: '0 0 8px 0', fontSize: '14px', fontWeight: 600, color: '#94a3b8' }}>MACD (12, 26, 9)</h3>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis dataKey="date" hide />
            <YAxis tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
              labelFormatter={formatTime}
            />
            <Bar dataKey="macd_hist" fill="#10b981" isAnimationActive={false}>
              {
                data.map((entry, index) => (
                  <cell key={`cell-${index}`} fill={entry.macd_hist >= 0 ? '#10b981' : '#ef4444'} />
                ))
              }
            </Bar>
            <Line type="monotone" dataKey="macd" stroke="#3b82f6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="macd_signal" stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TechnicalIndicators;
