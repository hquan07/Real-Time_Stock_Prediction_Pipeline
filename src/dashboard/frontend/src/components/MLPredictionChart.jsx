import React from 'react';
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

const MLPredictionChart = ({ data }) => {
  if (!data || data.length === 0) return null;

  // Combine actual, predicted, and confidence intervals
  const formattedData = data.map(d => ({
    date: d.prediction_date,
    actual: d.actual_close,
    predicted: d.predicted_close,
    conf_lower: d.confidence_lower,
    conf_upper: d.confidence_upper,
    // For area chart we need [bottom, top]
    confidence_interval: [d.confidence_lower, d.confidence_upper]
  }));

  const formatTime = (timeStr) => {
    const d = new Date(timeStr);
    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${d.getMinutes().toString().padStart(2, '0')}`;
  };

  return (
    <div className="glass-card" style={{ width: '100%', height: '350px' }}>
      <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: 600 }}>ML Prediction vs Actual</h3>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={formattedData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis dataKey="date" hide />
          <YAxis domain={['auto', 'auto']} tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
          <Tooltip 
            contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
            labelFormatter={formatTime}
            formatter={(value) => [`$${parseFloat(value).toFixed(2)}`]}
          />
          <Legend wrapperStyle={{ bottom: -10 }} />
          
          {/* Confidence Interval */}
          <Area 
            type="monotone" 
            dataKey="confidence_interval" 
            fill="rgba(139, 92, 246, 0.2)" 
            stroke="none" 
            name="95% Confidence" 
            isAnimationActive={false} 
          />
          
          {/* Actual Price */}
          <Line 
            type="monotone" 
            dataKey="actual" 
            stroke="#10b981" 
            strokeWidth={2} 
            dot={false} 
            name="Actual Price" 
            isAnimationActive={false} 
          />
          
          {/* Predicted Price */}
          <Line 
            type="monotone" 
            dataKey="predicted" 
            stroke="#ec4899" 
            strokeWidth={2} 
            strokeDasharray="5 5"
            dot={false} 
            name="Predicted Price" 
            isAnimationActive={false} 
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MLPredictionChart;
