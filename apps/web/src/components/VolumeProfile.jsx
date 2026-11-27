import React, { useMemo } from 'react';
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
          Price Range: ${data.rangeStart.toFixed(2)} - ${data.rangeEnd.toFixed(2)}
        </p>
        <p style={{ margin: '4px 0', color: '#10b981', fontSize: '14px', fontWeight: 500 }}>
          Volume: {(data.volume / 1000).toFixed(1)}K
        </p>
      </div>
    );
  }
  return null;
};

const VolumeProfile = ({ data }) => {
  const profileData = useMemo(() => {
    if (!data || data.length === 0) return [];

    let minPrice = Number.MAX_VALUE;
    let maxPrice = Number.MIN_VALUE;

    // Find min and max close price
    data.forEach(d => {
      if (d.close < minPrice) minPrice = d.close;
      if (d.close > maxPrice) maxPrice = d.close;
    });

    if (minPrice === maxPrice) return [];

    const numBuckets = 12;
    const bucketSize = (maxPrice - minPrice) / numBuckets;
    
    // Initialize buckets
    const buckets = Array.from({ length: numBuckets }, (_, i) => ({
      rangeStart: minPrice + i * bucketSize,
      rangeEnd: minPrice + (i + 1) * bucketSize,
      priceLabel: `$${(minPrice + (i + 0.5) * bucketSize).toFixed(1)}`,
      volume: 0
    }));

    // Assign volume to buckets
    data.forEach(d => {
      if (!d.volume || !d.close) return;
      
      let bucketIndex = Math.floor((d.close - minPrice) / bucketSize);
      if (bucketIndex >= numBuckets) bucketIndex = numBuckets - 1;
      if (bucketIndex < 0) bucketIndex = 0;
      
      buckets[bucketIndex].volume += d.volume;
    });

    return buckets;
  }, [data]);

  if (profileData.length === 0) return null;

  return (
    <div className="glass-card" style={{ padding: '16px', height: '350px' }}>
      <h4 style={{ margin: '0 0 16px 0', fontSize: '16px', color: '#fff' }}>Volume Profile</h4>
      <ResponsiveContainer width="100%" height="90%">
        <BarChart 
          data={profileData} 
          layout="vertical"
          margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={true} vertical={false} />
          <XAxis 
            type="number" 
            hide={true} 
          />
          <YAxis 
            dataKey="priceLabel" 
            type="category" 
            stroke="#64748b" 
            tick={{ fill: '#64748b', fontSize: 11 }} 
            axisLine={false}
            tickLine={false}
            width={50}
          />
          <Tooltip content={<CustomTooltip />} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
          <Bar 
            dataKey="volume" 
            fill="#3b82f6" 
            opacity={0.6}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default VolumeProfile;
