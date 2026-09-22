import React from 'react';
import Plot from 'react-plotly.js';

const PriceChart = ({ data, ticker }) => {
  if (!data || data.length === 0) {
    return <div style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>No data available</div>;
  }

  const dates = data.map(d => d.date);
  
  const traces = [
    {
      x: dates,
      open: data.map(d => d.open),
      high: data.map(d => d.high),
      low: data.map(d => d.low),
      close: data.map(d => d.close),
      type: 'candlestick',
      name: ticker,
      increasing: { line: { color: '#10b981' } },
      decreasing: { line: { color: '#ef4444' } },
    }
  ];

  // Add Moving Averages if they exist
  if (data[0].ma20) {
    traces.push({
      x: dates, y: data.map(d => d.ma20),
      type: 'scatter', mode: 'lines',
      name: 'MA20',
      line: { color: '#f59e0b', width: 1.5 }
    });
  }
  
  if (data[0].bb_upper && data[0].bb_lower) {
    traces.push({
      x: dates, y: data.map(d => d.bb_upper),
      type: 'scatter', mode: 'lines',
      name: 'BB Upper',
      line: { color: 'rgba(99, 102, 241, 0.5)', width: 1, dash: 'dash' }
    });
    traces.push({
      x: dates, y: data.map(d => d.bb_lower),
      type: 'scatter', mode: 'lines',
      name: 'BB Lower',
      fill: 'tonexty', fillcolor: 'rgba(99, 102, 241, 0.05)',
      line: { color: 'rgba(99, 102, 241, 0.5)', width: 1, dash: 'dash' }
    });
  }

  return (
    <div className="glass-card" style={{ width: '100%', height: '100%' }}>
      <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: 600 }}>{ticker} Price History</h3>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { t: 10, r: 10, b: 40, l: 40 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#94a3b8' },
          xaxis: { 
            gridcolor: 'rgba(255,255,255,0.05)',
            rangeslider: { visible: false }
          },
          yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
          legend: { orientation: 'h', y: 1.1, x: 0 },
          hovermode: 'x unified'
        }}
        useResizeHandler={true}
        style={{ width: '100%', height: 'calc(100% - 40px)' }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
};

export default PriceChart;
