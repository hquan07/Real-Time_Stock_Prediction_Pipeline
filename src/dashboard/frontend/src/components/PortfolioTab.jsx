import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react';

const PortfolioTab = () => {
  const [portfolio, setPortfolio] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Assuming 'default_user' for now as per backend implementation
    axios.get('http://localhost:8000/api/portfolio/default_user')
      .then(res => {
        setPortfolio(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  // Calculate summary metrics
  const totalValue = portfolio.reduce((sum, pos) => sum + (pos.total_value || 0), 0);
  const totalCost = portfolio.reduce((sum, pos) => sum + (pos.total_cost || 0), 0);
  const totalPnL = totalValue - totalCost;
  const pnlPercent = totalCost > 0 ? (totalPnL / totalCost) * 100 : 0;
  const pnlColor = totalPnL >= 0 ? '#10b981' : '#ef4444';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Summary Cards */}
      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        <div className="glass-card" style={{ flex: 1, minWidth: '200px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ padding: '16px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '12px' }}>
            <DollarSign size={24} color="#6366f1" />
          </div>
          <div>
            <div style={{ color: '#94a3b8', fontSize: '14px', marginBottom: '4px' }}>Total Portfolio Value</div>
            <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#fff' }}>${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
          </div>
        </div>
        
        <div className="glass-card" style={{ flex: 1, minWidth: '200px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ padding: '16px', background: `${pnlColor}1a`, borderRadius: '12px' }}>
            <Activity size={24} color={pnlColor} />
          </div>
          <div>
            <div style={{ color: '#94a3b8', fontSize: '14px', marginBottom: '4px' }}>Unrealized P&L</div>
            <div style={{ fontSize: '28px', fontWeight: 'bold', color: pnlColor, display: 'flex', alignItems: 'center', gap: '8px' }}>
              {totalPnL >= 0 ? '+' : '-'}${Math.abs(totalPnL).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              <span style={{ fontSize: '16px', background: `${pnlColor}33`, padding: '4px 8px', borderRadius: '8px' }}>
                {pnlPercent >= 0 ? '▲' : '▼'} {Math.abs(pnlPercent).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Portfolio Table */}
      <div className="glass-card" style={{ overflowX: 'auto' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: 600 }}>Current Holdings</h3>
        
        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: '#6366f1', animation: 'pulse 1.5s infinite' }}>Loading Portfolio...</div>
        ) : portfolio.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: '#94a3b8' }}>No holdings found in portfolio.</div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                <th style={{ padding: '12px 16px', color: '#94a3b8', fontWeight: 500 }}>Asset</th>
                <th style={{ padding: '12px 16px', color: '#94a3b8', fontWeight: 500 }}>Quantity</th>
                <th style={{ padding: '12px 16px', color: '#94a3b8', fontWeight: 500 }}>Avg Price</th>
                <th style={{ padding: '12px 16px', color: '#94a3b8', fontWeight: 500 }}>Current Price</th>
                <th style={{ padding: '12px 16px', color: '#94a3b8', fontWeight: 500 }}>Total Value</th>
                <th style={{ padding: '12px 16px', color: '#94a3b8', fontWeight: 500 }}>Unrealized P&L</th>
              </tr>
            </thead>
            <tbody>
              {portfolio.map((row) => {
                const isProfit = row.unrealized_pnl >= 0;
                return (
                  <tr key={row.ticker} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', transition: 'background 0.2s' }}>
                    <td style={{ padding: '16px', fontWeight: 600 }}>{row.ticker}</td>
                    <td style={{ padding: '16px' }}>{row.quantity}</td>
                    <td style={{ padding: '16px' }}>${row.average_price.toFixed(2)}</td>
                    <td style={{ padding: '16px' }}>${row.current_price.toFixed(2)}</td>
                    <td style={{ padding: '16px', fontWeight: 500 }}>${row.total_value.toFixed(2)}</td>
                    <td style={{ padding: '16px', color: isProfit ? '#10b981' : '#ef4444', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '4px' }}>
                      {isProfit ? <TrendingUp size={16}/> : <TrendingDown size={16}/>}
                      ${Math.abs(row.unrealized_pnl).toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default PortfolioTab;
