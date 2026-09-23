import React, { useState } from 'react';
import useStore from '../store/useStore';

const PriceAlertsTab = () => {
  const { tickers, selectedTicker, setSelectedTicker } = useStore();
  const [alerts, setAlerts] = useState([
    { id: 1, ticker: 'AAPL', condition: 'ABOVE', price: 180.00, active: true },
    { id: 2, ticker: 'MSFT', condition: 'BELOW', price: 300.00, active: false }
  ]);
  const [newAlertPrice, setNewAlertPrice] = useState('');
  const [newAlertCondition, setNewAlertCondition] = useState('ABOVE');

  const addAlert = () => {
    if (!newAlertPrice) return;
    const newAlert = {
      id: Date.now(),
      ticker: selectedTicker,
      condition: newAlertCondition,
      price: parseFloat(newAlertPrice),
      active: true
    };
    setAlerts([...alerts, newAlert]);
    setNewAlertPrice('');
  };

  const toggleAlert = (id) => {
    setAlerts(alerts.map(a => a.id === id ? { ...a, active: !a.active } : a));
  };

  const deleteAlert = (id) => {
    setAlerts(alerts.filter(a => a.id !== id));
  };

  const gridColumns = '1.5fr 2fr 1.5fr 1.5fr 100px 100px';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="glass-card" style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        <h3 style={{ margin: 0, fontSize: '16px', width: '100%' }}>Create New Alert</h3>
        
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

        <select 
          value={newAlertCondition}
          onChange={(e) => setNewAlertCondition(e.target.value)}
          style={{ 
            background: 'rgba(0,0,0,0.3)', color: '#fff', 
            border: '1px solid rgba(255,255,255,0.1)', 
            padding: '8px 16px', borderRadius: '8px', 
            outline: 'none', fontSize: '16px'
          }}
        >
          <option value="ABOVE">Goes Above</option>
          <option value="BELOW">Drops Below</option>
        </select>

        <div style={{ display: 'flex', alignItems: 'center', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)', padding: '0 16px' }}>
          <span style={{ color: '#94a3b8' }}>$</span>
          <input 
            type="number"
            value={newAlertPrice}
            onChange={(e) => setNewAlertPrice(e.target.value)}
            placeholder="Target Price"
            style={{ 
              background: 'transparent', color: '#fff', 
              border: 'none', padding: '8px', 
              outline: 'none', fontSize: '16px', width: '120px'
            }}
          />
        </div>

        <button 
          onClick={addAlert}
          style={{ 
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: '#fff',
            border: 'none', padding: '8px 24px', borderRadius: '8px',
            fontSize: '16px', fontWeight: 'bold', cursor: 'pointer',
            boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)'
          }}
        >
          Add Alert
        </button>
      </div>

      <div className="glass-card">
        <h3 style={{ margin: '0 0 16px 0', fontSize: '18px' }}>Active Alerts</h3>
        <div style={{ display: 'grid', gridTemplateColumns: gridColumns, gap: '16px', color: '#94a3b8', paddingBottom: '8px', borderBottom: '1px solid rgba(255,255,255,0.1)', marginBottom: '12px' }}>
          <div>Ticker</div>
          <div>Condition</div>
          <div>Target Price</div>
          <div>Status</div>
          <div></div>
          <div></div>
        </div>
        
        {alerts.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#64748b', padding: '24px 0' }}>No alerts configured.</div>
        ) : (
          alerts.map(alert => (
            <div key={alert.id} style={{ display: 'grid', gridTemplateColumns: gridColumns, gap: '16px', alignItems: 'center', padding: '12px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontWeight: 'bold', color: '#fff' }}>{alert.ticker}</div>
              <div style={{ color: alert.condition === 'ABOVE' ? '#10b981' : '#ef4444' }}>{alert.condition === 'ABOVE' ? '▲ Goes Above' : '▼ Drops Below'}</div>
              <div>${alert.price.toFixed(2)}</div>
              <div>
                <span style={{ 
                  padding: '4px 8px', borderRadius: '12px', fontSize: '12px', fontWeight: 'bold',
                  background: alert.active ? 'rgba(16, 185, 129, 0.2)' : 'rgba(148, 163, 184, 0.2)',
                  color: alert.active ? '#10b981' : '#94a3b8'
                }}>
                  {alert.active ? 'ACTIVE' : 'INACTIVE'}
                </span>
              </div>
              <button 
                onClick={() => toggleAlert(alert.id)}
                style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.2)', color: '#fff', padding: '4px 12px', borderRadius: '4px', cursor: 'pointer', textAlign: 'center' }}
              >
                {alert.active ? 'Pause' : 'Resume'}
              </button>
              <button 
                onClick={() => deleteAlert(alert.id)}
                style={{ background: 'rgba(239, 68, 68, 0.2)', border: 'none', color: '#ef4444', padding: '4px 12px', borderRadius: '4px', cursor: 'pointer', textAlign: 'center' }}
              >
                Delete
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default PriceAlertsTab;
