import React, { useState } from 'react';
import ArchitectureTab from './components/ArchitectureTab';
import DashboardTab from './components/DashboardTab';
import PortfolioTab from './components/PortfolioTab';
import TechnicalTab from './components/TechnicalTab';
import PredictionsTab from './components/PredictionsTab';
import PriceAlertsTab from './components/PriceAlertsTab';

function App() {
  const [activeTab, setActiveTab] = useState('architecture');

  return (
    <div style={{ padding: '24px', maxWidth: '1600px', margin: '0 auto' }}>
      {/* Header */}
      <div className="glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '40px', height: '40px', background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '20px' }}>
            AI
          </div>
          <h1 style={{ fontSize: '24px', fontWeight: 'bold', margin: 0, background: 'linear-gradient(to right, #fff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Real-Time Stock Prediction
          </h1>
        </div>
        
        {/* Navigation Tabs */}
        <div style={{ display: 'flex', gap: '8px', background: 'rgba(0,0,0,0.2)', padding: '6px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
          <button 
            onClick={() => setActiveTab('dashboard')}
            style={{ 
              padding: '8px 16px', borderRadius: '8px', border: 'none', cursor: 'pointer', fontWeight: 500, transition: 'all 0.2s',
              background: activeTab === 'dashboard' ? 'rgba(99, 102, 241, 0.2)' : 'transparent',
              color: activeTab === 'dashboard' ? '#6366f1' : '#94a3b8'
            }}>
            Overview
          </button>
          <button 
            onClick={() => setActiveTab('technical')}
            style={{ 
              padding: '8px 16px', borderRadius: '8px', border: 'none', cursor: 'pointer', fontWeight: 500, transition: 'all 0.2s',
              background: activeTab === 'technical' ? 'rgba(99, 102, 241, 0.2)' : 'transparent',
              color: activeTab === 'technical' ? '#6366f1' : '#94a3b8'
            }}>
            Technical
          </button>
          <button 
            onClick={() => setActiveTab('predictions')}
            style={{ 
              padding: '8px 16px', borderRadius: '8px', border: 'none', cursor: 'pointer', fontWeight: 500, transition: 'all 0.2s',
              background: activeTab === 'predictions' ? 'rgba(99, 102, 241, 0.2)' : 'transparent',
              color: activeTab === 'predictions' ? '#6366f1' : '#94a3b8'
            }}>
            ML Predictions
          </button>
          <button 
            onClick={() => setActiveTab('portfolio')}
            style={{ 
              padding: '8px 16px', borderRadius: '8px', border: 'none', cursor: 'pointer', fontWeight: 500, transition: 'all 0.2s',
              background: activeTab === 'portfolio' ? 'rgba(99, 102, 241, 0.2)' : 'transparent',
              color: activeTab === 'portfolio' ? '#6366f1' : '#94a3b8'
            }}>
            Portfolio
          </button>
          <button 
            onClick={() => setActiveTab('alerts')}
            style={{ 
              padding: '8px 16px', borderRadius: '8px', border: 'none', cursor: 'pointer', fontWeight: 500, transition: 'all 0.2s',
              background: activeTab === 'alerts' ? 'rgba(99, 102, 241, 0.2)' : 'transparent',
              color: activeTab === 'alerts' ? '#6366f1' : '#94a3b8'
            }}>
            Price Alerts
          </button>
          <button 
            onClick={() => setActiveTab('architecture')}
            style={{ 
              padding: '8px 16px', borderRadius: '8px', border: 'none', cursor: 'pointer', fontWeight: 500, transition: 'all 0.2s',
              background: activeTab === 'architecture' ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
              color: activeTab === 'architecture' ? '#8b5cf6' : '#94a3b8'
            }}>
            Architecture
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <main>
        {activeTab === 'architecture' && <ArchitectureTab />}
        {activeTab === 'dashboard' && <DashboardTab />}
        {activeTab === 'portfolio' && <PortfolioTab />}
        {activeTab === 'technical' && <TechnicalTab />}
        {activeTab === 'predictions' && <PredictionsTab />}
        {activeTab === 'alerts' && <PriceAlertsTab />}
      </main>
    </div>
  );
}

export default App;
