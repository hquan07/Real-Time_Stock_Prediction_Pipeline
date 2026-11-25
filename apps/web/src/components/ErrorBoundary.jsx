import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="glass-card" role="alert" style={{ margin: '24px', padding: '40px', textAlign: 'center' }}>
          <h2>Dashboard unavailable</h2>
          <p style={{ color: '#94a3b8' }}>An unexpected interface error occurred. No trading action was taken.</p>
          <button type="button" onClick={() => window.location.reload()}>Reload dashboard</button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
