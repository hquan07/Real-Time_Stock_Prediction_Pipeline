import React, { useState, useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
  applyNodeChanges,
  applyEdgeChanges,
  Handle,
  Position,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  Monitor,
  Server,
  TerminalSquare,
  Database,
  Activity,
  Zap,
  Globe,
  TrendingUp,
  Cpu,
  BarChart3,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

// --- Metrics Data Mock ---
const getMetricsForNode = (nodeId) => {
  const metricsMap = {
    client: { desc: 'React/Vite Dashboard', stat: 'Status: Active | Latency: 42ms' },
    api_gateway: { desc: 'FastAPI Backend', stat: 'Requests: 124/s | P99: 15ms' },
    ingestion: { desc: 'YFinance Fetcher', stat: 'Rate Limit: 98% remaining' },
    kafka: { desc: 'Apache Kafka', stat: 'Throughput: 540 Msg/s' },
    spark: { desc: 'Spark Streaming', stat: 'Micro-batch: 2.4s' },
    pg: { desc: 'PostgreSQL DB', stat: 'Connections: 12/100 | Size: 1.2GB' },
    ml: { desc: 'ML Inference Engine', stat: 'Model: RandomForest | Acc: 94%' },
  };
  return metricsMap[nodeId] || { desc: 'Unknown Component', stat: 'N/A' };
};

// --- Custom Node Component ---
const CustomNode = ({ id, data }) => {
  const [isHealthy, setIsHealthy] = useState(true);
  const [showMetrics, setShowMetrics] = useState(false);

  // Icons mapping
  const IconMap = {
    monitor: Monitor,
    globe: Globe,
    server: Server,
    terminal: TerminalSquare,
    database: Database,
    activity: Activity,
    zap: Zap,
    trending: TrendingUp,
    cpu: Cpu,
    chart: BarChart3
  };
  const Icon = IconMap[data.icon] || Server;

  const handleDoubleClick = () => setIsHealthy(!isHealthy);
  const handleClick = () => setShowMetrics(!showMetrics);

  const metrics = getMetricsForNode(id);

  return (
    <div
      onDoubleClick={handleDoubleClick}
      onClick={handleClick}
      style={{
        padding: '12px 16px',
        borderRadius: '12px',
        background: isHealthy ? 'rgba(26, 26, 46, 0.95)' : 'rgba(127, 29, 29, 0.95)',
        border: `2px solid ${isHealthy ? data.color || '#4f46e5' : '#ef4444'}`,
        color: '#fff',
        width: '220px',
        boxShadow: isHealthy ? `0 4px 15px -1px ${data.color}40` : '0 0 20px rgba(239, 68, 68, 0.8)',
        position: 'relative',
        transition: 'all 0.3s ease',
        cursor: 'pointer',
        backdropFilter: 'blur(8px)'
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: '#94a3b8', width: 8, height: 8 }} />
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
        <div style={{ padding: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '8px', display: 'flex' }}>
          <Icon size={20} color={isHealthy ? (data.color || '#fff') : '#fff'} />
        </div>
        <div>
          <div style={{ fontWeight: 600, fontSize: '14px', letterSpacing: '0.5px' }}>{data.label}</div>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>{data.sublabel}</div>
        </div>
      </div>

      {/* Health Status Indicator */}
      <div style={{
        position: 'absolute', top: '-6px', right: '-6px',
        width: '16px', height: '16px', borderRadius: '50%',
        background: isHealthy ? '#10b981' : '#ef4444',
        border: '2px solid #0f0f1a',
        boxShadow: isHealthy ? '0 0 10px #10b981' : '0 0 10px #ef4444',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        animation: isHealthy ? 'none' : 'pulse 1s infinite'
      }}>
        {isHealthy ? <CheckCircle2 size={10} color="#0f0f1a"/> : <AlertCircle size={10} color="#fff"/>}
      </div>

      {/* Metrics Tooltip (Absolute positioned) */}
      {showMetrics && (
        <div style={{
          position: 'absolute', top: '105%', left: 0, width: '100%',
          background: 'rgba(15, 15, 26, 0.95)', border: `1px solid ${data.color}`,
          borderRadius: '8px', padding: '12px', fontSize: '12px', zIndex: 50,
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)'
        }}>
          <div style={{ color: '#94a3b8', marginBottom: '6px' }}>{metrics.desc}</div>
          <div style={{ fontWeight: 'bold', color: '#6366f1' }}>{metrics.stat}</div>
        </div>
      )}

      <Handle type="source" position={Position.Bottom} style={{ background: '#94a3b8', width: 8, height: 8 }} />
    </div>
  );
};

const nodeTypes = {
  custom: CustomNode,
};

// --- Initial Data ---
const initialNodes = [
  // User/Client
  { id: 'client', type: 'custom', position: { x: 400, y: 50 }, data: { label: 'React Dashboard', sublabel: 'Web Client', icon: 'monitor', color: '#3b82f6' } },
  
  // API Layer
  { id: 'api_gateway', type: 'custom', position: { x: 400, y: 200 }, data: { label: 'FastAPI Backend', sublabel: 'REST API', icon: 'globe', color: '#8b5cf6' } },
  
  // Data Sources & Ingestion
  { id: 'ingestion', type: 'custom', position: { x: 100, y: 350 }, data: { label: 'Data Ingestion', sublabel: 'YFinance Fetcher', icon: 'trending', color: '#10b981' } },
  
  // Streaming Pipeline
  { id: 'kafka', type: 'custom', position: { x: 400, y: 350 }, data: { label: 'Apache Kafka', sublabel: 'Message Broker', icon: 'activity', color: '#f59e0b' } },
  { id: 'spark', type: 'custom', position: { x: 400, y: 500 }, data: { label: 'Spark Streaming', sublabel: 'ETL & Aggregation', icon: 'zap', color: '#f59e0b' } },
  
  // ML & Storage
  { id: 'ml', type: 'custom', position: { x: 700, y: 350 }, data: { label: 'ML Inference', sublabel: 'Predictive Engine', icon: 'cpu', color: '#ec4899' } },
  { id: 'pg', type: 'custom', position: { x: 400, y: 650 }, data: { label: 'PostgreSQL', sublabel: 'Core Database', icon: 'database', color: '#0ea5e9' } },
];

const defaultEdgeOptions = {
  animated: true,
  markerEnd: { type: MarkerType.ArrowClosed, color: '#64748b' },
  style: { strokeWidth: 2, stroke: '#64748b' },
  labelStyle: { fill: '#fff', fontWeight: 600, fontSize: 11 },
  labelBgStyle: { fill: '#1e1e2e', fillOpacity: 0.8 },
};

const initialEdges = [
  // Client to API
  { id: 'e-client-api', source: 'client', target: 'api_gateway', label: 'HTTP/REST', ...defaultEdgeOptions, style: { strokeWidth: 2, stroke: '#3b82f6' } },
  
  // API to Database (reads)
  { id: 'e-api-pg', source: 'api_gateway', target: 'pg', label: 'Query', ...defaultEdgeOptions, type: 'step', style: { strokeWidth: 2, stroke: '#8b5cf6' } },
  
  // Ingestion to Kafka
  { id: 'e-ingest-kafka', source: 'ingestion', target: 'kafka', label: 'Publish', ...defaultEdgeOptions, style: { strokeWidth: 2, stroke: '#10b981' } },
  
  // Kafka to Spark
  { id: 'e-kafka-spark', source: 'kafka', target: 'spark', label: 'Consume', ...defaultEdgeOptions, style: { strokeWidth: 3, stroke: '#f59e0b' } },
  
  // Spark to PG
  { id: 'e-spark-pg', source: 'spark', target: 'pg', label: 'Write Raw', ...defaultEdgeOptions, style: { strokeWidth: 2, stroke: '#f59e0b' } },
  
  // ML to PG
  { id: 'e-ml-pg', source: 'ml', target: 'pg', label: 'Write Predictions', ...defaultEdgeOptions, type: 'step', style: { strokeWidth: 2, stroke: '#ec4899' } },
  
  // API to ML (Inference requests)
  { id: 'e-api-ml', source: 'api_gateway', target: 'ml', label: 'Trigger', ...defaultEdgeOptions, style: { strokeWidth: 2, stroke: '#8b5cf6', strokeDasharray: '5,5' } },
];

export default function ArchitectureTab() {
  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);

  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );
  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h2 style={{ fontSize: '28px', margin: '0 0 8px 0', fontWeight: 'bold', color: '#fff' }}>System Architecture</h2>
          <p style={{ color: '#94a3b8', margin: 0, fontSize: '15px' }}>
            Real-Time Stock Prediction Pipeline Topology. 
            <span style={{ color: '#6366f1', fontWeight: 600, marginLeft: '8px' }}>Click</span> node to view metrics. 
            <span style={{ color: '#ef4444', fontWeight: 600, marginLeft: '8px' }}>Double-click</span> to toggle health status.
          </p>
        </div>
      </div>

      {/* React Flow Canvas container */}
      <div style={{ 
          height: '750px', 
          width: '100%', 
          background: '#0f0f1a', // Match Premium Dark Theme background
          borderRadius: '16px', 
          border: '1px solid rgba(255,255,255,0.1)', 
          overflow: 'hidden',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
        }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          attributionPosition="bottom-left"
          defaultEdgeOptions={defaultEdgeOptions}
        >
          <Background variant="dots" gap={24} size={1} color="rgba(255,255,255,0.05)" />
          <Controls style={{ background: 'rgba(26, 26, 46, 0.8)', color: '#fff', fill: '#fff', border: '1px solid rgba(255,255,255,0.1)', backdropFilter: 'blur(8px)' }} />
        </ReactFlow>
      </div>
      
      {/* CSS for pulse animation */}
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
          70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
          100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
      `}} />
    </div>
  );
}
