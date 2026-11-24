import axios from 'axios';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api';

export const apiClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 15_000,
});

export const createWebSocketUrl = (path) => {
  const configuredBase = import.meta.env.VITE_WS_BASE_URL;
  if (configuredBase) {
    return new URL(path, configuredBase).toString();
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
};
