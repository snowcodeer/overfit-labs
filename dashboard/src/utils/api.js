const API_BASE = 'http://localhost:8000/api';

export const fetchRuns = async () => {
  const response = await fetch(`${API_BASE}/runs`);
  if (!response.ok) throw new Error('Failed to fetch runs');
  return response.json();
};

export const fetchRunHistory = async (group, runId) => {
  const response = await fetch(`${API_BASE}/run/${group}/${runId}/history`);
  if (!response.ok) throw new Error('Failed to fetch history');
  return response.json();
};

export const fetchRunFiles = async (group, runId) => {
  const response = await fetch(`${API_BASE}/run/${group}/${runId}/files`);
  if (!response.ok) throw new Error('Failed to fetch files');
  return response.json();
};

export const getFileUrl = (path) => `http://localhost:8000/${path}`;
