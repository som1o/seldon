function errorFromResponse(status, body) {
  if (body && typeof body.error === 'string' && body.error.trim()) {
    return body.error;
  }
  return `HTTP ${status}`;
}

const RETRYABLE_STATUS = new Set([502, 503, 504]);

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function apiRequest(path, options = {}) {
  const method = String(options.method || 'GET').toUpperCase();
  const shouldRetry = method === 'GET' || method === 'HEAD';
  const maxRetries = shouldRetry ? 2 : 0;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (error) {
      if (attempt < maxRetries) {
        await wait(200 * (attempt + 1));
        continue;
      }
      throw new Error('Network error — please check your connection and server availability.');
    }

    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (attempt < maxRetries && RETRYABLE_STATUS.has(response.status)) {
        await wait(200 * (attempt + 1));
        continue;
      }
      throw new Error(errorFromResponse(response.status, body));
    }
    return body;
  }

  throw new Error('Request failed after retries.');
}

export function getConfig() {
  return apiRequest('/api/config');
}

export function getWorkspaces() {
  return apiRequest('/api/workspaces');
}

export function createWorkspace(name) {
  return apiRequest(`/api/workspaces?name=${encodeURIComponent(name)}`, { method: 'POST' });
}

export function deleteWorkspace(id) {
  return apiRequest(`/api/workspaces/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export function getWorkspaceNotes(id) {
  return apiRequest(`/api/workspaces/${encodeURIComponent(id)}/notes`);
}

export function saveWorkspaceNotes(id, markdown) {
  return apiRequest(`/api/workspaces/${encodeURIComponent(id)}/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: markdown,
  });
}

export function getWorkspaceAnalyses(id) {
  return apiRequest(`/api/workspaces/${encodeURIComponent(id)}/analyses`);
}

export async function uploadDataset(workspaceId, file) {
  const formData = new FormData();
  formData.append('dataset', file);
  return apiRequest(`/api/upload?workspace_id=${encodeURIComponent(workspaceId)}`, {
    method: 'POST',
    body: formData,
  });
}

export function runAnalysis(params) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null) {
      return;
    }
    query.set(key, String(value));
  });
  return apiRequest(`/api/analyses/run?${query.toString()}`, { method: 'POST' });
}

export function getAnalysis(id) {
  return apiRequest(`/api/analyses/${encodeURIComponent(id)}`);
}

export function cancelAnalysis(id) {
  return apiRequest(`/api/analyses/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
}

export function deleteAnalysis(id) {
  return apiRequest(`/api/analyses/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export function getAnalysisNotes(id) {
  return apiRequest(`/api/analyses/${encodeURIComponent(id)}/notes`);
}

export function saveAnalysisNotes(id, markdown) {
  return apiRequest(`/api/analyses/${encodeURIComponent(id)}/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: markdown,
  });
}

export function getAnalysisResults(id, options = {}) {
  const query = new URLSearchParams();
  if (typeof options.limit === 'number') {
    query.set('limit', String(options.limit));
  }
  if (typeof options.offset === 'number') {
    query.set('offset', String(options.offset));
  }
  const suffix = query.toString() ? `?${query.toString()}` : '';
  return apiRequest(`/api/analyses/${encodeURIComponent(id)}/results${suffix}`);
}

export function getAnalysisDownloadUrl(id) {
  return `/api/analyses/${encodeURIComponent(id)}/download`;
}

export function createShareLink(id) {
  return apiRequest(`/api/analyses/${encodeURIComponent(id)}/share`, { method: 'POST' });
}

export function getSharedToken(token) {
  return apiRequest(`/api/share/${encodeURIComponent(token)}`);
}