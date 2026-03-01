export const MAX_UPLOAD_BYTES = 1024 * 1024 * 1024;

const STORAGE_KEY = 'seldon-web-state';

function loadPersistedState() {
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    return {
      currentWorkspaceId: parsed.currentWorkspaceId || null,
      currentAnalysisId: parsed.currentAnalysisId || null,
    };
  } catch {
    return {};
  }
}

function persistState(state) {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
      currentWorkspaceId: state.currentWorkspaceId || null,
      currentAnalysisId: state.currentAnalysisId || null,
    }));
  } catch {
  }
}

const persisted = loadPersistedState();

const state = {
  wsPort: null,
  currentWorkspaceId: persisted.currentWorkspaceId || null,
  currentAnalysisId: persisted.currentAnalysisId || null,
  analyses: [],
  tableRows: [],
  tableVisibleCount: 40,
  tableOffset: 0,
  tableLimit: 500,
  tableTotalRows: 0,
  tableHasMore: false,
  tableLoading: false,
  tableLoadingAll: false,
  charts: [],
  chartGroups: {
    univariate: [],
    bivariate: [],
    overall: [],
  },
  chartsVisibleCount: 12,
  reports: {
    univariate: '',
    bivariate: '',
    neural_synthesis: '',
    final_analysis: '',
    report: '',
  },
  selectedReportKey: 'report',
  inFlight: new Set(),
};

const listeners = new Set();

export function getState() {
  return state;
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function updateState(patch) {
  Object.assign(state, patch);
  persistState(state);
  listeners.forEach((listener) => listener(state));
}

export function setInFlight(key, value) {
  if (value) {
    state.inFlight.add(key);
  } else {
    state.inFlight.delete(key);
  }
  listeners.forEach((listener) => listener(state));
}

export function isInFlight(key) {
  return state.inFlight.has(key);
}