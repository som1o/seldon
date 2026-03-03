import * as api from './api.js';
import { getState, updateState } from './state.js';
import { ProgressSocket } from './websocket.js';
import {
  clearValidationErrors,
  copyShareLink,
  el,
  getAnalysisNotes,
  getLaunchParams,
  getSanitizedText,
  getWorkspaceNotes,
  openFullAnalysisPage,
  renderAnalyses,
  renderAnalysisSummary,
  renderDatasetPreview,
  renderProgress,
  renderResults,
  renderTables,
  renderWorkspaceOptions,
  setTableHandlers,
  setAnalysisNotes,
  setButtonLoading,
  setShareLink,
  setValidationError,
  setWorkspaceNotes,
  setWsStatus,
  showToast,
  validateLaunchForm,
} from './ui.js';

let progressSocket;
let analysesRefreshTimer = null;
let liveTimerHandle = null;
const TABLE_CHUNK_SIZE = 500;

function resetSummaryMetrics() {
  updateState({
    summaryMetrics: {
      univariateCount: 0,
      bivariateCount: 0,
      overallCount: 0,
      analysisSeconds: 0,
    },
  });
  renderAnalysisSummary();
}

function normalizedPlotModes(raw) {
  const allowed = new Set(['univariate', 'bivariate', 'overall']);
  const ordered = [];
  String(raw || '')
    .split(',')
    .map((token) => token.trim().toLowerCase())
    .forEach((token) => {
      if (!allowed.has(token) || ordered.includes(token)) {
        return;
      }
      ordered.push(token);
    });
  return ordered.length ? ordered.join(',') : 'bivariate,univariate,overall';
}

function formatElapsed(totalSeconds) {
  const value = Number.isFinite(totalSeconds) ? Math.max(0, Math.floor(totalSeconds)) : 0;
  const minutes = Math.floor(value / 60);
  const seconds = value % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function parseTimestamp(value) {
  if (!value) return NaN;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : NaN;
}

function resolveTimerSource() {
  const state = getState();
  const analyses = Array.isArray(state.analyses) ? state.analyses : [];
  if (!state.currentAnalysisId) {
    return null;
  }
  return analyses.find((item) => item.id === state.currentAnalysisId) || null;
}

function updateLiveTimer() {
  const node = document.getElementById('liveTimer');
  if (!node) return;

  const item = resolveTimerSource();
  if (!item) {
    node.textContent = 'Elapsed: 00:00';
    return;
  }

  const startedAt = parseTimestamp(item.started_at);
  const finishedAt = parseTimestamp(item.finished_at);
  if (!Number.isFinite(startedAt)) {
    node.textContent = 'Elapsed: 00:00';
    return;
  }

  const end = Number.isFinite(finishedAt) ? finishedAt : Date.now();
  const elapsedSeconds = Math.max(0, Math.floor((end - startedAt) / 1000));
  node.textContent = `Elapsed: ${formatElapsed(elapsedSeconds)}`;
}

function ensureLiveTimer() {
  if (liveTimerHandle) return;
  updateLiveTimer();
  liveTimerHandle = setInterval(updateLiveTimer, 1000);
}

function getRunningAnalysisId() {
  const state = getState();
  const analyses = Array.isArray(state.analyses) ? state.analyses : [];
  const selected = analyses.find((analysis) => analysis.id === state.currentAnalysisId && analysis.status === 'running');
  if (selected) {
    return selected.id;
  }
  const firstRunning = analyses.find((analysis) => analysis.status === 'running');
  return firstRunning ? firstRunning.id : null;
}

function shouldRenderProgressForMessage(message) {
  const currentAnalysisId = getState().currentAnalysisId;
  if (!currentAnalysisId) {
    return true;
  }
  return message.analysis_id === currentAnalysisId;
}

function syncCancelButton() {
  const button = el('cancelAnalysisBtn');
  if (!button) {
    return;
  }
  const runningAnalysisId = getRunningAnalysisId();
  button.disabled = !runningAnalysisId;
  button.dataset.analysisId = runningAnalysisId || '';
}

function scheduleRefreshAnalyses() {
  if (analysesRefreshTimer) {
    return;
  }
  analysesRefreshTimer = setTimeout(async () => {
    analysesRefreshTimer = null;
    await refreshAnalyses().catch((error) => showToast(error.message, 'error'));
  }, 350);
}

async function refreshWorkspaces() {
  const out = await api.getWorkspaces();
  const workspaces = out.workspaces || [];
  const state = getState();
  let currentWorkspaceId = state.currentWorkspaceId;

  if (!currentWorkspaceId && workspaces.length) {
    currentWorkspaceId = workspaces[0].id;
  }
  if (currentWorkspaceId && !workspaces.find((workspace) => workspace.id === currentWorkspaceId)) {
    currentWorkspaceId = workspaces[0]?.id || null;
  }

  updateState({ currentWorkspaceId });
  renderWorkspaceOptions(workspaces, currentWorkspaceId);

  if (currentWorkspaceId) {
    await loadWorkspaceNotes();
    await refreshAnalyses();
  } else {
    renderAnalyses([], { onOpen: () => {}, onDelete: () => {}, onDownload: () => {}, onCancel: () => {} });
    setWorkspaceNotes('');
    resetSummaryMetrics();
    syncCancelButton();
  }
}

async function loadWorkspaceNotes() {
  const { currentWorkspaceId } = getState();
  if (!currentWorkspaceId) {
    return;
  }
  const notes = await api.getWorkspaceNotes(currentWorkspaceId);
  setWorkspaceNotes(notes.text || '');
}

async function saveWorkspaceNotes() {
  const saveButton = el('saveWorkspaceNotesBtn');
  const { currentWorkspaceId } = getState();
  if (!currentWorkspaceId) {
    setValidationError('workspaceError', 'Select a workspace before saving notes.');
    return;
  }

  setButtonLoading(saveButton, true, 'Saving…');
  try {
    await api.saveWorkspaceNotes(currentWorkspaceId, getSanitizedText(getWorkspaceNotes()));
    showToast('Workspace notes saved.', 'success');
  } finally {
    setButtonLoading(saveButton, false);
  }
}

async function refreshAnalyses() {
  const state = getState();
  if (!state.currentWorkspaceId) {
    return;
  }

  const out = await api.getWorkspaceAnalyses(state.currentWorkspaceId);
  const analyses = out.analyses || [];
  updateState({ analyses });

  if (state.currentAnalysisId && !analyses.find((analysis) => analysis.id === state.currentAnalysisId)) {
    updateState({ currentAnalysisId: null });
    resetSummaryMetrics();
  }

  renderAnalyses(analyses, {
    onOpen: (analysisId, openFullPage) => {
      if (openFullPage) {
        openFullAnalysisPage(analysisId);
        return;
      }
      selectAnalysis(analysisId).catch((error) => showToast(error.message, 'error'));
    },
    onDelete: (analysisId) => deleteAnalysis(analysisId).catch((error) => showToast(error.message, 'error')),
    onDownload: (analysisId) => downloadAnalysisBundle(analysisId),
    onCancel: (analysisId) => cancelAnalysis(analysisId).catch((error) => showToast(error.message, 'error')),
  });
  syncCancelButton();
  updateLiveTimer();
}

function downloadAnalysisBundle(analysisId) {
  if (!analysisId) {
    return;
  }
  const link = document.createElement('a');
  link.href = api.getAnalysisDownloadUrl(analysisId);
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  document.body.appendChild(link);
  link.click();
  link.remove();
}

async function refreshAnalysis(analysisId) {
  const analysis = await api.getAnalysis(analysisId);
  if (getState().currentAnalysisId !== analysisId) {
    updateLiveTimer();
    return;
  }
  renderProgress(analysis);
  if (analysis.status === 'completed') {
    await loadAnalysisResults(analysisId);
    scheduleRefreshAnalyses();
    return;
  }
  if (analysis.status === 'failed' || analysis.status === 'canceled') {
    scheduleRefreshAnalyses();
  }
    updateLiveTimer();
}

async function loadAnalysisResults(analysisId, options = {}) {
  const {
    append = false,
    offset = 0,
    limit = TABLE_CHUNK_SIZE,
  } = options;

  const results = await api.getAnalysisResults(analysisId, { limit, offset });
  if (!append) {
    renderResults(results);
    return;
  }

  const prev = getState();
  const incomingRows = Array.isArray(results.tables) ? results.tables : [];
  updateState({
    tableRows: [...prev.tableRows, ...incomingRows],
    tableOffset: Number(results.next_offset || prev.tableRows.length + incomingRows.length),
    tableLimit: Number(results.limit || limit),
    tableTotalRows: Number(results.total_table_rows || prev.tableTotalRows),
    tableHasMore: Boolean(results.has_more),
    tableLoading: false,
    tableLoadingAll: false,
  });
  renderTables();
}

async function fetchNextTableChunk() {
  const { currentAnalysisId, tableOffset, tableHasMore, tableLoading, tableLoadingAll } = getState();
  if (!currentAnalysisId || !tableHasMore || tableLoading || tableLoadingAll) {
    return;
  }
  updateState({ tableLoading: true });
  renderTables();
  try {
    await loadAnalysisResults(currentAnalysisId, {
      append: true,
      offset: tableOffset,
      limit: TABLE_CHUNK_SIZE,
    });
  } catch (error) {
    updateState({ tableLoading: false });
    renderTables();
    throw error;
  }
}

async function loadAllTableRows() {
  const { currentAnalysisId, tableLoading, tableLoadingAll } = getState();
  if (!currentAnalysisId || tableLoading || tableLoadingAll) {
    return;
  }
  updateState({ tableLoadingAll: true, tableLoading: false });
  renderTables();

  try {
    let guard = 0;
    while (getState().tableHasMore && guard < 10000) {
      guard += 1;
      await loadAnalysisResults(currentAnalysisId, {
        append: true,
        offset: getState().tableOffset,
        limit: TABLE_CHUNK_SIZE,
      });
      if (!getState().tableHasMore) {
        break;
      }
    }
    showToast('Loaded all table rows.', 'success');
  } catch (error) {
    updateState({ tableLoadingAll: false });
    renderTables();
    throw error;
  }
}

async function selectAnalysis(analysisId) {
  updateState({ currentAnalysisId: analysisId });
  renderAnalysisSummary();
  progressSocket?.subscribeToAnalysis(analysisId);
  await refreshAnalysis(analysisId);

  const notes = await api.getAnalysisNotes(analysisId);
  setAnalysisNotes(notes.text || '');
  await loadAnalysisResults(analysisId);
  updateLiveTimer();
}

async function saveAnalysisNotes() {
  const button = el('saveAnalysisNotesBtn');
  const { currentAnalysisId } = getState();
  if (!currentAnalysisId) {
    showToast('Select an analysis first.', 'error');
    return;
  }

  setButtonLoading(button, true, 'Saving…');
  try {
    await api.saveAnalysisNotes(currentAnalysisId, getSanitizedText(getAnalysisNotes()));
    showToast('Analysis notes saved.', 'success');
  } finally {
    setButtonLoading(button, false);
  }
}

async function shareAnalysis() {
  const button = el('shareAnalysisBtn');
  const { currentAnalysisId } = getState();
  if (!currentAnalysisId) {
    showToast('Select an analysis first.', 'error');
    return;
  }

  setButtonLoading(button, true, 'Creating…');
  try {
    const out = await api.createShareLink(currentAnalysisId);
    const link = `${location.origin}${out.link}`;
    setShareLink(link);
    showToast('Share link created.', 'success');
  } finally {
    setButtonLoading(button, false);
  }
}

async function deleteAnalysis(analysisId) {
  if (!window.confirm('Delete this analysis?')) {
    return;
  }
  await api.deleteAnalysis(analysisId);
  if (getState().currentAnalysisId === analysisId) {
    updateState({ currentAnalysisId: null });
    setAnalysisNotes('');
    renderResults({
      tables: [],
      charts: [],
      report_html: '',
      reports: {
        analysis: '',
      },
      summary: {
        univariate_charts: 0,
        bivariate_charts: 0,
        overall_charts: 0,
        total_graphs: 0,
        analysis_seconds: 0,
      },
    });
  }
  await refreshAnalyses();
  showToast('Analysis deleted.', 'success');
}

async function cancelAnalysis(analysisId) {
  const resolvedAnalysisId = analysisId || el('cancelAnalysisBtn')?.dataset.analysisId || getRunningAnalysisId();
  if (!resolvedAnalysisId) {
    showToast('No running analysis to cancel.', 'error');
    syncCancelButton();
    return;
  }
  if (!window.confirm('Cancel this running analysis?')) {
    return;
  }

  const cancelButton = el('cancelAnalysisBtn');
  const shouldShowLoading = cancelButton && cancelButton.dataset.analysisId === resolvedAnalysisId;
  if (shouldShowLoading) {
    setButtonLoading(cancelButton, true, 'Canceling…');
  }
  try {
    await api.cancelAnalysis(resolvedAnalysisId);
    showToast('Cancel request sent.', 'success');
    await refreshAnalyses();
    if (getState().currentAnalysisId === resolvedAnalysisId) {
      await refreshAnalysis(resolvedAnalysisId);
    }
  } finally {
    if (shouldShowLoading) {
      setButtonLoading(cancelButton, false);
    }
    syncCancelButton();
  }
}

async function deleteWorkspace() {
  const button = el('deleteWorkspaceBtn');
  const { currentWorkspaceId } = getState();
  if (!currentWorkspaceId) {
    setValidationError('workspaceError', 'Select a workspace to delete.');
    return;
  }
  if (!window.confirm('Delete this workspace and associated analyses?')) {
    return;
  }

  setButtonLoading(button, true, 'Deleting…');
  try {
    await api.deleteWorkspace(currentWorkspaceId);
    updateState({ currentWorkspaceId: null, currentAnalysisId: null });
    resetSummaryMetrics();
    await refreshWorkspaces();
    showToast('Workspace deleted.', 'success');
  } finally {
    setButtonLoading(button, false);
  }
}

async function createWorkspace() {
  const button = el('createWorkspaceBtn');
  const nameInput = el('workspaceName');

  clearValidationErrors();
  setButtonLoading(button, true, 'Creating…');
  try {
    const name = nameInput.value.trim() || 'Untitled Workspace';
    const out = await api.createWorkspace(name);
    updateState({ currentWorkspaceId: out.id });
    await refreshWorkspaces();
    showToast('Workspace created.', 'success');
  } finally {
    setButtonLoading(button, false);
  }
}

async function renameWorkspace() {
  const button = el('renameWorkspaceBtn');
  const nameInput = el('workspaceName');
  const { currentWorkspaceId } = getState();
  if (!currentWorkspaceId) {
    setValidationError('workspaceError', 'Select a workspace to rename.');
    return;
  }

  const nextName = nameInput.value.trim();
  if (!nextName) {
    setValidationError('workspaceError', 'Enter a non-empty workspace name.');
    return;
  }

  setButtonLoading(button, true, 'Renaming…');
  try {
    await api.renameWorkspace(currentWorkspaceId, nextName);
    await refreshWorkspaces();
    showToast('Workspace renamed.', 'success');
  } finally {
    setButtonLoading(button, false);
  }
}

async function uploadAndRun(event) {
  event.preventDefault();
  clearValidationErrors();

  const state = getState();
  if (!validateLaunchForm(state.currentWorkspaceId)) {
    setValidationError('uploadFormError', 'Fix validation errors before submitting.');
    return;
  }

  const submitButton = el('uploadRunBtn');
  const file = el('datasetFile').files[0];
  const params = getLaunchParams();

  setButtonLoading(submitButton, true, 'Uploading…');
  try {
    const upload = await api.uploadDataset(state.currentWorkspaceId, file);
    setButtonLoading(submitButton, true, 'Launching…');

    const run = await api.runAnalysis({
      workspace_id: state.currentWorkspaceId,
      dataset_path: upload.dataset_path,
      name: params.name,
      target: params.target,
      feature_strategy: params.feature_strategy,
      neural_strategy: params.neural_strategy,
      bivariate_strategy: params.bivariate_strategy,
      plots: normalizedPlotModes(params.plots),
      plot_univariate: true,
      plot_overall: true,
      plot_bivariate: true,
      benchmark_mode: true,
      generate_html: false,
    });

    updateState({ currentAnalysisId: run.analysis_id });
    progressSocket?.subscribeToAnalysis(run.analysis_id);
    await refreshAnalyses();
    await selectAnalysis(run.analysis_id);
    showToast('Analysis started.', 'success');
  } finally {
    setButtonLoading(submitButton, false);
  }
}

async function loadSharedView(token) {
  const out = await api.getSharedToken(token);
  document.title = `Shared: ${out.name}`;
  setWorkspaceNotes(out.notes || '');
  setAnalysisNotes(out.notes || '');
  el('progressText').textContent = `${out.analysis_id}: ${out.status}`;
}

function bindUiEvents() {
  el('createWorkspaceBtn').onclick = () => createWorkspace().catch((error) => showToast(error.message, 'error'));
  el('renameWorkspaceBtn').onclick = () => renameWorkspace().catch((error) => showToast(error.message, 'error'));
  el('deleteWorkspaceBtn').onclick = () => deleteWorkspace().catch((error) => showToast(error.message, 'error'));
  el('workspaceSelect').onchange = async (event) => {
    updateState({ currentWorkspaceId: event.target.value, currentAnalysisId: null });
    resetSummaryMetrics();
    clearValidationErrors();
    await loadWorkspaceNotes();
    await refreshAnalyses();
  };
  el('saveWorkspaceNotesBtn').onclick = () => saveWorkspaceNotes().catch((error) => showToast(error.message, 'error'));
  el('uploadForm').onsubmit = (event) => uploadAndRun(event).catch((error) => showToast(error.message, 'error'));
  el('cancelAnalysisBtn').onclick = () => cancelAnalysis().catch((error) => showToast(error.message, 'error'));
  el('saveAnalysisNotesBtn').onclick = () => saveAnalysisNotes().catch((error) => showToast(error.message, 'error'));
  el('shareAnalysisBtn').onclick = () => shareAnalysis().catch((error) => showToast(error.message, 'error'));
  el('copyShareLinkBtn').onclick = () => copyShareLink().catch((error) => showToast(error.message, 'error'));
  el('datasetFile').onchange = (event) => {
    const file = event.target.files?.[0];
    renderDatasetPreview(file).catch(() => showToast('Unable to render file preview.', 'error'));
  };

  setTableHandlers({
    onLoadMore: () => fetchNextTableChunk().catch((error) => showToast(error.message, 'error')),
    onLoadAll: () => loadAllTableRows().catch((error) => showToast(error.message, 'error')),
  });

  syncCancelButton();
}

function initWebSocket() {
  progressSocket = new ProgressSocket({
    getWsPort: () => getState().wsPort,
    getCurrentAnalysisId: () => getState().currentAnalysisId,
    onStatus: (text, mode) => setWsStatus(text, mode),
    onProgress: (message) => {
      if (shouldRenderProgressForMessage(message)) {
        renderProgress(message);
      }

      if (message.analysis_id === getState().currentAnalysisId) {
        refreshAnalysis(message.analysis_id).catch((error) => showToast(error.message, 'error'));
      }

      if (message.analysis_id === getState().currentAnalysisId &&
          message.status === 'completed' &&
          message.total_steps > 0 &&
          message.step >= message.total_steps) {
        loadAnalysisResults(message.analysis_id).catch((error) => showToast(error.message, 'error'));
      }
      scheduleRefreshAnalyses();
    },
  });
  progressSocket.start();
}

async function init() {
  bindUiEvents();
  resetSummaryMetrics();

  const config = await api.getConfig();
  updateState({ wsPort: config.ws_port });
  initWebSocket();

  if (window.SELDON_SHARE_TOKEN) {
    await loadSharedView(window.SELDON_SHARE_TOKEN);
    return;
  }

  ensureLiveTimer();
  await refreshWorkspaces();
}

init().catch((error) => {
  console.error(error);
  showToast(error.message || String(error), 'error');
});
