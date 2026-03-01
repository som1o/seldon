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
const TABLE_CHUNK_SIZE = 500;

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
    renderAnalyses([], { onOpen: () => {}, onDelete: () => {} });
    setWorkspaceNotes('');
  }
}

async function loadWorkspaceNotes() {
  const { currentWorkspaceId } = getState();
  if (!currentWorkspaceId) {
    return;
  }
  const notes = await api.getWorkspaceNotes(currentWorkspaceId);
  setWorkspaceNotes(notes.markdown || '');
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
  });
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
    return;
  }
  renderProgress(analysis);
  if (analysis.status === 'completed') {
    await loadAnalysisResults(analysisId);
    scheduleRefreshAnalyses();
  }
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
  setAnalysisNotes(notes.markdown || '');
  await loadAnalysisResults(analysisId);
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
      report_markdown: '',
      final_markdown: '',
      reports: {
        univariate: '',
        bivariate: '',
        neural_synthesis: '',
        final_analysis: '',
        report: '',
      },
    });
  }
  await refreshAnalyses();
  showToast('Analysis deleted.', 'success');
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
      plots: params.plots,
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
  el('reportMarkdown').textContent = out.summary || '';
  el('progressText').textContent = `${out.analysis_id}: ${out.status}`;
}

function bindUiEvents() {
  el('createWorkspaceBtn').onclick = () => createWorkspace().catch((error) => showToast(error.message, 'error'));
  el('deleteWorkspaceBtn').onclick = () => deleteWorkspace().catch((error) => showToast(error.message, 'error'));
  el('workspaceSelect').onchange = async (event) => {
    updateState({ currentWorkspaceId: event.target.value, currentAnalysisId: null });
    clearValidationErrors();
    await loadWorkspaceNotes();
    await refreshAnalyses();
  };
  el('saveWorkspaceNotesBtn').onclick = () => saveWorkspaceNotes().catch((error) => showToast(error.message, 'error'));
  el('uploadForm').onsubmit = (event) => uploadAndRun(event).catch((error) => showToast(error.message, 'error'));
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
}

function initWebSocket() {
  progressSocket = new ProgressSocket({
    getWsPort: () => getState().wsPort,
    getCurrentAnalysisId: () => getState().currentAnalysisId,
    onStatus: (text, mode) => setWsStatus(text, mode),
    onProgress: (message) => {
      renderProgress(message);
      if (message.analysis_id === getState().currentAnalysisId) {
        refreshAnalysis(message.analysis_id).catch((error) => showToast(error.message, 'error'));
      }
      if (message.step >= message.total_steps) {
        loadAnalysisResults(message.analysis_id).catch((error) => showToast(error.message, 'error'));
      }
      scheduleRefreshAnalyses();
    },
  });
  progressSocket.start();
}

async function init() {
  bindUiEvents();

  const config = await api.getConfig();
  updateState({ wsPort: config.ws_port });
  initWebSocket();

  if (window.SELDON_SHARE_TOKEN) {
    await loadSharedView(window.SELDON_SHARE_TOKEN);
    return;
  }

  await refreshWorkspaces();
}

init().catch((error) => {
  console.error(error);
  showToast(error.message || String(error), 'error');
});
