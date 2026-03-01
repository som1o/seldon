import { MAX_UPLOAD_BYTES, getState, updateState } from './state.js';

const PREVIEW_READ_BYTES = 64 * 1024;
let tableHandlers = {
  onLoadMore: null,
  onLoadAll: null,
};

const chartObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) {
      return;
    }
    const image = entry.target;
    const src = image.dataset.src;
    if (src) {
      image.src = src;
      image.removeAttribute('data-src');
    }
    chartObserver.unobserve(image);
  });
}, { rootMargin: '120px' });

export function el(id) {
  return document.getElementById(id);
}

export function showToast(message, type = 'info') {
  const container = el('toastContainer');
  const toast = document.createElement('div');
  const typeClass = type === 'error'
    ? 'bg-rose-800 border-rose-600'
    : type === 'success'
      ? 'bg-emerald-800 border-emerald-600'
      : 'bg-slate-800 border-slate-600';
  toast.className = `border text-slate-100 px-3 py-2 rounded-lg shadow-lg ${typeClass}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3500);
}

export function setWsStatus(text, mode) {
  el('wsStatus').textContent = text;
  const dot = el('wsStatusDot');
  dot.classList.remove('bg-emerald-400', 'bg-rose-400', 'bg-amber-400');
  if (mode === 'connected') {
    dot.classList.add('bg-emerald-400');
  } else if (mode === 'error') {
    dot.classList.add('bg-rose-400');
  } else {
    dot.classList.add('bg-amber-400');
  }
}

export function setButtonLoading(button, loading, loadingText) {
  if (!button) {
    return;
  }
  if (loading) {
    if (!button.dataset.originalText) {
      button.dataset.originalText = button.textContent;
    }
    button.disabled = true;
    button.innerHTML = `<span class="inline-flex items-center gap-2"><span class="inline-block h-4 w-4 border-2 border-slate-200 border-t-transparent rounded-full animate-spin" aria-hidden="true"></span>${loadingText}</span>`;
    return;
  }
  button.disabled = false;
  if (button.dataset.originalText) {
    button.textContent = button.dataset.originalText;
  }
}

export function clearValidationErrors() {
  ['workspaceError', 'datasetFileError', 'targetColumnError', 'uploadFormError'].forEach((id) => {
    const node = el(id);
    node.textContent = '';
    node.classList.add('hidden');
  });
}

export function setValidationError(id, message) {
  const node = el(id);
  node.textContent = message;
  node.classList.remove('hidden');
}

export function validateLaunchForm(currentWorkspaceId) {
  const fileInput = el('datasetFile');
  const targetColumn = el('targetColumn').value;
  let valid = true;
  let firstInvalidInput = null;

  if (!currentWorkspaceId) {
    setValidationError('workspaceError', 'Select or create a workspace before launching analysis.');
    valid = false;
    firstInvalidInput = firstInvalidInput || el('workspaceSelect');
  }

  if (!fileInput.files || !fileInput.files.length) {
    setValidationError('datasetFileError', 'Choose a dataset file.');
    valid = false;
    firstInvalidInput = firstInvalidInput || fileInput;
  } else if (fileInput.files[0].size > MAX_UPLOAD_BYTES) {
    setValidationError('datasetFileError', 'File exceeds the 1GB upload limit.');
    valid = false;
    firstInvalidInput = firstInvalidInput || fileInput;
  }

  if (targetColumn && !targetColumn.trim()) {
    setValidationError('targetColumnError', 'Target column cannot be only whitespace.');
    valid = false;
    firstInvalidInput = firstInvalidInput || el('targetColumn');
  }

  if (!valid && firstInvalidInput?.focus) {
    firstInvalidInput.focus();
  }

  return valid;
}

export function setTableHandlers(handlers = {}) {
  tableHandlers = {
    ...tableHandlers,
    ...handlers,
  };
}

export function renderWorkspaceOptions(workspaces, currentWorkspaceId) {
  const select = el('workspaceSelect');
  select.innerHTML = '';
  workspaces.forEach((workspace) => {
    const option = document.createElement('option');
    option.value = workspace.id;
    option.textContent = `${workspace.name} (${workspace.id})`;
    select.appendChild(option);
  });
  if (currentWorkspaceId) {
    select.value = currentWorkspaceId;
  }
}

export function renderAnalyses(analyses, { onOpen, onDelete, onDownload }) {
  const rows = el('analysisRows');
  rows.innerHTML = '';

  const sorted = [...analyses].sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  sorted.forEach((analysis) => {
    const tr = document.createElement('tr');
    tr.className = 'border-t border-slate-800';

    const idTd = document.createElement('td');
    idTd.className = 'py-2 pr-2 font-mono text-xs';
    idTd.textContent = analysis.id;

    const statusTd = document.createElement('td');
    statusTd.textContent = analysis.status;

    const stepTd = document.createElement('td');
    const isCompleted = String(analysis.status || '').toLowerCase() === 'completed' &&
      Number(analysis.total_steps) > 0 &&
      Number(analysis.step) >= Number(analysis.total_steps);
    stepTd.textContent = isCompleted ? '—' : `${analysis.step}/${analysis.total_steps}`;

    const actionsTd = document.createElement('td');
    actionsTd.className = 'flex flex-wrap gap-2 py-1';

    const inspectButton = document.createElement('button');
    inspectButton.className = 'px-2 py-1 rounded-lg bg-slate-700 hover:bg-slate-600 text-xs min-h-11';
    inspectButton.textContent = 'Inspect';
    inspectButton.onclick = () => onOpen(analysis.id, false);

    const openButton = document.createElement('button');
    openButton.className = 'px-2 py-1 rounded-lg bg-indigo-600/80 hover:bg-indigo-500/80 text-xs min-h-11';
    openButton.textContent = 'Open';
    openButton.onclick = () => onOpen(analysis.id, true);

    const deleteButton = document.createElement('button');
    deleteButton.className = 'px-2 py-1 rounded-lg bg-rose-700/90 hover:bg-rose-600 text-xs min-h-11';
    deleteButton.textContent = 'Delete';
    deleteButton.setAttribute('aria-label', `Delete analysis ${analysis.id}`);
    deleteButton.onclick = () => onDelete(analysis.id);

    const downloadButton = document.createElement('button');
    downloadButton.className = 'px-2 py-1 rounded-lg bg-emerald-700/90 hover:bg-emerald-600 text-xs min-h-11';
    downloadButton.textContent = 'Download';
    downloadButton.setAttribute('aria-label', `Download analysis ${analysis.id} bundle`);
    downloadButton.onclick = () => onDownload(analysis.id);

    actionsTd.appendChild(inspectButton);
    actionsTd.appendChild(openButton);
    actionsTd.appendChild(downloadButton);
    actionsTd.appendChild(deleteButton);

    tr.appendChild(idTd);
    tr.appendChild(statusTd);
    tr.appendChild(stepTd);
    tr.appendChild(actionsTd);
    rows.appendChild(tr);
  });
}

export function renderProgress(progress) {
  const percent = progress.total_steps > 0 ? Math.round((progress.step / progress.total_steps) * 100) : 0;
  el('progressBar').style.width = `${percent}%`;
  el('progressText').textContent = `${progress.id || progress.analysis_id}: ${progress.message || 'Running'} (${progress.step}/${progress.total_steps})`;
}

export function renderResults(results) {
  const reports = {
    univariate: results?.reports?.univariate || '',
    bivariate: results?.reports?.bivariate || '',
    neural_synthesis: results?.reports?.neural_synthesis || '',
    final_analysis: results?.reports?.final_analysis || results?.final_markdown || '',
    report: results?.reports?.report || results?.report_markdown || '',
  };

  const grouped = {
    univariate: [],
    bivariate: [],
    overall: [],
  };
  const charts = Array.isArray(results?.charts) ? results.charts : [];
  charts.forEach((chartUrl) => {
    if (chartUrl.includes('/univariate/')) {
      grouped.univariate.push(chartUrl);
    } else if (chartUrl.includes('/bivariate/')) {
      grouped.bivariate.push(chartUrl);
    } else {
      grouped.overall.push(chartUrl);
    }
  });

  const prevKey = getState().selectedReportKey;
  const selectedReportKey = reports[prevKey] ? prevKey : 'report';

  updateState({
    tableRows: Array.isArray(results.tables) ? results.tables : [],
    tableVisibleCount: 40,
    tableOffset: Number(results.next_offset || 0),
    tableLimit: Number(results.limit || 500),
    tableTotalRows: Number(results.total_table_rows || (Array.isArray(results.tables) ? results.tables.length : 0)),
    tableHasMore: Boolean(results.has_more),
    tableLoading: false,
    tableLoadingAll: false,
    charts,
    chartGroups: grouped,
    chartsVisibleCount: 12,
    reports,
    selectedReportKey,
  });

  renderResultStats();
  renderReportSwitches();
  renderTables();
  renderCharts();
  renderSelectedReport();
  el('resultsHeading').focus();
}

function renderResultStats() {
  const statsNode = el('resultStats');
  if (!statsNode) {
    return;
  }
  const state = getState();
  const reportCount = Object.values(state.reports).filter((text) => text && text.trim()).length;
  const data = [
    { label: 'Stat Rows', value: state.tableRows.length },
    { label: 'Charts', value: state.charts.length },
    { label: 'Reports', value: reportCount },
  ];

  statsNode.innerHTML = '';
  data.forEach((item) => {
    const card = document.createElement('div');
    card.className = 'rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-2';
    card.innerHTML = `<div class="text-xs text-slate-400">${item.label}</div><div class="text-lg font-semibold text-slate-100">${item.value}</div>`;
    statsNode.appendChild(card);
  });
}

function renderReportSwitches() {
  const switcher = el('reportSwitches');
  if (!switcher) {
    return;
  }
  const state = getState();
  const reportLabels = {
    univariate: 'Univariate',
    bivariate: 'Bivariate',
    neural_synthesis: 'Neural Synthesis',
    final_analysis: 'Final Analysis',
    report: 'Deterministic Report',
  };

  switcher.innerHTML = '';
  Object.entries(reportLabels).forEach(([key, label]) => {
    const btn = document.createElement('button');
    const hasContent = Boolean(state.reports[key] && state.reports[key].trim());
    const active = state.selectedReportKey === key;
    btn.className = active
      ? 'px-3 py-2 rounded-lg text-sm bg-indigo-600/80 text-slate-50 min-h-11'
      : 'px-3 py-2 rounded-lg text-sm bg-slate-700/90 hover:bg-slate-600 text-slate-200 min-h-11';
    btn.disabled = !hasContent;
    if (!hasContent) {
      btn.classList.add('opacity-40', 'cursor-not-allowed');
    }
    btn.textContent = label;
    btn.onclick = () => {
      updateState({ selectedReportKey: key });
      renderReportSwitches();
      renderSelectedReport();
    };
    switcher.appendChild(btn);
  });
}

function renderSelectedReport() {
  const { reports, selectedReportKey } = getState();
  const titleNode = el('reportTitle');
  const bodyNode = el('reportMarkdown');
  if (!bodyNode) {
    return;
  }

  const labels = {
    univariate: 'Univariate Report',
    bivariate: 'Bivariate Report',
    neural_synthesis: 'Neural Synthesis',
    final_analysis: 'Final Analysis',
    report: 'Deterministic Report',
  };

  if (titleNode) {
    titleNode.textContent = labels[selectedReportKey] || 'Report';
  }
  bodyNode.textContent = reports[selectedReportKey] || 'No report content available yet.';
}

export function renderTables() {
  const {
    tableRows,
    tableVisibleCount,
    tableHasMore,
    tableTotalRows,
    tableLoading,
    tableLoadingAll,
  } = getState();
  const container = el('tablesContainer');
  container.innerHTML = '';
  if (!tableRows.length) {
    return;
  }

  const status = document.createElement('div');
  status.className = 'text-xs text-slate-400 mb-2';
  const totalLabel = tableTotalRows > 0 ? tableTotalRows : tableRows.length;
  status.textContent = `Loaded ${tableRows.length} of ${totalLabel} rows`;
  container.appendChild(status);

  const table = document.createElement('table');
  table.className = 'w-full text-xs border border-slate-700';
  tableRows.slice(0, tableVisibleCount).forEach((row, index) => {
    const tr = document.createElement('tr');
    tr.className = index % 2 === 0 ? 'bg-slate-950' : 'bg-slate-900';
    row.forEach((cell) => {
      const td = document.createElement(index === 0 ? 'th' : 'td');
      td.className = 'px-2 py-1 border border-slate-800 text-left';
      td.textContent = String(cell ?? '');
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  container.appendChild(table);

  const controls = document.createElement('div');
  controls.className = 'mt-3 flex flex-wrap gap-2';

  if (tableRows.length > tableVisibleCount) {
    const loadMoreButton = document.createElement('button');
    loadMoreButton.className = 'mt-3 px-3 py-2 rounded bg-slate-700 hover:bg-slate-600 min-h-11';
    loadMoreButton.textContent = `Load more rows (${tableRows.length - tableVisibleCount} remaining)`;
    loadMoreButton.onclick = () => {
      updateState({ tableVisibleCount: tableVisibleCount + 40 });
      renderTables();
    };
    controls.appendChild(loadMoreButton);
  }

  if (tableHasMore) {
    const fetchMore = document.createElement('button');
    fetchMore.className = 'px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 min-h-11 disabled:opacity-60';
    fetchMore.textContent = tableLoading ? 'Loading…' : 'Fetch next chunk';
    fetchMore.disabled = tableLoading || tableLoadingAll;
    fetchMore.onclick = () => tableHandlers.onLoadMore?.();
    controls.appendChild(fetchMore);

    const loadAll = document.createElement('button');
    loadAll.className = 'px-3 py-2 rounded-lg bg-indigo-600/80 hover:bg-indigo-500/80 min-h-11 disabled:opacity-60';
    loadAll.textContent = tableLoadingAll ? 'Loading all…' : 'Load all rows';
    loadAll.disabled = tableLoading || tableLoadingAll;
    loadAll.onclick = () => tableHandlers.onLoadAll?.();
    controls.appendChild(loadAll);
  }

  if (controls.children.length) {
    container.appendChild(controls);
  }
}

export function renderCharts() {
  const { chartGroups, chartsVisibleCount } = getState();
  const container = el('chartsContainer');
  container.innerHTML = '';

  const grouped = [
    { key: 'univariate', title: 'Univariate Charts', charts: chartGroups.univariate },
    { key: 'bivariate', title: 'Bivariate Charts', charts: chartGroups.bivariate },
    { key: 'overall', title: 'Overall Charts', charts: chartGroups.overall },
  ];

  grouped.forEach((group) => {
    const wrap = document.createElement('section');
    wrap.className = 'space-y-2';
    const heading = document.createElement('h4');
    heading.className = 'text-sm text-slate-300 font-medium';
    heading.textContent = group.title;
    wrap.appendChild(heading);

    const grid = document.createElement('div');
    grid.className = 'grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-2';

    group.charts.slice(0, chartsVisibleCount).forEach((url) => {
      const card = document.createElement('a');
      card.href = url;
      card.target = '_blank';
      card.rel = 'noopener noreferrer';
      card.className = 'block border border-slate-700 rounded-lg overflow-hidden bg-slate-900/80';

      const image = document.createElement('img');
      image.alt = `${group.title} chart`;
      image.loading = 'lazy';
      image.className = 'w-full h-44 object-contain bg-slate-900';
      image.dataset.src = url;
      chartObserver.observe(image);

      card.appendChild(image);
      grid.appendChild(card);
    });

    if (!group.charts.length) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-slate-500';
      empty.textContent = 'No charts found for this section.';
      grid.appendChild(empty);
    }

    wrap.appendChild(grid);
    container.appendChild(wrap);
  });

  const allChartsCount = grouped.reduce((sum, group) => sum + group.charts.length, 0);
  if (allChartsCount > chartsVisibleCount) {
    const more = document.createElement('button');
    more.className = 'px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 min-h-11';
    more.textContent = `Load more charts (${allChartsCount - chartsVisibleCount} remaining)`;
    more.onclick = () => {
      updateState({ chartsVisibleCount: chartsVisibleCount + 12 });
      renderCharts();
    };
    container.appendChild(more);
  }
}

function parseCsvPreview(text, maxRows = 6) {
  return text
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .slice(0, maxRows)
    .map((line) => line.split(','));
}

export async function renderDatasetPreview(file) {
  const previewContainer = el('datasetPreviewContainer');
  previewContainer.innerHTML = '';
  if (!file) {
    return;
  }

  try {
    const text = await file.slice(0, PREVIEW_READ_BYTES).text();
    const rows = parseCsvPreview(text);
    if (!rows.length) {
      previewContainer.textContent = 'No preview available for this file.';
      return;
    }

    const table = document.createElement('table');
    table.className = 'w-full text-xs border border-slate-700';
    rows.forEach((row, rowIndex) => {
      const tr = document.createElement('tr');
      tr.className = rowIndex % 2 === 0 ? 'bg-slate-950' : 'bg-slate-900';
      row.forEach((cell) => {
        const td = document.createElement(rowIndex === 0 ? 'th' : 'td');
        td.className = 'px-2 py-1 border border-slate-800 text-left';
        td.textContent = cell;
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    previewContainer.appendChild(table);
  } catch {
    previewContainer.textContent = 'Preview unavailable for this file type.';
  }
}

export async function copyShareLink() {
  const link = el('shareLink').value;
  if (!link) {
    return;
  }
  await navigator.clipboard?.writeText(link);
  const hint = el('copyHint');
  hint.classList.remove('hidden');
  setTimeout(() => hint.classList.add('hidden'), 1200);
}

export function setShareLink(link) {
  el('shareLink').value = link;
}

export function setWorkspaceNotes(markdown) {
  el('workspaceNotes').value = markdown || '';
}

export function setAnalysisNotes(markdown) {
  el('analysisNotes').value = markdown || '';
}

export function getWorkspaceNotes() {
  return el('workspaceNotes').value;
}

export function getAnalysisNotes() {
  return el('analysisNotes').value;
}

export function getLaunchParams() {
  return {
    name: el('analysisName').value.trim() || 'Web Analysis',
    target: el('targetColumn').value.trim(),
    feature_strategy: el('featureStrategy').value,
    neural_strategy: el('neuralStrategy').value,
    bivariate_strategy: el('bivariateStrategy').value,
    plots: el('plots').value,
  };
}

export function getSanitizedText(value) {
  if (window.DOMPurify?.sanitize) {
    return window.DOMPurify.sanitize(value, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
  }
  return value;
}

export function openFullAnalysisPage(analysisId) {
  if (!analysisId) {
    return;
  }
  window.open(`/analysis/${encodeURIComponent(analysisId)}`, '_blank', 'noopener,noreferrer');
}