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
  const styles = type === 'error'
    ? 'background:rgba(204,51,51,0.12); border-color:rgba(204,51,51,0.40); color:#cc5555;'
    : type === 'success'
      ? 'background:rgba(26,158,90,0.10); border-color:rgba(26,158,90,0.40); color:#1fa864;'
      : 'background:rgba(8,10,18,0.90); border-color:rgba(255,255,255,0.10); color:#6b7485;';
  toast.setAttribute('style',
    `border:1px solid; padding:6px 11px; font-size:12px; font-weight:500;
     font-family:'Inter',sans-serif; pointer-events:none; max-width:300px;
     backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); ${styles}`);
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3500);
}

export function setWsStatus(text, mode) {
  el('wsStatus').textContent = `WS: ${text}`;
  const dot = el('wsStatusDot');
  const liveIndicator = el('wsLiveIndicator');
  const color = mode === 'connected' ? '#1a9e5a'
    : mode === 'error' ? '#cc3333'
    : '#d4820a';
  const glow = mode === 'connected' ? 'rgba(26,158,90,0.55)'
    : mode === 'error' ? 'rgba(204,51,51,0.55)'
    : 'rgba(212,130,10,0.55)';
  dot.style.background = color;
  dot.style.color = color;
  dot.style.boxShadow = `0 0 6px 1px ${glow}`;
  dot.classList.toggle('ws-live', mode === 'connected');
  if (liveIndicator) {
    liveIndicator.textContent = mode === 'connected' ? 'LIVE' : mode === 'error' ? 'ERR' : 'IDLE';
    liveIndicator.style.color = color;
    liveIndicator.style.textShadow = mode === 'connected' ? `0 0 6px ${glow}` : 'none';
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
    button.innerHTML = `<span style="display:inline-flex;align-items:center;"><span class="spin" aria-hidden="true"></span>${loadingText}</span>`;
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

export function renderAnalyses(analyses, { onOpen, onDelete, onDownload, onCancel }) {
  const rows = el('analysisRows');
  rows.innerHTML = '';

  const sorted = [...analyses].sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  sorted.forEach((analysis) => {
    const tr = document.createElement('tr');

    const idTd = document.createElement('td');
    idTd.className = 'mono';
    idTd.style.cssText = 'font-size:11px; padding:3px 8px; border-bottom:1px solid var(--c-border); border-right:1px solid var(--c-border); color:var(--c-cyan); max-width:110px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;';
    idTd.textContent = analysis.id;
    idTd.title = analysis.id;

    const statusTd = document.createElement('td');
    statusTd.style.cssText = 'font-size:12px; padding:3px 8px; border-bottom:1px solid var(--c-border); border-right:1px solid var(--c-border); color:var(--c-dim);';
    const statusLower = String(analysis.status || '').toLowerCase();
    const statusColor = statusLower === 'completed' ? '#1a9e5a'
      : statusLower === 'running' ? '#e8940a'
      : statusLower === 'canceled' ? '#cc5555'
      : statusLower === 'failed' ? '#cc3333'
      : 'var(--c-dim)';
    statusTd.innerHTML = `<span style="color:${statusColor};font-weight:500;">${analysis.status}</span>`;

    const stepTd = document.createElement('td');
    stepTd.className = 'mono';
    stepTd.style.cssText = 'font-size:11px; padding:3px 8px; border-bottom:1px solid var(--c-border); border-right:1px solid var(--c-border); color:var(--c-muted);';
    const isCompleted = statusLower === 'completed' &&
      Number(analysis.total_steps) > 0 &&
      Number(analysis.step) >= Number(analysis.total_steps);
    stepTd.textContent = isCompleted ? '—' : `${analysis.step}/${analysis.total_steps}`;

    const actionsTd = document.createElement('td');
    actionsTd.style.cssText = 'padding:2px 6px; border-bottom:1px solid var(--c-border); white-space:nowrap;';

    const mkBtn = (label, style) => {
      const b = document.createElement('button');
      b.className = 'btn';
      b.style.cssText = `padding:3px 8px; font-size:11px; margin-right:3px; ${style}`;
      b.textContent = label;
      return b;
    };

    const inspectButton = mkBtn('Inspect', 'background:rgba(255,255,255,0.03); border-color:rgba(255,255,255,0.08); color:var(--c-dim);');
    inspectButton.onclick = () => onOpen(analysis.id, false);

    const downloadButton = mkBtn('Export', 'background:rgba(26,158,90,0.10); border-color:rgba(26,158,90,0.35); color:#1a9e5a;');
    downloadButton.setAttribute('aria-label', `Download analysis ${analysis.id} bundle`);
    downloadButton.onclick = () => onDownload(analysis.id);

    const deleteButton = mkBtn('Delete', 'background:rgba(204,51,51,0.10); border-color:rgba(204,51,51,0.35); color:#cc4444;');
    deleteButton.setAttribute('aria-label', `Delete analysis ${analysis.id}`);
    deleteButton.onclick = () => onDelete(analysis.id);

    actionsTd.appendChild(inspectButton);
    actionsTd.appendChild(downloadButton);
    if (statusLower === 'running' && typeof onCancel === 'function') {
      const cancelButton = mkBtn('Cancel', 'background:rgba(204,51,51,0.10); border-color:rgba(204,51,51,0.35); color:#cc4444;');
      cancelButton.setAttribute('aria-label', `Cancel analysis ${analysis.id}`);
      cancelButton.onclick = () => onCancel(analysis.id);
      actionsTd.appendChild(cancelButton);
    }
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
  renderAnalysisSummary();
  el('resultsHeading')?.focus();
}

function renderResultStats() {
  const statsNode = el('resultStats');
  if (!statsNode) {
    return;
  }
  const state = getState();
  const reportCount = Object.values(state.reports).filter((text) => text && text.trim()).length;
  const data = [
    { label: 'Univariate', value: state.chartGroups?.univariate?.length ?? 0, sub: 'charts' },
    { label: 'Bivariate', value: state.chartGroups?.bivariate?.length ?? 0, sub: 'charts' },
    { label: 'Overall', value: state.chartGroups?.overall?.length ?? 0, sub: 'charts' },
    { label: 'Reports', value: reportCount, sub: 'generated' },
  ];

  statsNode.innerHTML = '';
  data.forEach((item) => {
    const card = document.createElement('div');
    card.setAttribute('role', 'listitem');
    card.style.cssText = 'border-right:1px solid var(--c-border); background:rgba(255,255,255,0.015); padding:7px 10px;';
    card.innerHTML = `<div style="font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--c-muted);">${item.label}</div><div class="mono" style="font-size:22px;font-weight:500;color:var(--c-cyan);line-height:1.1;margin-top:2px;text-shadow:0 0 10px rgba(232,148,10,0.45);">${item.value}</div><div style="font-size:11px;color:var(--c-muted);margin-top:1px;">${item.sub}</div>`;
    statsNode.appendChild(card);
  });
}

export function renderAnalysisSummary() {
  const emptyNode = el('analysisSummaryEmpty');
  const reportListNode = el('analysisReportList');
  const openBtn = el('openFullAnalysisBtn');
  const state = getState();

  const hasResults =
    (state.charts?.length ?? 0) > 0 ||
    Object.values(state.reports ?? {}).some((t) => t && t.trim());

  if (emptyNode) {
    emptyNode.style.display = hasResults ? 'none' : '';
  }
  if (openBtn) {
    openBtn.disabled = !state.currentAnalysisId;
    openBtn.onclick = () => openFullAnalysisPage(getState().currentAnalysisId);
  }

  if (!reportListNode) {
    return;
  }
  reportListNode.innerHTML = '';

  const reportLabels = {
    report: 'Deterministic',
    univariate: 'Univariate',
    bivariate: 'Bivariate',
    neural_synthesis: 'Neural',
    final_analysis: 'Final',
  };

  Object.entries(reportLabels).forEach(([key, label]) => {
    const hasContent = Boolean(state.reports?.[key]?.trim());
    const badge = document.createElement('span');
    badge.className = `report-badge${hasContent ? ' has-content' : ''}`;
    badge.innerHTML = `<span class="dot"></span>${label}`;
    reportListNode.appendChild(badge);
  });
}

function renderReportSwitches() {
  const switcher = el('reportSwitches');
  if (!switcher || switcher.closest('[aria-hidden]')) {
    return;
  }
  const state = getState();
  const reportLabels = {
    univariate: 'Univariate',
    bivariate: 'Bivariate',
    neural_synthesis: 'Neural',
    final_analysis: 'Final',
    report: 'Deterministic',
  };

  switcher.innerHTML = '';
  Object.entries(reportLabels).forEach(([key, label]) => {
    const btn = document.createElement('button');
    const hasContent = Boolean(state.reports[key] && state.reports[key].trim());
    const active = state.selectedReportKey === key;
    btn.className = 'btn';
    btn.style.cssText = active
      ? 'padding:4px 10px; font-size:12px; background:rgba(185,114,8,0.20); border-color:rgba(232,148,10,0.50); color:#e8940a;'
      : 'padding:4px 10px; font-size:12px; background:rgba(255,255,255,0.03); border-color:rgba(255,255,255,0.07); color:var(--c-dim);';
    btn.disabled = !hasContent;
    if (!hasContent) {
      btn.style.opacity = '0.35';
      btn.style.cursor = 'not-allowed';
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
  if (!bodyNode || bodyNode.closest('[aria-hidden]')) {
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
  if (!container || container.closest('[aria-hidden]')) {
    return;
  }
  container.innerHTML = '';
  if (!tableRows.length) {
    return;
  }

  const status = document.createElement('div');
  status.className = 'mono';
  status.style.cssText = 'font-size:12px; color:var(--c-muted); margin-bottom:5px; letter-spacing:.05em;';
  const totalLabel = tableTotalRows > 0 ? tableTotalRows : tableRows.length;
  status.textContent = `// ${tableRows.length} of ${totalLabel} rows loaded`;
  container.appendChild(status);

  const table = document.createElement('table');
  table.className = 'data-table';
  tableRows.slice(0, tableVisibleCount).forEach((row, index) => {
    const tr = document.createElement('tr');
    tr.style.background = index % 2 === 0 ? 'var(--c-surface)' : 'var(--c-bg)';
    row.forEach((cell) => {
      const td = document.createElement(index === 0 ? 'th' : 'td');
      td.textContent = String(cell ?? '');
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  container.appendChild(table);

  const controls = document.createElement('div');
  controls.style.cssText = 'margin-top:6px; display:flex; flex-wrap:wrap; gap:4px;';

  if (tableRows.length > tableVisibleCount) {
    const loadMoreButton = document.createElement('button');
    loadMoreButton.className = 'btn btn-secondary';
    loadMoreButton.textContent = `Load more rows (${tableRows.length - tableVisibleCount} remaining)`;
    loadMoreButton.onclick = () => {
      updateState({ tableVisibleCount: tableVisibleCount + 40 });
      renderTables();
    };
    controls.appendChild(loadMoreButton);
  }

  if (tableHasMore) {
    const fetchMore = document.createElement('button');
    fetchMore.className = 'btn btn-secondary';
    fetchMore.textContent = tableLoading ? 'Loading…' : 'Fetch next chunk';
    fetchMore.disabled = tableLoading || tableLoadingAll;
    fetchMore.onclick = () => tableHandlers.onLoadMore?.();
    controls.appendChild(fetchMore);

    const loadAll = document.createElement('button');
    loadAll.className = 'btn btn-primary';
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
  if (!container || container.closest('[aria-hidden]')) {
    return;
  }
  container.innerHTML = '';

  const grouped = [
    { key: 'univariate', title: 'Univariate Charts', charts: chartGroups.univariate },
    { key: 'bivariate', title: 'Bivariate Charts', charts: chartGroups.bivariate },
    { key: 'overall', title: 'Overall Charts', charts: chartGroups.overall },
  ];

  grouped.forEach((group) => {
    const wrap = document.createElement('section');
    const heading = document.createElement('h4');
    heading.style.cssText = 'font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--c-muted); margin-bottom:5px;';
    heading.textContent = group.title;
    wrap.appendChild(heading);

    const grid = document.createElement('div');
    grid.style.cssText = 'display:grid; grid-template-columns:repeat(3,1fr); gap:1px; border:1px solid var(--c-border);';

    group.charts.slice(0, chartsVisibleCount).forEach((url) => {
      const card = document.createElement('a');
      card.href = url;
      card.target = '_blank';
      card.rel = 'noopener noreferrer';
      card.style.cssText = 'display:block; border:1px solid var(--c-border); overflow:hidden; background:rgba(255,255,255,0.018);';

      const image = document.createElement('img');
      image.alt = `${group.title} chart`;
      image.loading = 'lazy';
      image.style.cssText = 'width:100%; height:150px; object-fit:contain; background:rgba(255,255,255,0.018); display:block;';
      image.dataset.src = url;
      chartObserver.observe(image);

      card.appendChild(image);
      grid.appendChild(card);
    });

    if (!group.charts.length) {
      const empty = document.createElement('div');
      empty.className = 'mono';
      empty.style.cssText = 'font-size:11px; color:var(--c-muted); padding:3px 0;';
      empty.textContent = '// no charts for this section';
      grid.appendChild(empty);
    }

    wrap.appendChild(grid);
    container.appendChild(wrap);
  });

  const allChartsCount = grouped.reduce((sum, group) => sum + group.charts.length, 0);
  if (allChartsCount > chartsVisibleCount) {
    const more = document.createElement('button');
    more.className = 'btn btn-secondary';
    more.style.cssText += 'margin-top:5px;';
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

  // Excel files are binary/ZIP formats; show an info banner instead of
  // trying to parse them as CSV (which produces garbage).
  const lowerName = file.name.toLowerCase();
  if (lowerName.endsWith('.xlsx') || lowerName.endsWith('.xls')) {
    const msg = document.createElement('p');
    msg.style.cssText = 'font-size:12px; color:var(--c-dim); margin:3px 0;';
    msg.textContent = `Excel file selected (${file.name}) — preview not available in the browser. The engine will convert it server-side before analysis.`;
    previewContainer.appendChild(msg);
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
    table.className = 'data-table';
    rows.forEach((row, rowIndex) => {
      const tr = document.createElement('tr');
      tr.style.background = rowIndex % 2 === 0 ? 'var(--c-surface)' : 'var(--c-bg)';
      row.forEach((cell) => {
        const td = document.createElement(rowIndex === 0 ? 'th' : 'td');
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