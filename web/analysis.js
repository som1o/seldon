import { getAnalysis, getAnalysisResults } from './api.js';

function id(nodeId) {
  return document.getElementById(nodeId);
}

function parseAnalysisId() {
  if (window.SELDON_ANALYSIS_ID) {
    return window.SELDON_ANALYSIS_ID;
  }
  const match = window.location.pathname.match(/\/analysis\/([A-Za-z0-9_\-]+)/);
  return match ? decodeURIComponent(match[1]) : '';
}

function groupedCharts(charts) {
  const grouped = { univariate: [], bivariate: [], overall: [] };
  (charts || []).forEach((chart) => {
    if (chart.includes('/univariate/')) {
      grouped.univariate.push(chart);
    } else if (chart.includes('/bivariate/')) {
      grouped.bivariate.push(chart);
    } else if (chart.includes('/overall/')) {
      grouped.overall.push(chart);
    }
  });
  return grouped;
}

function renderChartGroup(containerId, charts, label) {
  const container = id(containerId);
  container.innerHTML = '';
  if (!charts.length) {
    const empty = document.createElement('div');
    empty.style.cssText = 'font-size:13px; color:var(--c-muted); font-family:\'JetBrains Mono\',monospace; padding:4px 0;';
    empty.textContent = `// no ${label.toLowerCase()} charts available`;
    container.appendChild(empty);
    return;
  }

  charts.forEach((chartUrl) => {
    const card = document.createElement('a');
    card.href = chartUrl;
    card.target = '_blank';
    card.rel = 'noopener noreferrer';
    card.style.cssText = 'display:block; border:1px solid var(--c-border); overflow:hidden; background:var(--c-surface);';

    const image = document.createElement('img');
    image.src = chartUrl;
    image.loading = 'lazy';
    image.alt = `${label} chart`;
    image.style.cssText = 'width:100%; height:160px; object-fit:contain; background:var(--c-surface); display:block;';

    card.appendChild(image);
    container.appendChild(card);
  });
}

function renderReport(reportHtml) {
  const title = id('activeReportTitle');
  const body = id('activeReportBody');
  title.textContent = 'Analysis Report';

  const trimmed = String(reportHtml || '').trim();
  if (!trimmed) {
    body.innerHTML = '<div class="mono" style="color:var(--c-muted);">No report content available.</div>';
    return;
  }

  const sanitized = window.DOMPurify?.sanitize ? window.DOMPurify.sanitize(trimmed) : trimmed;
  body.innerHTML = sanitized;
}

async function init() {
  const analysisId = parseAnalysisId();
  if (!analysisId) {
    id('analysisStatus').textContent = 'Missing analysis id in URL.';
    return;
  }

  const [analysis, results] = await Promise.all([
    getAnalysis(analysisId),
    getAnalysisResults(analysisId, { limit: 1000, offset: 0 }),
  ]);

  id('analysisStatus').textContent = `${analysis.id}: ${analysis.status} — ${analysis.message || 'no message'}`;

  const apiGroups = results?.chart_groups || {};
  const hasApiGroups = Array.isArray(apiGroups.univariate) || Array.isArray(apiGroups.bivariate) || Array.isArray(apiGroups.overall);
  const groups = hasApiGroups
    ? {
      univariate: Array.isArray(apiGroups.univariate) ? apiGroups.univariate : [],
      bivariate: Array.isArray(apiGroups.bivariate) ? apiGroups.bivariate : [],
      overall: Array.isArray(apiGroups.overall) ? apiGroups.overall : [],
    }
    : groupedCharts(results.charts || []);
  renderChartGroup('univariateCharts', groups.univariate, 'Univariate');
  renderChartGroup('bivariateCharts', groups.bivariate, 'Bivariate');
  renderChartGroup('overallCharts', groups.overall, 'Overall');

  const reportHtml = results?.report_html || results?.reports?.analysis || '';
  renderReport(reportHtml);
}

init().catch((error) => {
  console.error(error);
  id('analysisStatus').textContent = error.message || String(error);
});
