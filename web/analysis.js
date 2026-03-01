import { getAnalysis, getAnalysisResults } from './api.js';

const reportLabels = {
  univariate: 'Univariate Report',
  bivariate: 'Bivariate Report',
  neural_synthesis: 'Neural Synthesis',
  final_analysis: 'Final Analysis',
  report: 'Deterministic Report',
};

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
    } else {
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
    empty.className = 'text-xs text-slate-500';
    empty.textContent = `No ${label.toLowerCase()} charts available.`;
    container.appendChild(empty);
    return;
  }

  charts.forEach((chartUrl) => {
    const card = document.createElement('a');
    card.href = chartUrl;
    card.target = '_blank';
    card.rel = 'noopener noreferrer';
    card.className = 'block rounded-lg border border-slate-700 bg-slate-900/70 overflow-hidden';

    const image = document.createElement('img');
    image.src = chartUrl;
    image.loading = 'lazy';
    image.alt = `${label} chart`;
    image.className = 'w-full h-44 object-contain bg-slate-900';

    card.appendChild(image);
    container.appendChild(card);
  });
}

function renderReports(reports) {
  const buttons = id('reportButtons');
  const title = id('activeReportTitle');
  const body = id('activeReportBody');
  buttons.innerHTML = '';

  let selectedKey = 'report';
  if (!reports[selectedKey] || !reports[selectedKey].trim()) {
    selectedKey = Object.keys(reportLabels).find((key) => reports[key] && reports[key].trim()) || 'report';
  }

  const setActive = (key) => {
    selectedKey = key;
    title.textContent = reportLabels[key] || 'Report';
    body.textContent = reports[key] || 'No content available.';
    renderButtons();
  };

  const renderButtons = () => {
    buttons.innerHTML = '';
    Object.entries(reportLabels).forEach(([key, label]) => {
      const hasContent = Boolean(reports[key] && reports[key].trim());
      const btn = document.createElement('button');
      btn.className = key === selectedKey
        ? 'px-3 py-2 rounded-lg bg-indigo-600/80 text-sm min-h-11'
        : 'px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-sm min-h-11';
      btn.disabled = !hasContent;
      if (!hasContent) {
        btn.classList.add('opacity-40', 'cursor-not-allowed');
      }
      btn.textContent = label;
      btn.onclick = () => setActive(key);
      buttons.appendChild(btn);
    });
  };

  setActive(selectedKey);
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

  const groups = groupedCharts(results.charts || []);
  renderChartGroup('univariateCharts', groups.univariate, 'Univariate');
  renderChartGroup('bivariateCharts', groups.bivariate, 'Bivariate');
  renderChartGroup('overallCharts', groups.overall, 'Overall');

  const reports = {
    univariate: results?.reports?.univariate || '',
    bivariate: results?.reports?.bivariate || '',
    neural_synthesis: results?.reports?.neural_synthesis || '',
    final_analysis: results?.reports?.final_analysis || results?.final_markdown || '',
    report: results?.reports?.report || results?.report_markdown || '',
  };
  renderReports(reports);
}

init().catch((error) => {
  console.error(error);
  id('analysisStatus').textContent = error.message || String(error);
});
