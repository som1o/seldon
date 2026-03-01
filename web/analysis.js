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
      btn.className = 'btn';
      btn.style.cssText = key === selectedKey
        ? 'padding:5px 11px; font-size:12px; font-family:\'Inter\',sans-serif; font-weight:500; letter-spacing:.04em; text-transform:uppercase; border:1px solid var(--c-cyan); background:var(--c-cyan-dk); color:#fff; cursor:pointer; min-height:30px;'
        : 'padding:5px 11px; font-size:12px; font-family:\'Inter\',sans-serif; font-weight:500; letter-spacing:.04em; text-transform:uppercase; border:1px solid var(--c-line); background:var(--c-raised); color:var(--c-dim); cursor:pointer; min-height:30px;';
      btn.disabled = !hasContent;
      if (!hasContent) {
        btn.style.opacity = '0.35';
        btn.style.cursor = 'not-allowed';
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
