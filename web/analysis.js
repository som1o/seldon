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
  const grouped = { univariate: [], bivariate: [] };
  (charts || []).forEach((chart) => {
    if (chart.includes('/univariate/')) {
      grouped.univariate.push(chart);
    } else if (chart.includes('/bivariate/')) {
      grouped.bivariate.push(chart);
    }
  });
  return grouped;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderInline(markdown) {
  let html = escapeHtml(markdown);
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return html;
}

function splitMarkdownTableRow(line) {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '');
  return trimmed.split('|').map((cell) => cell.trim().replace(/\\\|/g, '|'));
}

function isMarkdownTableSeparator(line) {
  const cells = splitMarkdownTableRow(line);
  if (!cells.length) return false;
  return cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function renderMarkdown(markdownText) {
  const source = String(markdownText || '').replace(/\r/g, '');
  if (!source.trim()) {
    return '<div class="mono" style="color:var(--c-muted);">No content available.</div>';
  }

  const lines = source.split('\n');
  const out = [];

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) continue;

    if (trimmed.startsWith('```')) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i += 1;
      }
      out.push(`<pre style="overflow:auto; border:1px solid var(--c-border); padding:6px; background:rgba(255,255,255,0.02);"><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
      continue;
    }

    if (trimmed.startsWith('|') && i + 1 < lines.length && isMarkdownTableSeparator(lines[i + 1])) {
      const headers = splitMarkdownTableRow(line);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        rows.push(splitMarkdownTableRow(lines[i]));
        i += 1;
      }
      i -= 1;

      const thead = `<thead><tr>${headers.map((cell) => `<th>${renderInline(cell)}</th>`).join('')}</tr></thead>`;
      const tbody = `<tbody>${rows.map((row) => `<tr>${headers.map((_, idx) => `<td>${renderInline(row[idx] || '')}</td>`).join('')}</tr>`).join('')}</tbody>`;
      out.push(`<div style="overflow:auto; max-width:100%; max-height:420px; border:1px solid var(--c-border); margin:4px 0;"><table style="width:max-content; min-width:100%; border-collapse:collapse;">${thead}${tbody}</table></div>`);
      continue;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = Math.min(6, heading[1].length + 1);
      out.push(`<h${level} style="margin:6px 0 4px 0; color:var(--c-text);">${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      const items = [trimmed.slice(2)];
      while (i + 1 < lines.length) {
        const next = lines[i + 1].trim();
        if (!(next.startsWith('- ') || next.startsWith('* '))) break;
        items.push(next.slice(2));
        i += 1;
      }
      out.push(`<ul style="margin:4px 0 4px 16px; list-style:disc;">${items.map((item) => `<li>${renderInline(item)}</li>`).join('')}</ul>`);
      continue;
    }

    if (/^<[^>]+>/.test(trimmed)) {
      out.push(line);
      continue;
    }

    const paragraph = [trimmed];
    while (i + 1 < lines.length) {
      const next = lines[i + 1].trim();
      if (!next || next.startsWith('#') || next.startsWith('|') || next.startsWith('- ') || next.startsWith('* ') || next.startsWith('```')) break;
      paragraph.push(next);
      i += 1;
    }
    out.push(`<p style="margin:4px 0;">${renderInline(paragraph.join(' '))}</p>`);
  }

  const html = out.join('');
  if (window.DOMPurify?.sanitize) return window.DOMPurify.sanitize(html);
  return html;
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
    body.innerHTML = renderMarkdown(reports[key] || 'No content available.');
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
