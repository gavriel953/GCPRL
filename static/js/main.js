/**
 * GCPRL Medical Image Enhancement — Main Application JavaScript
 */

'use strict';

// ─────────────────────── State ───────────────────────
const App = {
  jobId: null,
  originalB64: null,
  enhancedB64: null,
  slider: null,
  theme: localStorage.getItem('gcprl-theme') || 'dark',
  lightboxImg: null,
  lightboxFilename: null,
};

// ─────────────────────── DOM refs ───────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ─────────────────────── Init ───────────────────────
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(App.theme);
  initUploadZone();
  initSliders();
  initButtons();
  initLightbox();
});

// ─────────────────────── Theme ───────────────────────
function applyTheme(t) {
  document.body.classList.toggle('light-mode', t === 'light');
  const btn = $('#themeToggle');
  if (btn) btn.textContent = t === 'dark' ? '☀ Light Mode' : '☾ Dark Mode';
  App.theme = t;
  localStorage.setItem('gcprl-theme', t);
}

window.toggleTheme = () => applyTheme(App.theme === 'dark' ? 'light' : 'dark');

// ─────────────────────── Upload Zone ───────────────────────
function initUploadZone() {
  const zone = $('#uploadZone');
  const input = $('#fileInput');
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', () => { if (input.files[0]) handleFile(input.files[0]); });
}

async function handleFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  showProgress('Uploading image…');
  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');

    App.jobId = data.job_id;
    App.originalB64 = data.original_image;

    // Show original image
    setImage('originalImg', data.original_image);
    showMeta(data.metadata);
    drawHistogram('histCanvas', data.histogram, null);

    // Show upload section, hide placeholder
    $('#uploadSection').classList.add('hidden');
    $('#workspaceSection').classList.remove('hidden');

    // Reset enhanced panel
    clearPanel('enhancedImg', 'enhancedPlaceholder');
    resetComparisons();

    // Enable enhance + auto-optimize buttons
    $('#enhanceBtn').disabled = false;
    $('#autoOptBtn').disabled = false;
    $('#enhanceBtn').focus();

    toast('Image uploaded successfully', 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    hideProgress();
  }
}

// ─────────────────────── Enhancement ───────────────────────
window.enhanceImage = async () => {
  if (!App.jobId) return;

  const k = parseFloat($('#kSlider').value);
  const winSize = parseInt($('#windowSlider').value);
  const preserve = $('#preserveCheck').checked;
  const localAlpha = parseFloat($('#alphaSlider')?.value || 0.45);
  const stretch = parseFloat($('#stretchSlider')?.value || 0.95);
  const brightness = parseFloat($('#brightnessSlider')?.value || 0);

  showProgress('Running GCPRL enhancement…');

  try {
    const res = await fetch('/enhance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        job_id: App.jobId, k, window_size: winSize,
        preserve_diagnostic: preserve, local_alpha: localAlpha, 
        stretch, brightness 
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Enhancement failed');

    App.enhancedB64 = data.enhanced_image;
    App.enhancedFilename = data.enhanced_filename;

    setImage('enhancedImg', data.enhanced_image);
    showMetrics(data.metrics);
    drawHistogram('histCanvas', data.histogram, data.histogram);

    // Enable comparison buttons
    $$('.needs-enhanced').forEach(b => b.disabled = false);

    // Build slider
    buildSlider();

    toast('GCPRL enhancement complete!', 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    hideProgress();
  }
};

// ─────────────────────── Compare Standard ───────────────────────
window.compareStandard = async () => {
  if (!App.jobId || !App.enhancedB64) return;

  showProgress('Processing HE, CLAHE, Min-Max…');
  try {
    const res = await fetch('/compare_standard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: App.jobId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    buildComparisonGrid(data.images, data.metrics);
    $('#comparisonSection').classList.add('active', 'fade-in');
    $('#comparisonSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    toast('Method comparison ready', 'info');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    hideProgress();
  }
};

function buildComparisonGrid(images, metrics) {
  const grid = $('#comparisonGrid');
  if (!grid) return;

  const methods = [
    { key: 'original', label: 'Original', badgeClass: 'badge-orig' },
    { key: 'gcprl',    label: 'GCPRL',    badgeClass: 'badge-gcprl' },
    { key: 'he',       label: 'Hist. EQ', badgeClass: 'badge-he' },
    { key: 'clahe',    label: 'CLAHE',    badgeClass: 'badge-clahe' },
    { key: 'minmax',   label: 'Min-Max',  badgeClass: 'badge-minmax' },
  ];

  grid.innerHTML = methods.map(m => `
    <div class="compare-card" onclick="openLightbox('${images[m.key]}', '${m.key}.png')">
      <div class="compare-card-header">
        <span class="method-name">${m.label}</span>
        <span class="method-badge ${m.badgeClass}">${m.key.toUpperCase()}</span>
      </div>
      <img src="data:image/png;base64,${images[m.key]}" alt="${m.label}" loading="lazy" />
    </div>
  `).join('');

  buildMetricsTable(metrics);
}

function buildMetricsTable(metrics) {
  const wrap = $('#metricsTableWrap');
  if (!wrap) return;

  const methods = ['original', 'gcprl', 'he', 'clahe', 'minmax'];
  const labels = { original:'Original', gcprl:'GCPRL', he:'Hist. EQ', clahe:'CLAHE', minmax:'Min-Max' };

  // Find best values per metric
  const metricKeys = ['cnr','entropy','edge_preservation','brightness','processing_time'];
  const bestIsHigh = { cnr:true, entropy:true, edge_preservation:true, brightness:true, processing_time:false };
  const bestVals = {};
  metricKeys.forEach(mk => {
    const vals = methods.map(m => metrics[m]?.[mk] ?? null).filter(v => v !== null);
    bestVals[mk] = bestIsHigh[mk] ? Math.max(...vals) : Math.min(...vals);
  });

  const rows = metricKeys.map(mk => {
    const displayLabels = {
      cnr: 'CNR', entropy: 'Entropy',
      edge_preservation: 'Edge Preservation',
      brightness: 'Brightness Preservation',
      processing_time: 'Processing Time (s)',
    };
    const cells = methods.map(m => {
      const v = metrics[m]?.[mk];
      if (v == null) return '<td>—</td>';
      const isBest = Math.abs(v - bestVals[mk]) < 0.001;
      return `<td class="${isBest ? 'best-val' : ''}">${typeof v === 'number' ? v.toFixed(4) : v}</td>`;
    }).join('');
    return `<tr><td>${displayLabels[mk]}</td>${cells}</tr>`;
  }).join('');

  wrap.innerHTML = `
    <div class="metrics-table-wrap">
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Metric</th>
            ${methods.map(m => `<th>${labels[m]}</th>`).join('')}
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
      <p style="margin-top:0.75rem;font-size:0.75rem;color:var(--text-muted);">
        <span style="color:var(--accent-success);">■</span> Green = best value for that metric
      </p>
    </div>
  `;
}

// ─────────────────────── Difference Map ───────────────────────
window.showDiffMap = async (colormap) => {
  if (!App.jobId || !App.enhancedB64) return;

  colormap = colormap || $('#colormapSelector .active')?.dataset.colormap || 'jet';
  showProgress('Computing difference map…');

  try {
    const res = await fetch('/difference_map', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: App.jobId, colormap }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    setDiffImages(data);
    updateDiffStats(data.max_diff, data.mean_diff);
    drawDiffHistogram('diffHistCanvas', data.diff_histogram);

    const diffSection = $('#diffSection');
    diffSection.classList.add('active', 'fade-in');
    diffSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    App.heatmapFilename = data.heatmap_filename;
    toast('Difference map generated', 'info');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    hideProgress();
  }
};

function setDiffImages(data) {
  const ids = {
    diffGrayImg: data.diff_gray,
    diffHeatmapImg: data.heatmap,
    diffOverlayImg: data.overlay,
  };
  Object.entries(ids).forEach(([id, b64]) => {
    const el = $(`#${id}`);
    if (el) el.src = `data:image/png;base64,${b64}`;
  });
}

function updateDiffStats(maxDiff, meanDiff) {
  const maxEl = $('#diffMaxVal');
  const meanEl = $('#diffMeanVal');
  if (maxEl) maxEl.textContent = maxDiff;
  if (meanEl) meanEl.textContent = meanDiff;
}

// ─────────────────────── Colormap selection ───────────────────────
window.selectColormap = (btn, colormap) => {
  $$('#colormapSelector .colormap-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  window.showDiffMap(colormap);
};

// ─────────────────────── Side-by-side slider ───────────────────────
function buildSlider() {
  if (!App.originalB64 || !App.enhancedB64) return;
  const container = $('#sliderContainer');
  if (!container) return;
  if (App.slider) {
    App.slider.update(App.enhancedB64, App.originalB64);
  } else {
    App.slider = new ImageComparisonSlider(container, App.enhancedB64, App.originalB64, ['Enhanced', 'Original']);
  }
  $('#sliderSection')?.classList.remove('hidden');
}

// ─────────────────────── UI Helpers ───────────────────────
function setImage(imgId, b64) {
  const img = $(`#${imgId}`);
  if (!img) return;
  img.src = `data:image/png;base64,${b64}`;
  img.classList.remove('hidden');
  img.onclick = () => openLightbox(b64);
  img.style.cursor = 'zoom-in';
  // hide placeholder
  const placeholder = img.nextElementSibling;
  if (placeholder?.classList.contains('image-placeholder')) placeholder.classList.add('hidden');
}

function clearPanel(imgId, placeholderId) {
  const img = $(`#${imgId}`);
  const ph = $(`#${placeholderId}`);
  if (img) { img.src = ''; img.classList.add('hidden'); }
  if (ph) ph.classList.remove('hidden');
}

function showMeta(meta) {
  const strip = $('#metaStrip');
  if (!strip) return;
  strip.innerHTML = [
    ['Dimensions', `${meta.width} × ${meta.height}px`],
    ['Channels', meta.channels],
    ['Depth', meta.dtype || '8-bit'],
    ['Modality', meta.modality || 'Unknown'],
  ].map(([l, v]) => `<span class="meta-item"><span class="meta-label">${l}:</span><span class="meta-value">${v}</span></span>`).join('');
  strip.classList.remove('hidden');
}

function showMetrics(m) {
  const section = $('#metricsSection');
  if (!section) return;
  section.classList.remove('hidden');

  const items = [
    { label: 'CNR', orig: m.cnr_original, enh: m.cnr_enhanced },
    { label: 'Entropy', orig: m.entropy_original, enh: m.entropy_enhanced },
    { label: 'Edge Preservation', orig: '—', enh: m.edge_preservation },
    { label: 'Brightness Score', orig: '—', enh: m.brightness_preservation },
  ];

  const grid = $('#metricsGrid');
  if (!grid) return;

  grid.innerHTML = items.map(item => {
    const origVal = typeof item.orig === 'number' ? item.orig.toFixed(4) : item.orig;
    const enhVal = typeof item.enh === 'number' ? item.enh.toFixed(4) : item.enh;
    let delta = '';
    if (typeof item.orig === 'number' && typeof item.enh === 'number') {
      const diff = item.enh - item.orig;
      const sign = diff >= 0 ? '+' : '';
      delta = `<span class="metric-delta ${diff >= 0 ? 'delta-pos' : 'delta-neg'}">${sign}${diff.toFixed(3)}</span>`;
    }
    return `
      <div class="metric-card">
        <div class="metric-label">${item.label}</div>
        <div class="metric-values">
          <span class="metric-original">${origVal}</span>
          <span class="metric-arrow">→</span>
          <span class="metric-enhanced">${enhVal}</span>
          ${delta}
        </div>
      </div>
    `;
  }).join('');

  // Enhancement info
  const affineEl = $('#affineInfo');
  if (affineEl) {
    const gainRange = m.gain_range || `${m.affine_a}–${m.affine_b}`;
    affineEl.innerHTML = `
      <span class="font-mono text-muted">Sigmoid Gain: </span>
      <span class="font-mono text-accent">${gainRange}</span>
      <span class="font-mono text-muted" style="margin-left:16px">k = </span>
      <span class="font-mono text-accent">${m.k}</span>
      <span class="font-mono text-muted" style="margin-left:16px">Time: </span>
      <span class="font-mono text-accent">${m.processing_time_s}s</span>
    `;
  }
}

function resetComparisons() {
  ['#comparisonSection', '#diffSection'].forEach(sel => {
    $(sel)?.classList.remove('active');
  });
  $$('.needs-enhanced').forEach(b => b.disabled = true);
  $('#sliderSection')?.classList.add('hidden');
  const metricsSection = $('#metricsSection');
  if (metricsSection) metricsSection.classList.add('hidden');
}

// ─────────────────────── Histograms ───────────────────────
function drawHistogram(canvasId, hist, enhHist) {
  const canvas = $(`#${canvasId}`);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || canvas.width;
  const H = canvas.height;
  canvas.width = W;
  ctx.clearRect(0, 0, W, H);

  const counts = hist.counts || hist.original_counts || [];
  const enhCounts = enhHist?.enhanced_counts || hist.enhanced_counts || null;

  if (!counts.length) return;

  const maxVal = Math.max(...counts, ...(enhCounts || []));
  const barW = W / counts.length;

  // Draw original histogram
  counts.forEach((c, i) => {
    const h = (c / maxVal) * (H - 4);
    ctx.fillStyle = 'rgba(144,168,195,0.4)';
    ctx.fillRect(i * barW, H - h, barW, h);
  });

  // Draw enhanced histogram overlay
  if (enhCounts) {
    enhCounts.forEach((c, i) => {
      const h = (c / maxVal) * (H - 4);
      ctx.fillStyle = 'rgba(0,184,212,0.5)';
      ctx.fillRect(i * barW, H - h, barW, h);
    });
  }

  // Legend
  ctx.font = '10px DM Mono, monospace';
  ctx.fillStyle = 'rgba(144,168,195,0.8)';
  ctx.fillText('■ Original', 8, 14);
  if (enhCounts) {
    ctx.fillStyle = 'rgba(0,184,212,0.9)';
    ctx.fillText('■ Enhanced', 80, 14);
  }
}

function drawDiffHistogram(canvasId, hist) {
  const canvas = $(`#${canvasId}`);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || canvas.width;
  const H = canvas.height;
  canvas.width = W;
  ctx.clearRect(0, 0, W, H);

  const counts = hist.counts || [];
  if (!counts.length) return;
  const maxVal = Math.max(...counts);
  const barW = W / counts.length;

  counts.forEach((c, i) => {
    const h = (c / maxVal) * (H - 4);
    const pct = i / counts.length;
    ctx.fillStyle = `hsla(${270 - pct * 90}, 80%, 60%, 0.7)`;
    ctx.fillRect(i * barW, H - h, barW, h);
  });
}

// ─────────────────────── Sliders ───────────────────────
function initSliders() {
  const kSlider = $('#kSlider');
  const kVal = $('#kValue');
  if (kSlider && kVal) {
    kVal.textContent = parseFloat(kSlider.value).toFixed(2);
    kSlider.addEventListener('input', () => { kVal.textContent = parseFloat(kSlider.value).toFixed(2); });
  }

  const wSlider = $('#windowSlider');
  const wVal = $('#windowValue');
  if (wSlider && wVal) {
    wVal.textContent = wSlider.value;
    wSlider.addEventListener('input', () => {
      let v = parseInt(wSlider.value);
      if (v % 2 === 0) { v++; wSlider.value = v; }
      wVal.textContent = v;
    });
  }

  const aSlider = $('#alphaSlider');
  const aVal = $('#alphaValue');
  if (aSlider && aVal) {
    aVal.textContent = parseFloat(aSlider.value).toFixed(2);
    aSlider.addEventListener('input', () => {
      aVal.textContent = parseFloat(aSlider.value).toFixed(2);
    });
  }

  const sSlider = $('#stretchSlider');
  const sVal = $('#stretchValue');
  if (sSlider && sVal) {
    sVal.textContent = parseFloat(sSlider.value).toFixed(2);
    sSlider.addEventListener('input', () => {
      sVal.textContent = parseFloat(sSlider.value).toFixed(2);
    });
  }

  const bSlider = $('#brightnessSlider');
  const bVal = $('#brightnessValue');
  if (bSlider && bVal) {
    bVal.textContent = bSlider.value;
    bSlider.addEventListener('input', () => {
      bVal.textContent = bSlider.value;
    });
  }
}

// ─────────────────────── Buttons ───────────────────────
function initButtons() {
  // Download enhanced
  const dlBtn = $('#downloadEnhanced');
  if (dlBtn) {
    dlBtn.addEventListener('click', () => {
      if (App.enhancedFilename) {
        window.location.href = `/download/${App.enhancedFilename}`;
      }
    });
  }

  // Report
  const rptBtn = $('#reportBtn');
  if (rptBtn) {
    rptBtn.addEventListener('click', () => {
      if (App.jobId) window.open(`/report/${App.jobId}`, '_blank');
    });
  }
}

// ─────────────────────── Lightbox ───────────────────────
function initLightbox() {
  const lb = $('#lightbox');
  if (!lb) return;
  lb.addEventListener('click', (e) => {
    if (e.target === lb) closeLightbox();
  });
}

window.openLightbox = (b64, filename) => {
  const lb = $('#lightbox');
  const img = $('#lightboxImg');
  if (!lb || !img) return;
  img.src = `data:image/png;base64,${b64}`;
  App.lightboxImg = b64;
  App.lightboxFilename = filename || 'image.png';
  lb.classList.add('open');
};

window.closeLightbox = () => $('#lightbox')?.classList.remove('open');

window.downloadLightbox = () => {
  if (!App.lightboxImg) return;
  const a = document.createElement('a');
  a.href = `data:image/png;base64,${App.lightboxImg}`;
  a.download = App.lightboxFilename || 'image.png';
  a.click();
};

// Close on Escape
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeLightbox();
});

// ─────────────────────── Progress ───────────────────────
function showProgress(msg) {
  const overlay = $('#progressOverlay');
  const text = $('#progressText');
  if (overlay) overlay.classList.add('active');
  if (text && msg) text.textContent = msg;
}

function hideProgress() {
  $('#progressOverlay')?.classList.remove('active');
}

// ─────────────────────── Toast ───────────────────────
function toast(msg, type = 'info') {
  const icons = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' };
  const container = $('#toastContainer');
  if (!container) return;

  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ'}</span><span>${msg}</span>`;
  container.appendChild(el);

  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(100%)';
    el.style.transition = '0.3s ease';
    setTimeout(() => el.remove(), 300);
  }, 3500);
}

// ─────────────────────── Auto-Optimize ───────────────────────
window.autoOptimize = async () => {
  if (!App.jobId) return;

  showProgress('Analyzing image characteristics…');
  try {
    const res = await fetch('/auto_optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: App.jobId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Auto-optimize failed');

    applyOptimizedParams(data.params);
    showRationale(data.params, data.rationale, data.stats);
    toast('Parameters optimized for this image!', 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    hideProgress();
  }
};

function applyOptimizedParams(params) {
  // Apply to sliders + update displayed values
  const set = (id, valId, val, fmt) => {
    const el = $(`#${id}`);
    const vEl = $(`#${valId}`);
    if (el) el.value = val;
    if (vEl) vEl.textContent = fmt ? fmt(val) : val;
  };
  set('kSlider',      'kValue',       params.k,           v => parseFloat(v).toFixed(2));
  set('windowSlider', 'windowValue',  params.window_size, v => v);
  set('alphaSlider',  'alphaValue',   params.local_alpha,  v => parseFloat(v).toFixed(2));
  set('stretchSlider','stretchValue', params.stretch,      v => parseFloat(v).toFixed(2));
  set('brightnessSlider','brightnessValue', params.brightness || 0, v => v);
  const cb = $('#preserveCheck');
  if (cb) cb.checked = params.preserve_diagnostic !== false;
}

function showRationale(params, rationale, stats) {
  const panel = $('#rationalePanel');
  if (!panel) return;

  const grid = $('#rationaleGrid');
  if (!grid) return;
  grid.innerHTML = '';

  // Add the Regression Model info first if available (CO 5)
  if (rationale.Model) {
    const modelDiv = document.createElement('div');
    modelDiv.style.gridColumn = "1 / -1";
    modelDiv.style.background = "rgba(0,184,212,0.1)";
    modelDiv.style.padding = "10px";
    modelDiv.style.borderRadius = "4px";
    modelDiv.style.marginBottom = "10px";
    modelDiv.style.fontSize = "0.85rem";
    modelDiv.innerHTML = `<strong>Σ Mathematical Logic (CO 5):</strong> ${rationale.Model}`;
    grid.appendChild(modelDiv);
  }

  // Stats pills
  const statsEl = $('#rationaleStats');
  if (statsEl) {
    const statItems = [
      ['Range', `${(stats.percentile_range * 100).toFixed(1)}%`],
      ['Std Dev', `${(stats.std_norm * 100).toFixed(1)}%`],
      ['Brightness', `${(stats.mean_norm * 100).toFixed(0)}%`],
      ['Entropy', stats.entropy.toFixed(2)],
      ['Structure', stats.mean_variance.toFixed(5)],
      ['Noise', stats.noise_estimate.toFixed(4)],
    ];
    statsEl.innerHTML = statItems.map(([l, v]) =>
      `<span class="stat-pill">${l}<span>${v}</span></span>`
    ).join('');
  }

  // Rationale cards
  const paramLabels = {
    k: 'Enhancement Strength (k)',
    window_size: 'Variance Window',
    local_alpha: 'Local Contrast (α)',
    stretch: 'CDF Stretch',
  };
  const paramValues = {
    k: parseFloat(params.k).toFixed(2),
    window_size: params.window_size,
    local_alpha: parseFloat(params.local_alpha).toFixed(2),
    stretch: parseFloat(params.stretch).toFixed(2),
  };

  const cardsHtml = Object.entries(paramLabels).map(([key, label]) => `
    <div class="rationale-card">
      <div class="rationale-param">
        ${label}
        <span class="rationale-value">${paramValues[key]}</span>
      </div>
      <div class="rationale-reason">${rationale[key] || '—'}</div>
    </div>
  `).join('');

  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = cardsHtml;
  while (tempDiv.firstChild) {
    grid.appendChild(tempDiv.firstChild);
  }

  panel.classList.remove('hidden');
}
