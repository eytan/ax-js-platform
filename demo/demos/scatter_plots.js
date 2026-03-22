import { libraryScript, vizScript, fixtureScript, penicillinFixture, branincurrinFixture, axHomeLink, axFavicon } from '../shared.js';

export default function() {
return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>axjs — Scatter Plots</title>
${axFavicon}
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #fff; color: #1a1a1a; padding: 2rem; min-height: 100vh; }
  h1 { font-size: 18px; font-weight: 500; color: #111; margin-bottom: 4px; }
  h2 { font-size: 15px; font-weight: 500; color: #333; margin: 24px 0 8px; }
  .subtitle { font-size: 13px; color: #666; margin-bottom: 16px; }
  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
  label { font-size: 13px; color: #555; }
  select, button, input[type=file] { font-size: 13px; padding: 5px 10px;
    border-radius: 6px; border: 0.5px solid #d0d0d0; background: #fff; color: #333; cursor: pointer; outline: none; }
  button:hover { background: #f0f0f0; }
  .hint { font-size: 11px; color: #999; margin-top: 8px; }
  .section { margin-bottom: 32px; border-bottom: 1px solid #f0f0f0; padding-bottom: 24px; }
  .section:last-child { border-bottom: none; }
  .grid { display: flex; flex-wrap: wrap; gap: 24px; }
  .panel { flex: 0 0 auto; }
  .tab-bar { display: flex; gap: 0; margin-bottom: 16px; }
  .tab { padding: 6px 16px; font-size: 13px; cursor: pointer; border: 0.5px solid #d0d0d0;
    background: #f8f8f8; color: #666; user-select: none; }
  .tab:first-child { border-radius: 6px 0 0 6px; }
  .tab:last-child { border-radius: 0 6px 6px 0; }
  .tab.active { background: #4872f9; color: #fff; border-color: #4872f9; }
</style>
</head>
<body>
<h1>${axHomeLink}Scatter Plots</h1>
<div class="subtitle">Generic scatter, observed vs predicted, Pareto frontier, and in-sample effects</div>
<div class="controls">
  <label>File: <input type="file" id="fileInput" accept=".json"></label>
  <button id="loadPenicillin">Penicillin (3 outcomes)</button>
  <button id="loadBranincurrin">Branin-Currin (2 outcomes)</button>
</div>
<div class="hint" style="margin-bottom:16px">Click any dot to pin it and see kernel-distance neighbors. Hover while pinned for temporary highlights. Click empty space to unpin.</div>

<div id="content"></div>

${libraryScript()}
${fixtureScript('__PENICILLIN__', penicillinFixture)}
${fixtureScript('__BRANINCURRIN__', branincurrinFixture)}
${vizScript()}
<script>
var Predictor = Ax.Predictor;
var predictor, fixture, currentName;

function loadFixtureData(data, name) {
  fixture = Ax.viz.normalizeFixture(data);
  predictor = new Predictor(fixture);
  currentName = name || fixture.metadata.name || 'Fixture';
  render();
}

function render() {
  var content = document.getElementById('content');
  content.innerHTML = '';

  var outcomes = predictor.outcomeNames;
  var hasMulti = outcomes.length > 1;

  function renderSection(title, hint, renderFn) {
    var sec = document.createElement('div');
    sec.className = 'section';
    sec.innerHTML = '<h2>' + title + '</h2><div class="hint" style="margin-bottom:8px">' + hint + '</div>';
    var plotDiv = document.createElement('div');
    sec.appendChild(plotDiv);
    content.appendChild(sec);
    try {
      renderFn(plotDiv);
    } catch(e) {
      plotDiv.innerHTML = '<div style="color:red;font-size:13px">Error: ' + e.message + '</div>';
      console.error(title + ':', e);
    }
  }

  renderSection('Observed vs Predicted',
    'In-sample predictions with 2\u03C3 CI whiskers and R\u00B2 annotation. Diagonal = perfect calibration.',
    function(div) {
      Ax.viz.renderObservedPredicted(div, predictor, { interactive: true, width: 460, height: 460 });
    });

  renderSection('In-Sample Effects',
    'Per-trial observed (red) vs LOO prediction (blue) with 2\u03C3 CI. Sort by trial order, predicted, or observed value.',
    function(div) {
      Ax.viz.renderEffectsPlot(div, predictor, { interactive: true, width: 520, height: 400 });
    });

  if (hasMulti) {
    renderSection('Pareto Frontier',
      'Bi-objective scatter with non-dominated step-line frontier and predictive CI whiskers.',
      function(div) {
        Ax.viz.renderParetoPlot(div, predictor, { interactive: true, width: 460, height: 460 });
      });
  }

  renderSection('Generic Scatter',
    'Flexible scatter \u2014 choose any outcome or parameter for each axis.',
    function(div) {
      Ax.viz.renderScatter(div, predictor, { interactive: true, width: 460, height: 460 });
    });
}

document.getElementById('fileInput').addEventListener('change', function(e) {
  var file = e.target.files[0]; if (!file) return;
  file.text().then(function(text) { loadFixtureData(JSON.parse(text), file.name); });
});

document.getElementById('loadPenicillin').addEventListener('click', function() {
  loadFixtureData(__PENICILLIN__, 'Penicillin');
});
document.getElementById('loadBranincurrin').addEventListener('click', function() {
  loadFixtureData(__BRANINCURRIN__, 'Branin-Currin');
});

// Default: load penicillin
loadFixtureData(__PENICILLIN__, 'Penicillin');
</script>
</body>
</html>`;
}
