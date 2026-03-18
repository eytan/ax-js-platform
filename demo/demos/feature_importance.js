import { libraryScript, vizScript, fixtureScript, penicillinFixture, axHomeLink } from '../shared.js';

export default function() {
return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>axjs — Feature Importance</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f0f11; color: #e0e0e0; padding: 2rem; min-height: 100vh; }
  h1 { font-size: 18px; font-weight: 500; color: #f0f0f0; margin-bottom: 4px; }
  .subtitle { font-size: 13px; color: #888; margin-bottom: 16px; }
  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
  label { font-size: 13px; color: #aaa; }
  select, input[type=file] { font-size: 13px; padding: 5px 10px;
    border-radius: 6px; border: 0.5px solid #444; background: #1a1a1d; color: #e0e0e0; cursor: pointer; outline: none; }
  .explanation { font-size: 12px; color: #666; margin-top: 16px; max-width: 600px; line-height: 1.5; }
</style>
</head>
<body>
<h1>${axHomeLink}Feature Importance</h1>
<div class="subtitle" id="subtitle">Relative importance from GP kernel lengthscales</div>
<div class="controls">
  <label>File: <input type="file" id="fileInput" accept=".json"></label>
</div>
<div id="chart"></div>
<div class="explanation">
  Bars show <b>1 / lengthscale</b> — shorter kernel lengthscales mean the model is more
  sensitive to that parameter. Importance is relative (normalized to the most important dimension).
</div>
${libraryScript()}
${fixtureScript('__DEFAULT_FIXTURE__', penicillinFixture)}
${vizScript()}
<script>
var Predictor = Ax.Predictor;
var predictor, fixture;

function loadFixtureData(data) {
  fixture = Ax.viz.normalizeFixture(data);
  predictor = new Predictor(fixture);
  document.getElementById('subtitle').textContent =
    (fixture.metadata.name || 'Fixture') + ' — ' + predictor.paramNames.length + ' parameters, ' +
    predictor.outcomeNames.length + ' outcome' + (predictor.outcomeNames.length > 1 ? 's' : '');
  render();
}

function render() {
  var chart = document.getElementById('chart');
  chart.innerHTML = '';
  Ax.viz.renderFeatureImportance(chart, predictor, { interactive: true });
}

Ax.viz.setupFileUpload('fileInput', function(data) { loadFixtureData(data); });

loadFixtureData(__DEFAULT_FIXTURE__);
</script>
</body>
</html>`;
}
