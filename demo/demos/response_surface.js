import { libraryScript, vizScript, fixtureScript, branincurrinFixture, axHomeLink } from '../shared.js';

export default function() {
return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>axjs — 2D Response Surface</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f0f11; color: #e0e0e0; padding: 2rem; min-height: 100vh; }
  h1 { font-size: 18px; font-weight: 500; color: #f0f0f0; margin-bottom: 4px; }
  .subtitle { font-size: 13px; color: #888; margin-bottom: 20px; }
  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
  label { font-size: 13px; color: #aaa; }
  select, input[type=file] { font-size: 13px; padding: 5px 10px;
    border-radius: 6px; border: 0.5px solid #444; background: #1a1a1d; color: #e0e0e0; cursor: pointer; outline: none; }
</style>
</head>
<body>
<h1>${axHomeLink}2D Response Surface</h1>
<p class="subtitle" id="subtitle">Load a fixture JSON to visualize GP posterior</p>
<div class="controls">
  <label>File: <input type="file" id="fileInput" accept=".json"></label>
</div>
<div id="plotContainer"></div>
${libraryScript()}
${vizScript()}
${fixtureScript('__DEFAULT_FIXTURE__', branincurrinFixture)}
<script>
var Predictor = Ax.Predictor;
var predictor, fixture;

function loadFixtureData(data) {
  fixture = Ax.viz.normalizeFixture(data);
  predictor = new Predictor(fixture);
  document.getElementById('subtitle').textContent =
    fixture.metadata.name + ' — ' + fixture.metadata.description;
  render();
}

function render() {
  var container = document.getElementById('plotContainer');
  container.innerHTML = '';
  Ax.viz.renderResponseSurface(container, predictor, {
    interactive: true,
    width: 460,
    height: 460
  });
}

document.getElementById('fileInput').addEventListener('change', function(e) {
  var file = e.target.files[0]; if (!file) return;
  file.text().then(function(text) { loadFixtureData(JSON.parse(text)); });
});

loadFixtureData(__DEFAULT_FIXTURE__);
</script>
</body>
</html>`;
}
