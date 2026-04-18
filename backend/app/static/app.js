const fieldConfig = [
  { name: "nitrogen", label: "Nitrogen" },
  { name: "phosphorous", label: "Phosphorous" },
  { name: "potassium", label: "Potassium" },
  { name: "ph", label: "pH" },
  { name: "temperature_c", label: "Temperature (C)" },
  { name: "humidity", label: "Humidity" },
  { name: "rainfall_mm", label: "Rainfall (mm)" },
  { name: "area", label: "Area (hectares)", readonly: true },
];

const INDIA_CENTER = [20.5937, 78.9629];
const MAP_ZOOM_DEFAULT = 5;
const MAP_ZOOM_CLOSE = 18;

const form = document.getElementById("prediction-form");
const formGrid = document.getElementById("form-grid");
const statusEl = document.getElementById("status");
const bestCropEl = document.getElementById("best-crop");
const idealGroundEl = document.getElementById("ideal-ground-card");
const table = document.getElementById("results-table");
const tbody = table.querySelector("tbody");
const metricsGrid = document.getElementById("training-metrics");
const xaiPanel = document.getElementById("xai-panel");
const localExplanationEl = document.getElementById("local-explanation");
const modelOverviewEl = document.getElementById("model-overview");
const globalFeaturesEl = document.getElementById("global-features");
const rejectedCropsEl = document.getElementById("rejected-crops");
const plotLightgbmEl = document.getElementById("plot-lightgbm");
const plotCatboostEl = document.getElementById("plot-catboost");
const areaDisplay = document.getElementById("area-display");
const coordsDisplay = document.getElementById("coords-display");
const recentRainfallDisplay = document.getElementById("recent-rainfall-display");
const climateRainfallDisplay = document.getElementById("climate-rainfall-display");
const locationBanner = document.getElementById("location-banner");
const locateBtn = document.getElementById("locate-btn");
const clearMapBtn = document.getElementById("clear-map-btn");
const drawPolygonBtn = document.getElementById("draw-polygon-btn");
const drawRectangleBtn = document.getElementById("draw-rectangle-btn");
const drawCircleBtn = document.getElementById("draw-circle-btn");

let metadata = null;
let map = null;
let drawnItems = null;
let activeShape = null;
let anchorMarker = null;
let drawHandlers = null;

const state = {
  areaHectares: null,
  latitude: null,
  longitude: null,
  locationLabel: "Mark or draw your land on the map.",
  recentRainfallMm: null,
  climateRainfallMm: null,
};

function setStatus(message) {
  statusEl.textContent = message;
}

function normalizeText(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function parseFlexibleNumber(value, fallback = null) {
  if (value === null || value === undefined) return fallback;
  const cleaned = String(value).trim().replace(/,/g, "");
  if (!cleaned) return fallback;
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return "--";
  return Number(value).toFixed(digits);
}

function createField(config, defaultValue) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  const label = document.createElement("label");
  label.setAttribute("for", config.name);
  label.textContent = config.label;

  const input = document.createElement("input");
  input.id = config.name;
  input.name = config.name;
  input.type = "text";
  input.inputMode = "decimal";
  if (defaultValue !== undefined && defaultValue !== null) {
    input.value = defaultValue;
  }
  if (config.readonly) {
    input.readOnly = true;
  }

  wrapper.append(label, input);
  return wrapper;
}

function renderForm() {
  const defaults = metadata?.default_inputs || {};
  formGrid.innerHTML = "";
  fieldConfig.forEach((config) => {
    const value = config.name === "area" ? "" : defaults[config.name];
    formGrid.appendChild(createField(config, value));
  });
}

function renderMetrics(report) {
  metricsGrid.innerHTML = "";
  if (!report) return;

  const cards = [
    ["Stacking Accuracy", report.classification_metrics?.stacking?.accuracy],
    ["Top-3 Accuracy", report.classification_metrics?.stacking?.top3_accuracy],
    ["LightGBM Accuracy", report.classification_metrics?.lightgbm?.accuracy],
    ["Yield R²", report.regression_metrics?.r2],
  ];

  cards.forEach(([label, value]) => {
    if (value === undefined) return;
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `<span>${label}</span><strong>${Number(value).toFixed(4)}</strong>`;
    metricsGrid.appendChild(card);
  });
}

function renderGraphRows(items) {
  modelOverviewEl.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "graph-row";
    row.innerHTML = `
      <header>
        <strong>${item.model}</strong>
        <span>${formatNumber(item.accuracy * 100, 2)}%</span>
      </header>
      <div class="graph-bar"><span style="width:${Math.max(2, item.accuracy * 100)}%"></span></div>
    `;
    modelOverviewEl.appendChild(row);
  });
}

function renderGlobalFeatures(features) {
  globalFeaturesEl.innerHTML = "";
  features.forEach((feature) => {
    const chip = document.createElement("div");
    chip.className = "feature-chip";
    chip.innerHTML = `
      <strong>${String(feature.feature || "").replace(/^num__|^cat__/, "").replace(/_/g, " ")}</strong>
      <span>importance ${formatNumber(Number(feature.mean_importance || 0), 3)}</span>
    `;
    globalFeaturesEl.appendChild(chip);
  });
}

function renderPlotImages(assets) {
  const lightgbmPlot =
    assets?.lightgbm_summary_plot ||
    assets?.summary_plot_urls?.lightgbm ||
    "/reports/realistic_v2/shap_summary_lightgbm.png";
  const catboostPlot =
    assets?.catboost_summary_plot ||
    assets?.summary_plot_urls?.catboost ||
    "/reports/realistic_v2/shap_summary_catboost.png";
  plotLightgbmEl.src = lightgbmPlot || "";
  plotCatboostEl.src = catboostPlot || "";
  plotLightgbmEl.style.display = lightgbmPlot ? "block" : "none";
  plotCatboostEl.style.display = catboostPlot ? "block" : "none";
}

function renderRejectedCrops(rejected) {
  rejectedCropsEl.innerHTML = "";
  if (!rejected?.length) {
    rejectedCropsEl.innerHTML = "<p>No crops were rejected for the current date and field condition.</p>";
    return;
  }
  const list = document.createElement("ul");
  list.className = "rejection-list";
  rejected.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.crop}: ${item.reason}`;
    list.appendChild(li);
  });
  rejectedCropsEl.appendChild(list);
}

function renderLocalExplanation(explainability) {
  localExplanationEl.innerHTML = "";
  const best = explainability?.best_crop_local_explanation;
  if (!best) {
    localExplanationEl.innerHTML = "<p>Prediction-level explainability will appear after you run a recommendation.</p>";
    return;
  }

  const blocks = [
    {
      title: "Rule-Based Positives",
      items: best.rule_based_positives || [],
    },
    {
      title: "Rule-Based Concerns",
      items: best.rule_based_concerns || [],
    },
    {
      title: "Classification SHAP Drivers",
      items: (best.classification_shap?.positive || []).slice(0, 4).map(
        (item) => `${item.feature}: +${formatNumber(item.contribution, 3)}`
      ),
    },
    {
      title: "Yield SHAP Drivers",
      items: (best.yield_shap?.positive || []).slice(0, 4).map(
        (item) => `${item.feature}: +${formatNumber(item.contribution, 3)}`
      ),
    },
  ];

  blocks.forEach((block) => {
    const wrapper = document.createElement("section");
    wrapper.className = "explanation-block";
    const items = block.items?.length
      ? `<ul class="explanation-list">${block.items.map((item) => `<li>${item}</li>`).join("")}</ul>`
      : "<p>No strong signals for this part.</p>";
    wrapper.innerHTML = `<h4>${block.title}</h4>${items}`;
    localExplanationEl.appendChild(wrapper);
  });
}

function renderExplainabilityMetadata(meta) {
  const xaiAssets = meta?.xai_assets || {};
  renderGlobalFeatures((xaiAssets.top_features || []).slice(0, 8));
  renderPlotImages(xaiAssets);

  const report = meta?.training_report || {};
  const modelOverview = Object.entries(report.classification_metrics || {})
    .filter(([, value]) => value && value.accuracy !== undefined)
    .map(([key, value]) => ({
      model: key.replace(/_/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase()),
      accuracy: Number(value.accuracy || 0),
    }))
    .sort((a, b) => b.accuracy - a.accuracy);
  renderGraphRows(modelOverview);
  renderLocalExplanation(null);
  renderRejectedCrops([]);
}

function renderPredictionExplainability(result) {
  renderLocalExplanation(result.explainability);
  renderRejectedCrops(result.rejected_crops || []);
  const modelOverview = result.explainability?.model_overview || [];
  if (modelOverview.length) {
    renderGraphRows(modelOverview);
  }
  const globalFeatures = result.explainability?.global_top_features || [];
  if (globalFeatures.length) {
    renderGlobalFeatures(globalFeatures);
  }
  renderPlotImages(result.explainability || {});
  xaiPanel.classList.remove("hidden");
}

function renderResults(result) {
  const best = result.top_crops[0];
  bestCropEl.classList.remove("hidden");
  bestCropEl.innerHTML = `
    <h3>Best Crop: ${best.crop}</h3>
    <p>
      Sow in ${best.sowing_month}, expect harvest in ${best.harvest_month},
      with a conservative yield of ${best.expected_yield_t_ha.toFixed(2)} t/ha,
      estimated profit of ${best.profit.toFixed(2)} for your full land.
      The first 4 results prioritize lower or medium spend, and the 5th slot keeps one higher-spend, higher-profit option.
    </p>
    <p>
      and final score ${best.final_score.toFixed(4)}.
    </p>
  `;

  const ideal = result.ideal_ground_recommendation;
  if (ideal) {
    idealGroundEl.classList.remove("hidden");
    idealGroundEl.innerHTML = `
      <h3>Ideal Ground Crop: ${ideal.crop}</h3>
      <p>
        This is the separate safe long-term recommendation for your land, independent of this month.
        It focuses on land fit, lower risk, lower cost, and stable profit instead of immediate sowing timing.
      </p>
      <p>
        Land suitability ${ideal.land_suitability.toFixed(4)}, expected yield ${ideal.expected_yield_t_ha.toFixed(2)} t/ha,
        stable price ${ideal.stable_price_rs_per_kg.toFixed(2)} ₹/kg,
        cost ${ideal.total_cost_rs_per_ha.toFixed(2)} ₹/ha,
        profit ${ideal.profit_rs_per_ha.toFixed(2)} ₹/ha,
        risk ${ideal.risk.toFixed(4)}, and ideal score ${ideal.ideal_ground_score.toFixed(4)}.
      </p>
      <p>${(ideal.why || []).slice(0, 3).join(" ")}</p>
    `;
  } else {
    idealGroundEl.classList.add("hidden");
    idealGroundEl.innerHTML = "";
  }

  tbody.innerHTML = "";
  result.top_crops.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.crop}</td>
      <td>${item.sowing_month}</td>
      <td>${item.harvest_month}</td>
      <td>${item.duration_months}</td>
      <td>${item.yield_start_month}</td>
      <td>${item.expected_yield_t_ha.toFixed(2)}</td>
      <td>${item.adjusted_price_rs_per_kg.toFixed(2)}</td>
      <td>${item.total_cost_rs_per_ha.toFixed(2)}</td>
      <td>${item.revenue_rs_per_ha.toFixed(2)}</td>
      <td>${item.profit_rs_per_ha.toFixed(2)}</td>
      <td>${item.risk.toFixed(4)}</td>
      <td>${item.sustainability_score.toFixed(4)}</td>
      <td>${item.final_score.toFixed(4)}</td>
    `;
    tbody.appendChild(row);
  });
  table.classList.remove("hidden");
  renderMetrics(result.training_summary);
  renderPredictionExplainability(result);
}

function updateAreaDisplay() {
  if (!state.areaHectares) {
    areaDisplay.textContent = "Draw on map";
    return;
  }
  areaDisplay.textContent = `${formatNumber(state.areaHectares, 3)} ha`;
}

function updateCoordsDisplay() {
  if (!Number.isFinite(state.latitude) || !Number.isFinite(state.longitude)) {
    coordsDisplay.textContent = "Waiting for location";
    return;
  }
  coordsDisplay.textContent = `${formatNumber(state.latitude, 5)}, ${formatNumber(state.longitude, 5)}`;
}

function updateLocationBanner() {
  locationBanner.textContent = state.locationLabel;
}

function sumPositive(values) {
  return values
    .filter((value) => Number.isFinite(value) && Number(value) > 0)
    .reduce((sum, value) => sum + Number(value), 0);
}

function updateRainfallDisplays() {
  recentRainfallDisplay.textContent = Number.isFinite(state.recentRainfallMm)
    ? `${formatNumber(state.recentRainfallMm, 2)} mm`
    : "Waiting for map";
  climateRainfallDisplay.textContent = Number.isFinite(state.climateRainfallMm)
    ? `${formatNumber(state.climateRainfallMm, 2)} mm`
    : "Waiting for map";
}

async function fetchLiveWeather(lat, lng) {
  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lng}` +
    `&current=temperature_2m,relative_humidity_2m,rain` +
    `&forecast_days=1&timezone=auto`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Weather API request failed.");
  }
  const data = await response.json();
  const current = data.current || {};
  return {
    temperature: current.temperature_2m,
    humidity: current.relative_humidity_2m,
    currentRain: current.rain,
  };
}

function formatDate(date) {
  return date.toISOString().slice(0, 10);
}

async function fetchRainfallHistory(lat, lng) {
  const endDate = new Date();
  const startDate = new Date();
  startDate.setDate(endDate.getDate() - 364);

  const url =
    `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lng}` +
    `&start_date=${formatDate(startDate)}&end_date=${formatDate(endDate)}` +
    `&daily=precipitation_sum&timezone=auto`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Rainfall history request failed.");
  }
  const data = await response.json();
  const daily = data.daily || {};
  const precipitationSeries = Array.isArray(daily.precipitation_sum) ? daily.precipitation_sum.map(Number) : [];

  const annualTotal = sumPositive(precipitationSeries);
  const recent30Total = sumPositive(precipitationSeries.slice(-30));
  const climateMonthlyEquivalent = annualTotal > 0 ? annualTotal / 12 : null;

  return {
    annualTotal,
    recent30Total,
    climateMonthlyEquivalent,
  };
}

async function fetchWeather(lat, lng) {
  const [liveWeather, history] = await Promise.all([
    fetchLiveWeather(lat, lng),
    fetchRainfallHistory(lat, lng),
  ]);

  const temperature = liveWeather.temperature;
  const humidity = liveWeather.humidity;
  const recent30Total = history.recent30Total;
  const climateMonthlyEquivalent = history.climateMonthlyEquivalent;
  const currentRain = liveWeather.currentRain;

  let modelRainfall = null;
  if (Number.isFinite(climateMonthlyEquivalent) && Number.isFinite(recent30Total)) {
    modelRainfall = 0.7 * climateMonthlyEquivalent + 0.3 * recent30Total;
  } else if (Number.isFinite(climateMonthlyEquivalent)) {
    modelRainfall = climateMonthlyEquivalent;
  } else if (Number.isFinite(recent30Total)) {
    modelRainfall = recent30Total;
  } else if (Number.isFinite(currentRain)) {
    modelRainfall = currentRain;
  }

  state.recentRainfallMm = Number.isFinite(recent30Total) ? recent30Total : currentRain;
  state.climateRainfallMm = Number.isFinite(modelRainfall) ? modelRainfall : null;
  updateRainfallDisplays();

  if (temperature !== undefined) document.getElementById("temperature_c").value = temperature;
  if (humidity !== undefined) document.getElementById("humidity").value = humidity;
  if (modelRainfall !== undefined && modelRainfall !== null) {
    document.getElementById("rainfall_mm").value = formatNumber(modelRainfall, 2);
  }
}

function setAnchorLocation(lat, lng) {
  state.latitude = lat;
  state.longitude = lng;
  updateCoordsDisplay();

  if (!anchorMarker) {
    anchorMarker = L.marker([lat, lng]).addTo(map);
  } else {
    anchorMarker.setLatLng([lat, lng]);
  }
}

async function syncLocationAndWeather(lat, lng) {
  setAnchorLocation(lat, lng);
  setStatus("Fetching location and live weather for the selected land...");
  state.locationLabel = `Selected location: ${formatNumber(lat, 5)}, ${formatNumber(lng, 5)}`;
  updateLocationBanner();
  await fetchWeather(lat, lng).catch(() => {
    setStatus("Weather fetch failed. You can still type values manually.");
  });
  setStatus("Land location synced. Draw the farm boundary, then predict the best crop.");
}

function clearActiveShape() {
  if (activeShape && drawnItems) {
    drawnItems.removeLayer(activeShape);
  }
  activeShape = null;
  state.areaHectares = null;
  updateAreaDisplay();
  const areaInput = document.getElementById("area");
  if (areaInput) {
    areaInput.value = "";
  }
  state.recentRainfallMm = null;
  state.climateRainfallMm = null;
  updateRainfallDisplays();
}

function updateAreaFromLayer(layer) {
  let squareMeters = null;
  if (layer instanceof L.Circle) {
    squareMeters = Math.PI * layer.getRadius() * layer.getRadius();
  } else if (window.L && L.GeometryUtil && typeof L.GeometryUtil.geodesicArea === "function") {
    const latLngGroups = layer.getLatLngs();
    const points = Array.isArray(latLngGroups[0]) ? latLngGroups[0] : latLngGroups;
    squareMeters = L.GeometryUtil.geodesicArea(points);
  }

  if (Number.isFinite(squareMeters) && squareMeters > 0) {
    state.areaHectares = squareMeters / 10000;
  } else {
    state.areaHectares = null;
  }
  updateAreaDisplay();
  const areaInput = document.getElementById("area");
  if (areaInput) {
    areaInput.value = state.areaHectares ? formatNumber(state.areaHectares, 4) : "";
  }
}

function getLayerCenter(layer) {
  if (typeof layer.getBounds === "function" && layer.getBounds().isValid()) {
    return layer.getBounds().getCenter();
  }
  if (typeof layer.getLatLng === "function") {
    return layer.getLatLng();
  }
  return null;
}

function handleShapeChange(layer) {
  activeShape = layer;
  updateAreaFromLayer(layer);
  const center = getLayerCenter(layer);
  if (center) {
    if (typeof layer.getBounds === "function") {
      map.fitBounds(layer.getBounds(), { maxZoom: MAP_ZOOM_CLOSE, padding: [30, 30] });
    } else {
      map.setView(center, MAP_ZOOM_CLOSE);
    }
    syncLocationAndWeather(center.lat, center.lng).catch((error) => setStatus(error.message));
  }
}

function initMap() {
  map = L.map("map", {
    zoomControl: false,
    preferCanvas: true,
  }).setView(INDIA_CENTER, MAP_ZOOM_DEFAULT);

  window.setTimeout(() => {
    map.invalidateSize();
  }, 150);

  L.control.zoom({ position: "topright" }).addTo(map);

  L.esri.basemapLayer("Imagery").addTo(map);
  L.esri.basemapLayer("ImageryLabels").addTo(map);

  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  const drawControl = new L.Control.Draw({
    position: "topright",
    draw: {
      polygon: { allowIntersection: false, showArea: true },
      rectangle: true,
      circle: true,
      marker: false,
      polyline: false,
      circlemarker: false,
    },
    edit: {
      featureGroup: drawnItems,
      edit: true,
      remove: true,
    },
  });
  map.addControl(drawControl);

  drawHandlers = {
    polygon: new L.Draw.Polygon(map, drawControl.options.draw.polygon),
    rectangle: new L.Draw.Rectangle(map, drawControl.options.draw.rectangle),
    circle: new L.Draw.Circle(map, drawControl.options.draw.circle),
  };

  map.on(L.Draw.Event.CREATED, (event) => {
    clearActiveShape();
    const layer = event.layer;
    drawnItems.addLayer(layer);
    handleShapeChange(layer);
  });

  map.on(L.Draw.Event.EDITED, (event) => {
    event.layers.eachLayer((layer) => handleShapeChange(layer));
  });

  map.on(L.Draw.Event.DELETED, () => {
    activeShape = null;
    state.areaHectares = null;
    state.recentRainfallMm = null;
    state.climateRainfallMm = null;
    updateAreaDisplay();
    updateRainfallDisplays();
    const areaInput = document.getElementById("area");
    if (areaInput) {
      areaInput.value = "";
    }
  });

  map.on("click", (event) => {
    syncLocationAndWeather(event.latlng.lat, event.latlng.lng).catch((error) => setStatus(error.message));
  });
}

function enableDrawMode(mode) {
  if (!drawHandlers || !drawHandlers[mode]) {
    setStatus("Drawing tools are not ready yet. Refresh once and try again.");
    return;
  }
  drawHandlers[mode].enable();
  setStatus(`Drawing mode active: ${mode}. Mark your land boundary on the map.`);
}

function collectPayload() {
  const payload = { top_k: 5 };

  fieldConfig.forEach((field) => {
    payload[field.name] = parseFlexibleNumber(document.getElementById(field.name)?.value);
  });

  return payload;
}

async function loadMetadata() {
  const response = await fetch("/api/metadata");
  if (!response.ok) {
    throw new Error("Models are not trained yet. Train the backend first.");
  }
  metadata = await response.json();
  renderForm();
  renderMetrics(metadata.training_report);
  renderExplainabilityMetadata(metadata);
  xaiPanel.classList.remove("hidden");
  setStatus(
    `Loaded ${metadata.crop_count} crops with realistic season, cost, yield, and profit logic.`
  );
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    if (!state.areaHectares) {
      throw new Error("Draw your land boundary on the map first so area can be calculated.");
    }

    setStatus("Running seasonal crop recommendation for the selected land...");
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed");
    }
    renderResults(data);
    setStatus(`Prediction completed. Best crop: ${data.best_crop}`);
  } catch (error) {
    setStatus(error.message);
  }
});

locateBtn.addEventListener("click", () => {
  if (!navigator.geolocation) {
    setStatus("Geolocation is not available in this browser.");
    return;
  }

  setStatus("Fetching your device location...");
  navigator.geolocation.getCurrentPosition(
    (position) => {
      const { latitude, longitude } = position.coords;
      map.setView([latitude, longitude], 17);
      syncLocationAndWeather(latitude, longitude).catch((error) => setStatus(error.message));
    },
    () => setStatus("Unable to access your current location."),
    { enableHighAccuracy: true, timeout: 15000 }
  );
});

clearMapBtn.addEventListener("click", () => {
  clearActiveShape();
  updateAreaDisplay();
  setStatus("Map drawing cleared. Click or draw again to continue.");
});

drawPolygonBtn.addEventListener("click", () => enableDrawMode("polygon"));
drawRectangleBtn.addEventListener("click", () => enableDrawMode("rectangle"));
drawCircleBtn.addEventListener("click", () => enableDrawMode("circle"));

initMap();
updateRainfallDisplays();
loadMetadata().catch((error) => {
  renderForm();
  setStatus(error.message);
});
