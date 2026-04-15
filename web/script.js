
const API_BASE = "http://localhost:8000/api";

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const imageInput = document.getElementById("imageInput");
const chatLatency = document.getElementById("chatLatency");

const dataForm = document.getElementById("dataForm");
const dataQuestion = document.getElementById("dataQuestion");
const sqlBox = document.getElementById("sqlBox");
const tableBox = document.getElementById("tableBox");
const displayStatusBox = document.getElementById("displayStatusBox");
const dataLatency = document.getElementById("dataLatency");

const plotForm = document.getElementById("plotForm");
const plotQuestion = document.getElementById("plotQuestion");
const plotSql = document.getElementById("plotSql");
const plotImg = document.getElementById("plotImg");
const plotAskBox = document.getElementById("plotAskBox");
const plotAskForm = document.getElementById("plotAskForm");
const plotAskText = document.getElementById("plotAskText");
const plotLatency = document.getElementById("plotLatency");
const plotAskLatency = document.getElementById("plotAskLatency");
const kgLatency = document.getElementById("kgLatency");
const modelSelect = document.getElementById("modelSelect");

function addMessage(role, text) {
  const bubble = document.createElement("div");
  bubble.className = role === "user" ? "bg-blue-100 p-2 rounded w-fit" : "bg-gray-200 p-2 rounded w-fit";
  bubble.textContent = (role === "user" ? "You: " : "Assistant: ") + text;
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function currentModel() {
  return modelSelect && modelSelect.value ? modelSelect.value : "";
}

async function loadModels() {
  if (!modelSelect) return;
  try {
    const base = API_BASE.replace(/\/api$/, "");
    const resp = await fetch(`${base}/health`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const models = Array.isArray(data.models) ? data.models : [];
    if (models.length) {
      modelSelect.innerHTML = "";
      for (const name of models) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        modelSelect.appendChild(opt);
      }
    }
    if (data.model) modelSelect.value = data.model;
  } catch (err) {
    // Keep the static options if health is unavailable.
  }
}

loadModels();

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  addMessage("user", message);
  const started = performance.now();

  const fd = new FormData();
  fd.append("message", message);
  const model = currentModel();
  if (model) fd.append("model", model);
  for (const file of imageInput.files) fd.append("images", file);

  try {
    const resp = await fetch(`${API_BASE}/chat`, { method: "POST", body: fd });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    addMessage("assistant", data.content || "[No content]");
    const elapsed = ((performance.now() - started) / 1000).toFixed(2);
    if (chatLatency) chatLatency.textContent = `Chat response in ${elapsed}s`;
  } catch (err) {
    addMessage("assistant", `[Chat error] ${(err && err.message) || err}`);
    if (chatLatency) chatLatency.textContent = `Chat failed after ${((performance.now() - started)/1000).toFixed(2)}s`;
  }
  chatInput.value = "";
  imageInput.value = "";
});

dataForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = dataQuestion.value.trim();
  if (!question) return;
  sqlBox.textContent = "Thinking...";
  tableBox.innerHTML = "";
  displayStatusBox.innerHTML = "";
  if (dataLatency) dataLatency.textContent = "";
  const started = performance.now();

  try {
    const resp = await fetch(`${API_BASE}/data/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, model: currentModel() || undefined })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    sqlBox.innerHTML = `<div class="font-mono bg-gray-100 p-2 rounded">SQL: ${escapeHtml(data.sql || "")}</div>`;

    if (data.columns && data.rows) {
      const table = document.createElement("table");
      table.className = "min-w-full text-sm border";
      const thead = document.createElement("thead");
      const trHead = document.createElement("tr");
      for (const c of data.columns) {
        const th = document.createElement("th");
        th.className = "border px-2 py-1 bg-gray-50";
        th.textContent = c;
        trHead.appendChild(th);
      }
      thead.appendChild(trHead);
      table.appendChild(thead);
      const tbody = document.createElement("tbody");
      for (const row of data.rows) {
        const tr = document.createElement("tr");
        for (const cell of row) {
          const td = document.createElement("td");
          td.className = "border px-2 py-1";
          td.textContent = cell;
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      tableBox.appendChild(table);
    } else {
      tableBox.textContent = "No rows.";
    }
    renderPervasiveStatus(data.pervasive_status || []);
    if (dataLatency) dataLatency.textContent = `Query completed in ${((performance.now() - started)/1000).toFixed(2)}s`;
  } catch (err) {
    sqlBox.innerHTML = `<div class="font-mono bg-red-100 text-red-800 p-2 rounded">Error: ${(err && err.message) || err}</div>`;
    tableBox.textContent = "";
    renderPervasiveStatus([]);
    if (dataLatency) dataLatency.textContent = `Query failed after ${((performance.now() - started)/1000).toFixed(2)}s`;
  }
});

function renderPervasiveStatus(items) {
  displayStatusBox.innerHTML = "";
  if (!items || !items.length) return;
  const heading = document.createElement("div");
  heading.className = "text-sm font-semibold text-gray-700";
  heading.textContent = "Pervasive display comfort status";
  displayStatusBox.appendChild(heading);

  for (const item of items) {
    const card = document.createElement("div");
    card.className = "border border-gray-200 rounded-lg p-3 bg-gray-50";

    const topRow = document.createElement("div");
    topRow.className = "flex items-center justify-between gap-3";

    const title = document.createElement("div");
    title.className = "text-sm font-medium text-gray-800";
    const metricName = friendlyMetricLabel(item.metric);
    title.textContent = item.location ? `${metricName} - ${item.location}` : metricName;

    const badge = document.createElement("span");
    const badgeStyle = statusStyle(item.status);
    badge.className = `text-xs font-semibold px-2 py-1 rounded ${badgeStyle.className}`;
    badge.textContent = badgeStyle.label;

    topRow.appendChild(title);
    topRow.appendChild(badge);

    const detail = document.createElement("div");
    detail.className = "mt-2 text-xs text-gray-600";
    const meanVal = typeof item.mean_value === "number" ? item.mean_value.toFixed(2) : item.mean_value;
    let rangeText = "";
    if (item.recommended_min != null || item.recommended_max != null) {
      const min = item.recommended_min != null ? item.recommended_min.toFixed(1) : "–";
      const max = item.recommended_max != null ? item.recommended_max.toFixed(1) : "–";
      rangeText = ` (comfort ${min} – ${max})`;
    }
    detail.textContent = `Mean value ${meanVal}${rangeText}`;

    card.appendChild(topRow);
    card.appendChild(detail);
    displayStatusBox.appendChild(card);
  }
}

function statusStyle(status) {
  switch (status) {
    case "ok":
      return { className: "bg-green-100 text-green-800", label: "Within range" };
    case "warn-high":
      return { className: "bg-red-100 text-red-800", label: "Above range" };
    case "warn-low":
      return { className: "bg-blue-100 text-blue-800", label: "Below range" };
    default:
      return { className: "bg-gray-200 text-gray-800", label: "Info" };
  }
}

function friendlyMetricLabel(metric) {
  const map = {
    temperature_c: "Temperature",
    humidity_rh: "Humidity",
    pressure_hpa: "Pressure",
    IAQ_proxy: "IAQ score",
    gas_resistance_ohms: "Gas resistance"
  };
  return map[metric] || metric;
}

plotForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = plotQuestion.value.trim();
  if (!question) return;
  plotSql.textContent = "Thinking...";
  plotImg.classList.add("hidden");
  plotAskBox.classList.add("hidden");
  if (plotLatency) plotLatency.textContent = "";
  const started = performance.now();

  try {
    const resp = await fetch(`${API_BASE}/data/plot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (data.image_b64) {
      plotImg.src = "data:image/png;base64," + data.image_b64;
      plotImg.dataset.b64 = data.image_b64;
      plotImg.classList.remove("hidden");
      plotAskBox.classList.remove("hidden");
    }
    plotSql.innerHTML = `<div class="font-mono bg-gray-100 p-2 rounded">SQL: ${escapeHtml(data.sql || "")}</div>`;
    if (plotLatency) plotLatency.textContent = `Plot generated in ${((performance.now() - started)/1000).toFixed(2)}s`;
  } catch (err) {
    plotSql.innerHTML = `<div class="font-mono bg-red-100 text-red-800 p-2 rounded">Error: ${(err && err.message) || err}</div>`;
    if (plotLatency) plotLatency.textContent = `Plot failed after ${((performance.now() - started)/1000).toFixed(2)}s`;
  }
});

plotAskForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = plotAskText.value.trim() || "Explain the main trend in this plot.";
  addMessage("user", msg + " [sent with chart]");
  if (plotAskLatency) plotAskLatency.textContent = "";
  const started = performance.now();

  const fd = new FormData();
  fd.append("message", msg);
  const model = currentModel();
  if (model) fd.append("model", model);
  const b64 = plotImg.dataset.b64 || "";
  if (b64) {
    const blob = b64ToBlob(b64, "image/png");
    fd.append("images", new File([blob], "plot.png", { type: "image/png" }));
  }
  try {
    const resp = await fetch(`${API_BASE}/chat`, { method: "POST", body: fd });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    addMessage("assistant", data.content || "[No content]");
    if (plotAskLatency) plotAskLatency.textContent = `Chart Q&A in ${((performance.now() - started)/1000).toFixed(2)}s`;
  } catch (err) {
    addMessage("assistant", `[Chart Q&A error] ${(err && err.message) || err}`);
    if (plotAskLatency) plotAskLatency.textContent = `Chart Q&A failed after ${((performance.now() - started)/1000).toFixed(2)}s`;
  }
  plotAskText.value = "";
});

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, m => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[m]));
}

function b64ToBlob(b64Data, contentType="image/png", sliceSize=512) {
  const byteCharacters = atob(b64Data);
  const byteArrays = [];
  for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
    const slice = byteCharacters.slice(offset, offset + sliceSize);
    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) byteNumbers[i] = slice.charCodeAt(i);
    byteArrays.push(new Uint8Array(byteNumbers));
  }
  return new Blob(byteArrays, {type: contentType});
}
