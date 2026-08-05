(() => {
  const form = document.getElementById("predict-form");
  const input = document.getElementById("image-input");
  const dropZone = document.getElementById("drop-zone");
  const fileName = document.getElementById("file-name");
  const predictBtn = document.getElementById("predict-btn");
  const results = document.getElementById("results");
  const mainImage = document.getElementById("main-image");
  const leafList = document.getElementById("leaf-list");
  const leafDetail = document.getElementById("leaf-detail");

  let selectedFile = null;
  let currentResult = null;
  let currentView = "overlay";

  function formatMs(value) {
    return `${Number(value || 0).toFixed(1)} ms`;
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function formatPct(value) {
    return `${(Number(value || 0) * 100).toFixed(1)}%`;
  }

  function setFile(file) {
    selectedFile = file;
    fileName.textContent = file ? file.name : "";
    predictBtn.disabled = !file;
  }

  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
    const file = event.dataTransfer.files?.[0];
    if (file) setFile(file);
  });
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) setFile(file);
  });

  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => {
      currentView = button.getAttribute("data-view");
      document.querySelectorAll("[data-view]").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      if (currentResult?.visualizations?.[currentView]) {
        mainImage.src = currentResult.visualizations[currentView];
      }
    });
  });

  function renderSummary(result) {
    const { summary, processing, image, model } = result;
    document.getElementById("stat-total").textContent = summary.total_leaves;
    document.getElementById("stat-diseased").textContent = summary.diseased_leaves;
    document.getElementById("stat-confidence").textContent = formatPct(summary.average_confidence);
    document.getElementById("stat-time").textContent = formatMs(processing.total_time_ms);

    document.getElementById("sum-healthy").textContent = summary.healthy_leaves;
    document.getElementById("sum-diseased").textContent = summary.diseased_leaves;
    document.getElementById("sum-diseases").textContent =
      summary.detected_diseases?.length ? summary.detected_diseases.join(", ") : "none";
    document.getElementById("sum-confidence").textContent = formatPct(summary.average_confidence);

    document.getElementById("info-resolution").textContent = `${image.width} × ${image.height}`;
    document.getElementById("info-size").textContent = formatBytes(image.file_size);
    document.getElementById("info-mode").textContent = image.mode;
    document.getElementById("info-preprocess").textContent = formatMs(processing.preprocess_time_ms);
    document.getElementById("info-segmentation").textContent = formatMs(processing.segmentation_time_ms);
    document.getElementById("info-classification").textContent = formatMs(processing.classification_time_ms);
    document.getElementById("info-total").textContent = formatMs(processing.total_time_ms);
    document.getElementById("info-segmenter").textContent = model.segmenter;
    document.getElementById("info-classifier").textContent = model.classifier;
  }

  function renderLeafDetail(leaf) {
    const probabilities = leaf.probabilities || {};
    const rows = Object.entries(probabilities)
      .map(([name, value]) => {
        const pct = Math.max(0, Math.min(100, Number(value) * 100));
        return `
          <div class="prob-row">
            <div>${name}</div>
            <div class="progress"><div class="progress-bar" style="width:${pct}%"></div></div>
            <div class="text-end">${pct.toFixed(1)}%</div>
          </div>`;
      })
      .join("");

    leafDetail.innerHTML = `
      <div class="leaf-detail-grid">
        <div>
          ${leaf.crop ? `<img class="leaf-crop" src="${leaf.crop}" alt="Leaf ${leaf.leaf_id} crop">` : ""}
        </div>
        <div>
          <div class="mb-2"><strong>Leaf #${leaf.leaf_id}</strong></div>
          <div class="mb-1">Prediction: <strong>${leaf.prediction}</strong></div>
          <div class="mb-3">Confidence: <strong>${formatPct(leaf.confidence)}</strong></div>
          <div class="mb-2 text-secondary">Probabilities</div>
          ${rows || "<div class='text-secondary'>No probabilities</div>"}
        </div>
      </div>`;
  }

  function renderLeaves(result) {
    leafList.innerHTML = "";
    result.leaves.forEach((leaf, index) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "leaf-item text-start";
      item.innerHTML = `
        <div class="title">Leaf #${leaf.leaf_id}</div>
        <div class="meta">${leaf.prediction} · ${formatPct(leaf.confidence)}</div>`;
      item.addEventListener("click", () => {
        document.querySelectorAll(".leaf-item").forEach((node) => node.classList.remove("active"));
        item.classList.add("active");
        renderLeafDetail(leaf);
      });
      leafList.appendChild(item);
      if (index === 0) {
        item.classList.add("active");
        renderLeafDetail(leaf);
      }
    });
    if (!result.leaves.length) {
      leafDetail.textContent = "Không phát hiện lá nào.";
    }
  }

  function showResult(result) {
    currentResult = result;
    results.classList.remove("d-none");
    mainImage.src = result.visualizations[currentView] || result.visualizations.overlay;
    renderSummary(result);
    renderLeaves(result);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selectedFile) return;
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";
    try {
      const body = new FormData();
      body.append("image", selectedFile);
      body.append("allow_single_leaf_fallback", "true");
      const response = await fetch("/predict", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed");
      }
      showResult(payload);
    } catch (error) {
      alert(error.message || String(error));
    } finally {
      predictBtn.disabled = !selectedFile;
      predictBtn.textContent = "Predict";
    }
  });

  async function loadHistoryIfRequested() {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");
    if (!id) return;
    const response = await fetch(`/history/${id}`);
    if (!response.ok) return;
    showResult(await response.json());
  }

  loadHistoryIfRequested();
})();
