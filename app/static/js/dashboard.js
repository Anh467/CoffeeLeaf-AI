(() => {
  const LABEL_COLORS = {
    healthy: "#40b44b",
    miner: "#ffb300",
    rust: "#e15759",
    phoma: "#744eaa",
    "multi-disease": "#1e88e5",
    unknown: "#505050",
  };

  const MODE_LABELS = {
    auto: "Auto detect leaves",
    whole_image: "Whole image segmentation",
    single_leaf: "Single leaf",
    tree: "Whole image segmentation",
  };

  const form = document.getElementById("predict-form");
  const input = document.getElementById("image-input");
  const dropZone = document.getElementById("drop-zone");
  const fileName = document.getElementById("file-name");
  const predictBtn = document.getElementById("predict-btn");
  const predictMode = document.getElementById("predict-mode");
  const results = document.getElementById("results");
  const resultMessage = document.getElementById("result-message");
  const viewerNote = document.getElementById("viewer-note");
  const mainImage = document.getElementById("main-image");
  const viewerFrame = document.getElementById("viewer-frame");
  const bboxHighlight = document.getElementById("bbox-highlight");
  const leafList = document.getElementById("leaf-list");
  const leafDetail = document.getElementById("leaf-detail");

  const SUMMARY_IDS = [
    "stat-total",
    "stat-diseased",
    "stat-confidence",
    "stat-time",
    "sum-total",
    "sum-healthy",
    "sum-diseased",
    "sum-miner",
    "sum-rust",
    "sum-phoma",
    "sum-multi",
    "sum-confidence",
  ];
  const INFO_IDS = [
    "info-resolution",
    "info-size",
    "info-requested-mode",
    "info-resolved-mode",
    "info-fallback",
    "info-raw-instances",
    "info-valid-instances",
    "info-preprocess",
    "info-segmentation",
    "info-classification",
    "info-total",
    "info-segmenter",
    "info-classifier",
  ];

  let selectedFile = null;
  let currentResult = null;
  let currentView = "overlay";
  let selectedLeaf = null;

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

  function modeLabel(value) {
    if (!value) return "—";
    return MODE_LABELS[value] || String(value);
  }

  function leafLabel(leaf) {
    return leaf.display_label || leaf.prediction || "unknown";
  }

  function leafColor(leaf) {
    const labels = (leaf.labels || []).filter((name) => name !== "healthy");
    if (!labels.length) return LABEL_COLORS.healthy;
    if (labels.length >= 2) return LABEL_COLORS["multi-disease"];
    return LABEL_COLORS[labels[0]] || LABEL_COLORS.unknown;
  }

  function bboxOf(leaf) {
    if (leaf?.bbox && typeof leaf.bbox === "object" && !Array.isArray(leaf.bbox)) {
      return {
        x1: Number(leaf.bbox.x1),
        y1: Number(leaf.bbox.y1),
        x2: Number(leaf.bbox.x2),
        y2: Number(leaf.bbox.y2),
      };
    }
    const values = leaf?.bbox_xyxy || leaf?.bbox || [];
    if (Array.isArray(values) && values.length === 4) {
      return {
        x1: Number(values[0]),
        y1: Number(values[1]),
        x2: Number(values[2]),
        y2: Number(values[3]),
      };
    }
    return null;
  }

  function diseaseCount(summary, name) {
    const counts = summary.disease_counts || {};
    if (Object.prototype.hasOwnProperty.call(counts, name)) {
      return Number(counts[name] || 0);
    }
    return Number((summary.class_counts || {})[name] || 0);
  }

  function isSingleLeafResult(result) {
    const processing = result?.processing || {};
    return (
      processing.resolved_mode === "single_leaf" ||
      processing.mode === "single_leaf" ||
      result?.image?.mode === "single_leaf"
    );
  }

  function setActiveView(view) {
    currentView = view;
    document.querySelectorAll("[data-view]").forEach((node) => {
      node.classList.toggle("active", node.getAttribute("data-view") === view);
    });
  }

  function updateSegmentationTabs(enabled) {
    document.querySelectorAll("[data-seg-tab]").forEach((button) => {
      button.disabled = !enabled;
      button.classList.toggle("d-none", !enabled);
    });
    if (!enabled && (currentView === "mask_overlay" || currentView === "boxes" || currentView === "mask")) {
      setActiveView("original");
    }
  }

  function resetPredictionUI() {
    currentResult = null;
    selectedLeaf = null;
    mainImage.removeAttribute("src");
    mainImage.onload = null;
    leafList.innerHTML = "";
    leafDetail.textContent = "Chọn một lá ở danh sách bên trái.";
    bboxHighlight.classList.add("d-none");
    results.classList.add("d-none");
    resultMessage.classList.add("d-none");
    resultMessage.textContent = "";
    if (viewerNote) {
      viewerNote.classList.add("d-none");
      viewerNote.textContent = "";
    }
    SUMMARY_IDS.forEach((id) => {
      const node = document.getElementById(id);
      if (node) node.textContent = "—";
    });
    INFO_IDS.forEach((id) => {
      const node = document.getElementById(id);
      if (node) node.textContent = "—";
    });
    setActiveView("overlay");
    updateSegmentationTabs(true);
    document.querySelectorAll(".leaf-item").forEach((node) => node.classList.remove("active"));
  }

  // Expose for unit-style checks in the browser console / tests.
  window.__coffeeLeafDashboard = {
    resetPredictionUI,
    getState: () => ({
      currentResult,
      selectedLeaf,
      currentView,
      mainImageSrc: mainImage.getAttribute("src"),
      resultsHidden: results.classList.contains("d-none"),
      leafCount: leafList.children.length,
    }),
  };

  function setFile(file) {
    resetPredictionUI();
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
  predictMode.addEventListener("change", () => {
    resetPredictionUI();
    const mode = predictMode.value || "auto";
    updateSegmentationTabs(mode !== "single_leaf");
    if (mode === "single_leaf") {
      setActiveView("original");
    }
  });

  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.disabled) return;
      setActiveView(button.getAttribute("data-view"));
      if (currentResult?.visualizations?.[currentView]) {
        mainImage.src = currentResult.visualizations[currentView];
      } else if (currentView === "mask" && currentResult?.visualizations?.mask_overlay) {
        mainImage.src = currentResult.visualizations.mask_overlay;
      }
      requestAnimationFrame(updateHighlight);
    });
  });

  function renderSummary(result) {
    const { summary, processing, image, model } = result;
    document.getElementById("stat-total").textContent = summary.total_leaves;
    document.getElementById("stat-diseased").textContent = summary.diseased_leaves;
    document.getElementById("stat-confidence").textContent = formatPct(summary.average_confidence);
    document.getElementById("stat-time").textContent = formatMs(processing.total_time_ms);

    document.getElementById("sum-total").textContent = summary.total_leaves;
    document.getElementById("sum-healthy").textContent = summary.healthy_leaves;
    document.getElementById("sum-diseased").textContent = summary.diseased_leaves;
    document.getElementById("sum-miner").textContent = diseaseCount(summary, "miner");
    document.getElementById("sum-rust").textContent = diseaseCount(summary, "rust");
    document.getElementById("sum-phoma").textContent = diseaseCount(summary, "phoma");
    document.getElementById("sum-multi").textContent = summary.multi_disease_leaves ?? 0;
    document.getElementById("sum-confidence").textContent = formatPct(summary.average_confidence);

    document.getElementById("info-resolution").textContent = `${image.width} × ${image.height}`;
    document.getElementById("info-size").textContent = formatBytes(image.file_size);
    document.getElementById("info-requested-mode").textContent = modeLabel(
      processing.requested_mode || predictMode.value
    );
    document.getElementById("info-resolved-mode").textContent = modeLabel(
      processing.resolved_mode || processing.mode || image.mode
    );
    document.getElementById("info-fallback").textContent =
      processing.fallback_to_single_leaf || summary.fallback_to_single_leaf ? "Yes" : "No";
    document.getElementById("info-raw-instances").textContent =
      processing.raw_instances ?? "—";
    document.getElementById("info-valid-instances").textContent =
      processing.valid_instances ?? summary.total_leaves;
    document.getElementById("info-preprocess").textContent = formatMs(processing.preprocess_time_ms);
    document.getElementById("info-segmentation").textContent = formatMs(processing.segmentation_time_ms);
    document.getElementById("info-classification").textContent = formatMs(processing.classification_time_ms);
    document.getElementById("info-total").textContent = formatMs(processing.total_time_ms);
    document.getElementById("info-segmenter").textContent = model.segmenter;
    document.getElementById("info-classifier").textContent = model.classifier;
  }

  function updateHighlight() {
    if (
      !selectedLeaf ||
      !currentResult ||
      !mainImage.naturalWidth ||
      isSingleLeafResult(currentResult)
    ) {
      bboxHighlight.classList.add("d-none");
      return;
    }
    const box = bboxOf(selectedLeaf);
    if (!box) {
      bboxHighlight.classList.add("d-none");
      return;
    }
    const frameRect = viewerFrame.getBoundingClientRect();
    const imageRect = mainImage.getBoundingClientRect();
    const scaleX = imageRect.width / currentResult.image.width;
    const scaleY = imageRect.height / currentResult.image.height;
    const offsetX = imageRect.left - frameRect.left;
    const offsetY = imageRect.top - frameRect.top;
    bboxHighlight.style.left = `${offsetX + box.x1 * scaleX}px`;
    bboxHighlight.style.top = `${offsetY + box.y1 * scaleY}px`;
    bboxHighlight.style.width = `${Math.max(0, (box.x2 - box.x1) * scaleX)}px`;
    bboxHighlight.style.height = `${Math.max(0, (box.y2 - box.y1) * scaleY)}px`;
    bboxHighlight.style.borderColor = leafColor(selectedLeaf);
    bboxHighlight.classList.remove("d-none");
  }

  function renderLeafDetail(leaf) {
    selectedLeaf = leaf;
    const probabilities = leaf.probabilities || {};
    const singleLeaf = isSingleLeafResult(currentResult);
    const box = singleLeaf ? null : bboxOf(leaf);
    const rows = Object.entries(probabilities)
      .map(([name, value]) => {
        const pct = Math.max(0, Math.min(100, Number(value) * 100));
        const color = LABEL_COLORS[name] || LABEL_COLORS.unknown;
        return `
          <div class="prob-row">
            <div class="prob-name"><span class="swatch" style="background:${color}"></span>${name}</div>
            <div class="progress"><div class="progress-bar" style="width:${pct}%;background:${color}"></div></div>
            <div class="text-end">${pct.toFixed(1)}%</div>
          </div>`;
      })
      .join("");

    const classificationConfidence =
      leaf.classification_confidence ?? leaf.confidence ?? 0;
    const segmentationConfidence =
      leaf.segmentation_confidence ?? leaf.detector_confidence ?? 0;

    leafDetail.innerHTML = `
      <div class="leaf-detail-grid">
        <div>
          ${leaf.crop ? `<img class="leaf-crop" src="${leaf.crop}" alt="Leaf ${leaf.leaf_id} crop">` : ""}
        </div>
        <div>
          <div class="mb-2"><strong>Leaf #${leaf.leaf_id}</strong></div>
          <div class="mb-1">Prediction: <strong>${leafLabel(leaf)}</strong></div>
          <div class="mb-1">Labels: <strong>${(leaf.labels || []).join(", ") || "—"}</strong></div>
          <div class="mb-1">Classification confidence: <strong>${formatPct(classificationConfidence)}</strong></div>
          <div class="mb-1">Segmentation confidence:
            <strong>${singleLeaf ? "n/a (single-leaf)" : formatPct(segmentationConfidence)}</strong>
          </div>
          <div class="mb-3">Bounding box:
            <strong>${
              box
                ? `(${box.x1}, ${box.y1}, ${box.x2}, ${box.y2})`
                : singleLeaf
                  ? "n/a (whole image classified)"
                  : "—"
            }</strong>
          </div>
          <div class="mb-2 text-secondary">Probabilities</div>
          ${rows || "<div class='text-secondary'>No probabilities</div>"}
        </div>
      </div>`;
    updateHighlight();
  }

  function renderLeaves(result) {
    leafList.innerHTML = "";
    selectedLeaf = null;
    result.leaves.forEach((leaf, index) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "leaf-item text-start";
      const color = leafColor(leaf);
      item.innerHTML = `
        <div class="title"><span class="swatch" style="background:${color}"></span>Leaf #${leaf.leaf_id}</div>
        <div class="meta">${leafLabel(leaf)} · ${formatPct(leaf.classification_confidence ?? leaf.confidence)}</div>`;
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
      bboxHighlight.classList.add("d-none");
    }
  }

  function showResult(result) {
    currentResult = result;
    results.classList.remove("d-none");
    if (result.message) {
      resultMessage.textContent = result.message;
      resultMessage.classList.remove("d-none");
    } else {
      resultMessage.classList.add("d-none");
    }

    const singleLeaf = isSingleLeafResult(result);
    updateSegmentationTabs(!singleLeaf);
    if (singleLeaf) {
      setActiveView("original");
      if (viewerNote) {
        viewerNote.textContent =
          "Marks visible in Original are part of the uploaded image, not model predictions.";
        viewerNote.classList.remove("d-none");
      }
    } else if (viewerNote) {
      viewerNote.classList.add("d-none");
      viewerNote.textContent = "";
    }

    const viz =
      result.visualizations[currentView] ||
      (currentView === "mask" ? result.visualizations.mask_overlay : null) ||
      result.visualizations.original ||
      result.visualizations.overlay;
    mainImage.src = viz;
    mainImage.onload = () => updateHighlight();
    renderSummary(result);
    renderLeaves(result);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selectedFile) return;
    resetPredictionUI();
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";
    try {
      const body = new FormData();
      body.append("image", selectedFile);
      body.append("mode", predictMode.value || "auto");
      const response = await fetch("/predict", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed");
      }
      showResult(payload);
    } catch (error) {
      resetPredictionUI();
      alert(error.message || String(error));
    } finally {
      predictBtn.disabled = !selectedFile;
      predictBtn.textContent = "Predict";
    }
  });

  window.addEventListener("resize", updateHighlight);

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
