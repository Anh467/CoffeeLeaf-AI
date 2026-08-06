(() => {
  const LABEL_COLORS = {
    healthy: "#40b44b",
    miner: "#ffb300",
    rust: "#e15759",
    phoma: "#744eaa",
    "multi-disease": "#1e88e5",
    unknown: "#505050",
  };

  const homeState = document.getElementById("home-state");
  const loadingState = document.getElementById("loading-state");
  const errorState = document.getElementById("error-state");
  const results = document.getElementById("results");
  const form = document.getElementById("predict-form");
  const input = document.getElementById("image-input");
  const dropZone = document.getElementById("drop-zone");
  const fileName = document.getElementById("file-name");
  const filePreviewWrap = document.getElementById("file-preview-wrap");
  const filePreview = document.getElementById("file-preview");
  const predictBtn = document.getElementById("predict-btn");
  const resultMessage = document.getElementById("result-message");
  const resultFilename = document.getElementById("result-filename");
  const resultBadge = document.getElementById("result-badge");
  const scopeBadge = document.getElementById("scope-badge");
  const mainImage = document.getElementById("main-image");
  const viewerFrame = document.getElementById("viewer-frame");
  const bboxHighlight = document.getElementById("bbox-highlight");
  const leafList = document.getElementById("leaf-list");
  const leafDetail = document.getElementById("leaf-detail");
  const leafPanelTitle = document.getElementById("leaf-panel-title");
  const detailPanelTitle = document.getElementById("detail-panel-title");
  const errorMessage = document.getElementById("error-message");
  const metricsModal = document.getElementById("metrics-modal");

  let selectedFile = null;
  let currentResult = null;
  let selectedLeafId = null;
  let currentImageView = "overlay";
  let previewObjectUrl = null;
  let isPredicting = false;

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
    return null;
  }

  function diseaseCount(summary, name) {
    const counts = summary.disease_counts || {};
    if (Object.prototype.hasOwnProperty.call(counts, name)) {
      return Number(counts[name] || 0);
    }
    return Number((summary.class_counts || {})[name] || 0);
  }

  function isFullImageScope(result) {
    const processing = result?.processing || {};
    return (
      processing.analysis_scope === "full_image" ||
      processing.fallback_to_single_leaf === true ||
      Boolean(result?.full_image_result)
    );
  }

  function setActiveView(view) {
    currentImageView = view;
    document.querySelectorAll("[data-view]").forEach((node) => {
      node.classList.toggle("active", node.getAttribute("data-view") === view);
    });
  }

  function updateSegmentationTabs(enabled) {
    document.querySelectorAll("[data-seg-tab]").forEach((button) => {
      button.disabled = !enabled;
      button.classList.toggle("d-none", !enabled);
    });
    if (!enabled && (currentImageView === "mask_overlay" || currentImageView === "boxes")) {
      setActiveView("original");
    }
  }

  function clearPreview() {
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }
    filePreview.removeAttribute("src");
    filePreviewWrap.classList.add("d-none");
  }

  function resetApplication() {
    currentResult = null;
    selectedLeafId = null;
    selectedFile = null;
    isPredicting = false;
    currentImageView = "overlay";
    fileName.textContent = "";
    clearPreview();
    if (input) input.value = "";
    predictBtn.disabled = true;
    predictBtn.textContent = "Bắt đầu phân tích";
    mainImage.removeAttribute("src");
    mainImage.onload = null;
    leafList.innerHTML = "";
    leafDetail.textContent = "Chọn một lá ở danh sách để xem chi tiết.";
    bboxHighlight.classList.add("d-none");
    resultMessage.classList.add("d-none");
    resultMessage.textContent = "";
    scopeBadge.classList.add("d-none");
    updateSegmentationTabs(true);
    setActiveView("overlay");
    closeMetricsModal();
    showHomeState();
  }

  function showHomeState() {
    homeState.classList.remove("d-none");
    loadingState.classList.add("d-none");
    errorState.classList.add("d-none");
    results.classList.add("d-none");
  }

  function showLoadingState() {
    homeState.classList.add("d-none");
    loadingState.classList.remove("d-none");
    errorState.classList.add("d-none");
    results.classList.add("d-none");
  }

  function showErrorState(message) {
    homeState.classList.add("d-none");
    loadingState.classList.add("d-none");
    results.classList.add("d-none");
    errorState.classList.remove("d-none");
    errorMessage.textContent = message || "Đã xảy ra lỗi không xác định.";
  }

  function showResultState(result) {
    homeState.classList.add("d-none");
    loadingState.classList.add("d-none");
    errorState.classList.add("d-none");
    results.classList.remove("d-none");
    currentResult = result;
    resultFilename.textContent = result.image?.filename || "—";
    resultBadge.textContent = "Thành công";
    if (result.message) {
      resultMessage.textContent = result.message;
      resultMessage.classList.remove("d-none");
    } else {
      resultMessage.classList.add("d-none");
    }
    renderSummary(result);
    if (isFullImageScope(result)) {
      scopeBadge.classList.remove("d-none");
      updateSegmentationTabs(false);
      setActiveView("original");
      renderFullImageFallback(result);
    } else {
      scopeBadge.classList.add("d-none");
      updateSegmentationTabs(true);
      setActiveView("overlay");
      renderLeaves(result);
    }
    updateVisualization(currentImageView);
  }

  function renderSummary(result) {
    const { summary, processing } = result;
    document.getElementById("stat-total").textContent = summary.total_leaves;
    document.getElementById("stat-healthy").textContent = summary.healthy_leaves;
    document.getElementById("stat-diseased").textContent = summary.diseased_leaves;
    document.getElementById("stat-confidence").textContent = formatPct(summary.average_confidence);
    document.getElementById("stat-time").textContent = formatMs(processing.total_time_ms);
    document.getElementById("sum-miner").textContent = diseaseCount(summary, "miner");
    document.getElementById("sum-rust").textContent = diseaseCount(summary, "rust");
    document.getElementById("sum-phoma").textContent = diseaseCount(summary, "phoma");
    document.getElementById("sum-multi").textContent = summary.multi_disease_leaves ?? 0;
  }

  function renderProbabilityRows(probabilities) {
    return Object.entries(probabilities || {})
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
  }

  function renderLeafDetail(leaf) {
    selectedLeafId = leaf.leaf_id;
    const box = bboxOf(leaf);
    leafDetail.innerHTML = `
      <div class="leaf-detail-grid">
        <div>
          ${leaf.crop ? `<img class="leaf-crop" src="${leaf.crop}" alt="Leaf ${leaf.leaf_id} crop">` : ""}
        </div>
        <div>
          <div class="mb-2"><strong>Leaf #${leaf.leaf_id}</strong></div>
          <div class="mb-1">Prediction: <strong>${leafLabel(leaf)}</strong></div>
          <div class="mb-1">Labels: <strong>${(leaf.labels || []).join(", ") || "—"}</strong></div>
          <div class="mb-1">Classification confidence: <strong>${formatPct(leaf.classification_confidence ?? leaf.confidence)}</strong></div>
          <div class="mb-1">Segmentation confidence: <strong>${formatPct(leaf.segmentation_confidence ?? leaf.detector_confidence)}</strong></div>
          <div class="mb-3">Bounding box:
            <strong>${box ? `(${box.x1}, ${box.y1}, ${box.x2}, ${box.y2})` : "—"}</strong>
          </div>
          <div class="mb-2 text-secondary">Probabilities</div>
          ${renderProbabilityRows(leaf.probabilities) || "<div class='text-secondary'>No probabilities</div>"}
        </div>
      </div>`;
    updateHighlight();
  }

  function renderLeaves(result) {
    leafPanelTitle.textContent = "Danh sách lá";
    detailPanelTitle.textContent = "Chi tiết lá";
    leafList.innerHTML = "";
    selectedLeafId = null;
    result.leaves.forEach((leaf, index) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "leaf-item";
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

  function renderFullImageFallback(result) {
    leafPanelTitle.textContent = "Kết quả toàn ảnh";
    detailPanelTitle.textContent = "Kết quả toàn ảnh";
    leafList.innerHTML = "";
    selectedLeafId = null;
    bboxHighlight.classList.add("d-none");

    const full = result.full_image_result;
    if (!full) {
      leafDetail.textContent = "Không có kết quả phân loại toàn ảnh.";
      return;
    }
    const color = leafColor(full);
    const card = document.createElement("div");
    card.className = "leaf-item active";
    card.innerHTML = `
      <div class="title"><span class="swatch" style="background:${color}"></span>Kết quả toàn ảnh</div>
      <div class="meta">${leafLabel(full)} · ${formatPct(full.classification_confidence ?? full.confidence)}</div>`;
    leafList.appendChild(card);

    leafDetail.innerHTML = `
      <div class="leaf-detail-grid">
        <div>
          ${full.crop ? `<img class="leaf-crop" src="${full.crop}" alt="Full image crop">` : ""}
        </div>
        <div>
          <div class="mb-2"><strong>Kết quả toàn ảnh</strong></div>
          <div class="mb-1">Prediction: <strong>${leafLabel(full)}</strong></div>
          <div class="mb-1">Labels: <strong>${(full.labels || []).join(", ") || "—"}</strong></div>
          <div class="mb-3">Classification confidence:
            <strong>${formatPct(full.classification_confidence ?? full.confidence)}</strong>
          </div>
          <div class="mb-2 text-secondary">Probabilities</div>
          ${renderProbabilityRows(full.probabilities) || "<div class='text-secondary'>No probabilities</div>"}
        </div>
      </div>`;
  }

  function updateVisualization(view) {
    if (!currentResult) return;
    setActiveView(view);
    const viz =
      currentResult.visualizations[view] ||
      (view === "mask" ? currentResult.visualizations.mask_overlay : null) ||
      currentResult.visualizations.original ||
      currentResult.visualizations.overlay;
    mainImage.src = viz;
    mainImage.onload = () => updateHighlight();
  }

  function updateHighlight() {
    if (
      !currentResult ||
      selectedLeafId == null ||
      isFullImageScope(currentResult) ||
      !mainImage.naturalWidth
    ) {
      bboxHighlight.classList.add("d-none");
      return;
    }
    const leaf = (currentResult.leaves || []).find((item) => item.leaf_id === selectedLeafId);
    const box = bboxOf(leaf);
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
    bboxHighlight.style.borderColor = leafColor(leaf);
    bboxHighlight.classList.remove("d-none");
  }

  function fillMetricsModal(result) {
    const { image, processing, model } = result;
    document.getElementById("m-filename").textContent = image.filename || "—";
    document.getElementById("m-resolution").textContent = `${image.width} × ${image.height}`;
    document.getElementById("m-size").textContent = formatBytes(image.file_size);
    document.getElementById("m-format").textContent = image.format || "—";
    document.getElementById("m-raw").textContent = processing.raw_instances ?? "—";
    document.getElementById("m-valid").textContent = processing.valid_instances ?? "—";
    document.getElementById("m-fallback").textContent = processing.fallback_to_single_leaf
      ? "Yes"
      : "No";
    document.getElementById("m-scope").textContent =
      processing.analysis_scope === "full_image"
        ? "Phân tích toàn ảnh"
        : "Phát hiện từng lá";
    document.getElementById("m-det-th").textContent = formatPct(model.detector_confidence);
    document.getElementById("m-preprocess").textContent = formatMs(processing.preprocess_time_ms);
    document.getElementById("m-segmentation").textContent = formatMs(
      processing.segmentation_time_ms
    );
    document.getElementById("m-classification").textContent = formatMs(
      processing.classification_time_ms
    );
    document.getElementById("m-visualization").textContent = formatMs(
      processing.visualization_time_ms
    );
    document.getElementById("m-total").textContent = formatMs(processing.total_time_ms);
    document.getElementById("m-segmenter").textContent = model.segmenter || "—";
    document.getElementById("m-classifier").textContent = model.classifier || "—";
    document.getElementById("m-version").textContent = model.version || "—";
    document.getElementById("m-device").textContent = model.device || "—";
    document.getElementById("m-deploy").textContent = model.deploy_ready ? "Yes" : "No";
    document.getElementById("m-cls-th").textContent = formatPct(model.classifier_confidence);
  }

  function openMetricsModal() {
    if (!currentResult) return;
    fillMetricsModal(currentResult);
    metricsModal.classList.remove("d-none");
    document.body.classList.add("modal-open");
  }

  function closeMetricsModal() {
    metricsModal.classList.add("d-none");
    document.body.classList.remove("modal-open");
  }

  function setFile(file) {
    selectedFile = file;
    fileName.textContent = file ? file.name : "";
    predictBtn.disabled = !file || isPredicting;
    clearPreview();
    if (file) {
      previewObjectUrl = URL.createObjectURL(file);
      filePreview.src = previewObjectUrl;
      filePreviewWrap.classList.remove("d-none");
    }
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
      if (button.disabled) return;
      updateVisualization(button.getAttribute("data-view"));
    });
  });

  document.getElementById("analyze-another-btn").addEventListener("click", () => {
    resetApplication();
  });
  document.getElementById("error-retry-btn").addEventListener("click", () => {
    resetApplication();
  });
  document.getElementById("open-metrics-btn").addEventListener("click", openMetricsModal);
  document.getElementById("metrics-close-btn").addEventListener("click", closeMetricsModal);
  document.getElementById("metrics-close-x").addEventListener("click", closeMetricsModal);
  document.getElementById("metrics-backdrop").addEventListener("click", closeMetricsModal);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMetricsModal();
  });
  window.addEventListener("resize", updateHighlight);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selectedFile || isPredicting) return;
    isPredicting = true;
    predictBtn.disabled = true;
    predictBtn.textContent = "Đang phân tích…";
    currentResult = null;
    showLoadingState();
    try {
      const body = new FormData();
      body.append("file", selectedFile);
      const response = await fetch("/predict", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed");
      }
      showResultState(payload);
    } catch (error) {
      showErrorState(error.message || String(error));
    } finally {
      isPredicting = false;
      predictBtn.disabled = !selectedFile;
      predictBtn.textContent = "Bắt đầu phân tích";
    }
  });

  async function loadHistoryIfRequested() {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");
    if (!id) return;
    showLoadingState();
    try {
      const response = await fetch(`/history/${id}`);
      if (!response.ok) throw new Error("Không tải được lịch sử dự đoán.");
      showResultState(await response.json());
    } catch (error) {
      showErrorState(error.message || String(error));
    }
  }

  window.__coffeeLeafDashboard = {
    resetApplication,
    showHomeState,
    openMetricsModal,
    closeMetricsModal,
    getState: () => ({
      selectedFile: Boolean(selectedFile),
      currentResult,
      selectedLeafId,
      currentImageView,
      homeHidden: homeState.classList.contains("d-none"),
      resultsHidden: results.classList.contains("d-none"),
      hasModeSelect: Boolean(document.getElementById("predict-mode")),
      predictDisabled: predictBtn.disabled,
    }),
  };

  showHomeState();
  loadHistoryIfRequested();
})();
