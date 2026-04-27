import {
    initViewer,
    loadMesh,
    loadPointCloud,
    resetCamera,
    setLowQuality,
    isLowQuality,
    enterEditMode,
    exitEditMode,
    isEditActive,
    deleteSelected,
    clearSelection,
    getSelectionCount,
    exportCleanedPLY,
    isGeometryDirty,
    clearDirtyFlag,
    getCurrentViewType,
} from "./viewer.js";

import {
    onJobComplete as samplesOnJobComplete,
    loadSamples,
    hideSaveBar,
    saveFileToSample,
} from "./samples.js";

import {
    initDashboard,
    loadDashboard,
} from "./dashboard.js";

// --- Tab navigation ---
const tabBtns = document.querySelectorAll(".tab-btn");
const tabContentDashboard = document.getElementById("tab-content-dashboard");
const tabContentScan = document.getElementById("tab-content-scan");
const tabContentSamples = document.getElementById("tab-content-samples");

const allTabs = { dashboard: tabContentDashboard, scan: tabContentScan, samples: tabContentSamples };

tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
        tabBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");

        const tab = btn.dataset.tab;
        Object.entries(allTabs).forEach(([key, el]) => {
            el.classList.toggle("hidden", key !== tab);
        });

        if (tab === "dashboard") {
            loadDashboard();
        }
        if (tab === "scan") {
            refreshPlyLoaderOptions();
        }
        if (tab === "samples") {
            loadSamples();
            if (currentJobId && currentResult) {
                samplesOnJobComplete(currentJobId);
            }
        }
    });
});

// --- Initialize Dashboard on load ---
initDashboard(document.getElementById("dashboard-container"));
loadDashboard();

// Handle "View" from dashboard - navigate to viewer with sample files
window._onDashboardViewSample = (sample) => {
    if (!sample.files || !sample.files.length) {
        alert("This sample has no output files yet.");
        return;
    }

    // Build result URLs from sample files (prefer original outputs)
    const result = {};
    const files = sample.files || [];

    // 1) Prefer exact original names first
    const meshOriginal = files.find(
        (f) => (f.filename || "").toLowerCase() === "mesh.ply"
    );
    const pointcloudOriginal = files.find(
        (f) => (f.filename || "").toLowerCase() === "pointcloud.ply"
    );
    if (meshOriginal) {
        result.mesh = `/api/samples/${sample.id}/files/${meshOriginal.filename}`;
    }
    if (pointcloudOriginal) {
        result.pointcloud = `/api/samples/${sample.id}/files/${pointcloudOriginal.filename}`;
    }

    // 2) Fallback (if original files are missing)
    // Prefer refined mesh only when mesh.ply doesn't exist.
    const meshRefined = files.find(
        (f) => (f.filename || "").toLowerCase() === "mesh_refined.ply"
    );
    if (!result.mesh && meshRefined) {
        result.mesh = `/api/samples/${sample.id}/files/${meshRefined.filename}`;
    }

    // Last-resort fallback by type/name contains
    for (const f of files) {
        const url = `/api/samples/${sample.id}/files/${f.filename}`;
        if (!result.mesh && f.filename.includes("mesh") && f.file_type === "mesh") {
            result.mesh = url;
        } else if (
            !result.pointcloud &&
            f.filename.includes("pointcloud") &&
            f.file_type === "mesh"
        ) {
            result.pointcloud = url;
        } else if (!result.mesh && f.file_type === "mesh") {
            result.mesh = url;
        }
    }

    if (!result.mesh && !result.pointcloud) {
        alert("No mesh or point cloud files found in this sample.");
        return;
    }

    currentResult = result;
    currentSampleId = sample.id;
    currentJobId = sample.job_id;

    // Switch to scan tab and show viewer
    tabBtns.forEach(b => b.classList.remove("active"));
    document.getElementById("tab-scan").classList.add("active");
    Object.entries(allTabs).forEach(([key, el]) => {
        el.classList.toggle("hidden", key !== "scan");
    });

    showSection(viewerSection);
    initViewer(document.getElementById("viewer-container"));

    if (result.mesh) {
        loadMesh(result.mesh);
        btnMesh.classList.add("active");
        btnPointcloud.classList.remove("active");
    } else if (result.pointcloud) {
        loadPointCloud(result.pointcloud);
        btnPointcloud.classList.add("active");
        btnMesh.classList.remove("active");
    }

    _exitEditUI();
};

// --- DOM elements ---
const uploadSection = document.getElementById("upload-section");
const progressSection = document.getElementById("progress-section");
const viewerSection = document.getElementById("viewer-section");
const errorSection = document.getElementById("error-section");

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

const progressStep = document.getElementById("progress-step");
const progressBar = document.getElementById("progress-bar");
const progressPct = document.getElementById("progress-pct");
const progressMessage = document.getElementById("progress-message");

const errorMessage = document.getElementById("error-message");
const btnRetry = document.getElementById("btn-retry");

const btnMesh = document.getElementById("btn-mesh");
const btnPointcloud = document.getElementById("btn-pointcloud");
const btnReset = document.getElementById("btn-reset");
const btnLowQuality = document.getElementById("btn-low-quality");
const btnEdit = document.getElementById("btn-edit");

// Edit toolbar
const editToolbar = document.getElementById("edit-toolbar");
const btnDelete = document.getElementById("btn-delete");
const btnClearSel = document.getElementById("btn-clear-sel");
const btnSaveEdit = document.getElementById("btn-save-edit");
const btnResetOriginal = document.getElementById("btn-reset-original");
const btnRefine = document.getElementById("btn-refine");
const btnExitEdit = document.getElementById("btn-exit-edit");
const selectionCount = document.getElementById("selection-count");
const hintAction = document.getElementById("hint-action");
const plyModeSample = document.getElementById("ply-mode-sample");
const plyModeFilename = document.getElementById("ply-mode-filename");
const plySampleSelect = document.getElementById("ply-sample-select");
const plyFileSelect = document.getElementById("ply-file-select");
const plyFilenameInput = document.getElementById("ply-filename-input");
const btnOpenPlySample = document.getElementById("btn-open-ply-sample");
const btnOpenPlyFilename = document.getElementById("btn-open-ply-filename");
const plyLoaderHint = document.getElementById("ply-loader-hint");
const plyModeRadios = document.querySelectorAll("input[name='ply-load-mode']");

let currentResult = null;
let currentJobId = null;
let currentSampleId = null;
let plySamplesCache = [];

// --- Section visibility ---
function showSection(section) {
    [uploadSection, progressSection, viewerSection, errorSection].forEach(
        (s) => s.classList.add("hidden")
    );
    section.classList.remove("hidden");
}

// --- PLY quick loader (Scan tab) ---
function _plyFilesFromSample(sample) {
    if (!sample?.files?.length) return [];
    return sample.files
        .filter((f) => (f.filename || "").toLowerCase().endsWith(".ply"))
        .map((f) => f.filename);
}

function _setPlyHint(message) {
    plyLoaderHint.textContent = message || "";
}

function _naturalSampleNameCompare(a, b) {
    const sa = String(a?.name || "").trim();
    const sb = String(b?.name || "").trim();
    const ma = sa.match(/^(.*?)(\d+)\s*$/);
    const mb = sb.match(/^(.*?)(\d+)\s*$/);
    if (!ma || !mb) {
        return sa.localeCompare(sb, undefined, { sensitivity: "base" });
    }
    const prefixCmp = (ma[1] || "").trim().localeCompare((mb[1] || "").trim(), undefined, {
        sensitivity: "base",
    });
    if (prefixCmp !== 0) return prefixCmp;
    const na = Number.parseInt(ma[2], 10);
    const nb = Number.parseInt(mb[2], 10);
    if (na !== nb) return na - nb;
    return sa.localeCompare(sb, undefined, { sensitivity: "base" });
}

function _setPlyOpenBusy(isBusy) {
    btnOpenPlySample.disabled = isBusy || !plyFileSelect.value;
    btnOpenPlyFilename.disabled = isBusy;
    if (isBusy) {
        btnOpenPlySample.textContent = "Opening...";
        btnOpenPlyFilename.textContent = "Opening...";
    } else {
        btnOpenPlySample.textContent = "Open PLY";
        btnOpenPlyFilename.textContent = "Find & Open";
    }
}

function _renderPlyFilesForSelectedSample() {
    const sampleId = plySampleSelect.value;
    const sample = plySamplesCache.find((s) => s.id === sampleId);
    const files = _plyFilesFromSample(sample);
    plyFileSelect.innerHTML = "";

    if (!files.length) {
        const emptyOpt = document.createElement("option");
        emptyOpt.value = "";
        emptyOpt.textContent = "No .ply file in this sample";
        plyFileSelect.appendChild(emptyOpt);
        btnOpenPlySample.disabled = true;
        return;
    }

    files.forEach((filename) => {
        const opt = document.createElement("option");
        opt.value = filename;
        opt.textContent = filename;
        plyFileSelect.appendChild(opt);
    });
    btnOpenPlySample.disabled = false;
}

async function refreshPlyLoaderOptions() {
    try {
        const res = await fetch("/api/samples");
        if (!res.ok) throw new Error("Failed to load samples");
        const samples = await res.json();
        plySamplesCache = samples.filter((s) => _plyFilesFromSample(s).length > 0);
        plySampleSelect.innerHTML = "";

        if (!plySamplesCache.length) {
            const emptyOpt = document.createElement("option");
            emptyOpt.value = "";
            emptyOpt.textContent = "No sample with .ply yet";
            plySampleSelect.appendChild(emptyOpt);
            plyFileSelect.innerHTML = "";
            const noFileOpt = document.createElement("option");
            noFileOpt.value = "";
            noFileOpt.textContent = "No file";
            plyFileSelect.appendChild(noFileOpt);
            btnOpenPlySample.disabled = true;
            _setPlyHint("Create or import a sample first to open PLY from Scan.");
            return;
        }

        plySamplesCache
            .sort(_naturalSampleNameCompare)
            .forEach((sample) => {
                const opt = document.createElement("option");
                opt.value = sample.id;
                opt.textContent = `${sample.name} (${_plyFilesFromSample(sample).length} ply)`;
                plySampleSelect.appendChild(opt);
            });

        _renderPlyFilesForSelectedSample();
        _setPlyHint("");
    } catch (err) {
        _setPlyHint(`Cannot load sample list: ${err.message}`);
    }
}

function _switchPlyMode(mode) {
    const useSampleMode = mode === "sample";
    plyModeSample.classList.toggle("hidden", !useSampleMode);
    plyModeFilename.classList.toggle("hidden", useSampleMode);
    _setPlyHint("");
}

async function _openPlyFromSampleSelection() {
    const sampleId = plySampleSelect.value;
    const filename = plyFileSelect.value;
    if (!sampleId || !filename) return;

    const sample = plySamplesCache.find((s) => s.id === sampleId);
    if (!sample) {
        _setPlyHint("Sample not found. Refreshing list...");
        await refreshPlyLoaderOptions();
        return;
    }
    _setPlyOpenBusy(true);

    const fileUrl = `/api/samples/${sampleId}/files/${encodeURIComponent(filename)}`;
    currentSampleId = sampleId;
    currentJobId = sample.job_id || null;
    currentResult = {};

    const lowerName = filename.toLowerCase();
    if (lowerName.includes("pointcloud")) {
        currentResult.pointcloud = fileUrl;
        showSection(viewerSection);
        initViewer(document.getElementById("viewer-container"));
        loadPointCloud(fileUrl);
        btnPointcloud.classList.add("active");
        btnMesh.classList.remove("active");
    } else {
        currentResult.mesh = fileUrl;
        showSection(viewerSection);
        initViewer(document.getElementById("viewer-container"));
        loadMesh(fileUrl);
        btnMesh.classList.add("active");
        btnPointcloud.classList.remove("active");
    }

    _exitEditUI();
    _setPlyHint(`Loading ${filename}... file lớn có thể mất 10-60 giây.`);
}

async function _openPlyByFilename() {
    const query = (plyFilenameInput.value || "").trim().toLowerCase();
    if (!query) {
        _setPlyHint("Please enter a filename.");
        return;
    }

    if (!plySamplesCache.length) {
        await refreshPlyLoaderOptions();
    }

    let matchedSample = null;
    let matchedFile = null;
    for (const sample of plySamplesCache) {
        const files = _plyFilesFromSample(sample);
        const exact = files.find((f) => f.toLowerCase() === query);
        const partial = files.find((f) => f.toLowerCase().includes(query));
        const hit = exact || partial;
        if (hit) {
            matchedSample = sample;
            matchedFile = hit;
            break;
        }
    }

    if (!matchedSample || !matchedFile) {
        _setPlyHint(`No file matched: ${query}`);
        return;
    }

    plySampleSelect.value = matchedSample.id;
    _renderPlyFilesForSelectedSample();
    plyFileSelect.value = matchedFile;
    await _openPlyFromSampleSelection();
}

window.addEventListener("viewer-load-start", () => {
    _setPlyOpenBusy(true);
});

window.addEventListener("viewer-load-success", (e) => {
    _setPlyOpenBusy(false);
    const url = e?.detail?.url || "";
    const file = _extractFilename(url).split("?")[0];
    _setPlyHint(`Loaded ${file} thành công.`);
});

window.addEventListener("viewer-load-error", (e) => {
    _setPlyOpenBusy(false);
    const msg = e?.detail?.message || "Không thể load file PLY.";
    _setPlyHint(`Load lỗi: ${msg}`);
    showError(msg);
});

// --- Upload ---
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});

plySampleSelect.addEventListener("change", _renderPlyFilesForSelectedSample);
btnOpenPlySample.addEventListener("click", _openPlyFromSampleSelection);
btnOpenPlyFilename.addEventListener("click", _openPlyByFilename);
plyFilenameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        _openPlyByFilename();
    }
});
plyModeRadios.forEach((radio) => {
    radio.addEventListener("change", () => _switchPlyMode(radio.value));
});

async function uploadFile(file) {
    if (!file.name.toLowerCase().endsWith(".zip")) {
        alert("Please select a .zip file.");
        return;
    }

    showSection(progressSection);
    updateProgress("Uploading...", 0, "Sending file to server...");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/upload", { method: "POST", body: formData });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Upload failed");
        }
        const data = await res.json();
        currentJobId = data.job_id;
        connectWebSocket(data.job_id);
    } catch (err) {
        showError(err.message);
    }
}

// --- Progress via WebSocket ---
function connectWebSocket(jobId, onComplete) {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${location.host}/ws/jobs/${jobId}`);

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.status === "complete") {
            ws.close();
            if (onComplete) {
                onComplete(data.result);
            } else {
                currentResult = data.result;
                onProcessingComplete();
            }
            return;
        }

        if (data.status === "error") {
            ws.close();
            showError(data.error || "Processing failed.");
            return;
        }

        updateProgress(data.step, data.progress, data.message);
    };

    ws.onerror = () => {
        ws.close();
        pollJobStatus(jobId, onComplete);
    };
}

function pollJobStatus(jobId, onComplete) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/api/jobs/${jobId}`);
            const data = await res.json();

            if (data.status === "complete") {
                clearInterval(interval);
                if (onComplete) {
                    onComplete(data.result);
                } else {
                    currentResult = data.result;
                    onProcessingComplete();
                }
                return;
            }
            if (data.status === "error") {
                clearInterval(interval);
                showError(data.error);
                return;
            }
            updateProgress(data.step, data.progress, data.message);
        } catch {
            // keep trying
        }
    }, 1000);
}

function updateProgress(step, progress, message) {
    const pct = Math.round(progress * 100);
    progressStep.textContent = step || "Processing...";
    progressBar.style.width = `${pct}%`;
    progressPct.textContent = `${pct}%`;
    if (message) progressMessage.textContent = message;
}

// --- Viewer ---
function onProcessingComplete() {
    showSection(viewerSection);
    initViewer(document.getElementById("viewer-container"));

    if (currentResult.mesh) {
        loadMesh(currentResult.mesh);
        btnMesh.classList.add("active");
        btnPointcloud.classList.remove("active");
    } else if (currentResult.pointcloud) {
        loadPointCloud(currentResult.pointcloud);
        btnPointcloud.classList.add("active");
        btnMesh.classList.remove("active");
    }

    _exitEditUI();

    // Enable "Save as Sample" in Samples tab
    samplesOnJobComplete(currentJobId);
}

// --- Toolbar: View ---
btnMesh.addEventListener("click", () => {
    if (!currentResult?.mesh) return;
    if (getCurrentViewType() === "mesh") return; // avoid redundant reload
    if (isGeometryDirty()) {
        if (!confirm("You have unsaved changes. Switch anyway? (Changes will be lost)")) return;
    }
    _exitEditUI();
    btnMesh.classList.add("active");
    btnPointcloud.classList.remove("active");
    loadMesh(currentResult.mesh);
});

btnPointcloud.addEventListener("click", () => {
    if (!currentResult?.pointcloud) return;
    if (getCurrentViewType() === "points") return; // avoid redundant reload
    if (isGeometryDirty()) {
        if (!confirm("You have unsaved changes. Switch anyway? (Changes will be lost)")) return;
    }
    _exitEditUI();
    btnPointcloud.classList.add("active");
    btnMesh.classList.remove("active");
    loadPointCloud(currentResult.pointcloud);
});

btnReset.addEventListener("click", resetCamera);

// --- Toolbar: Low Quality ---
btnLowQuality.addEventListener("click", () => {
    const nowLow = !isLowQuality();
    setLowQuality(nowLow);
    btnLowQuality.classList.toggle("active", nowLow);
});

// --- Toolbar: Edit Mode ---
btnEdit.addEventListener("click", () => {
    if (isEditActive()) return;
    enterEditMode();
    _enterEditUI();
});

btnExitEdit.addEventListener("click", () => {
    exitEditMode();
    _exitEditUI();
});

function _enterEditUI() {
    editToolbar.classList.remove("hidden");
    btnEdit.classList.add("active");
    hintAction.textContent = "select (box)";

    // Poll selection count
    _selectionPoller = setInterval(_updateSelCount, 200);
}

function _exitEditUI() {
    if (isEditActive()) exitEditMode();
    editToolbar.classList.add("hidden");
    btnEdit.classList.remove("active");
    hintAction.textContent = "rotate";
    if (_selectionPoller) {
        clearInterval(_selectionPoller);
        _selectionPoller = null;
    }
}

let _selectionPoller = null;

function _updateSelCount() {
    const n = getSelectionCount();
    selectionCount.textContent = `${n} selected`;
    btnDelete.disabled = n === 0;
    btnClearSel.disabled = n === 0;
}

// --- Toolbar: Edit Actions ---
btnDelete.addEventListener("click", () => {
    const n = deleteSelected();
    if (n > 0) _updateSelCount();
});

btnClearSel.addEventListener("click", () => {
    clearSelection();
    _updateSelCount();
});

// --- Toolbar: Save Changes ---
btnSaveEdit.addEventListener("click", async () => {
    if (!currentResult) return;

    const blob = exportCleanedPLY();
    if (!blob) {
        alert("No geometry to save.");
        return;
    }

    btnSaveEdit.disabled = true;
    btnSaveEdit.textContent = "Saving...";

    try {
        const viewType = getCurrentViewType(); // "mesh" | "points"
        const targetUrl =
            viewType === "points" ? currentResult?.pointcloud : currentResult?.mesh;
        const targetFilename = targetUrl ? _extractFilename(targetUrl) : null;
        if (!targetFilename) {
            throw new Error("Khong xac dinh duoc file dich de luu.");
        }

        // Determine the filename based on what we're currently viewing
        const currentUrl = currentResult.mesh || currentResult.pointcloud;
        const isSample = currentUrl && currentUrl.startsWith("/api/samples/");
        const isJob = currentUrl && currentUrl.startsWith("/api/files/");

        if (isSample && currentSampleId) {
            // Save to sample - use currently active view file (mesh or pointcloud)
            await saveFileToSample(currentSampleId, targetFilename, blob);
            const refreshedUrl = `/api/samples/${currentSampleId}/files/${encodeURIComponent(
                targetFilename
            )}?v=${Date.now()}`;
            if (viewType === "points") {
                currentResult.pointcloud = refreshedUrl;
            } else {
                currentResult.mesh = refreshedUrl;
            }
            clearDirtyFlag();
            alert("Changes saved successfully!");
        } else if (isJob && currentJobId) {
            // For job outputs, save as a sample first if not already one
            const name = prompt("Save as sample with name:", `scan_${currentJobId.slice(0, 6)}`);
            if (!name) {
                return;
            }

            // Create sample from job, then overwrite the mesh file
            const createRes = await fetch("/api/samples", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, job_id: currentJobId }),
            });
            if (!createRes.ok) throw new Error("Failed to create sample");
            const sample = await createRes.json();
            currentSampleId = sample.id;

            // Now save the edited file
            await saveFileToSample(sample.id, targetFilename, blob);

            // Update URLs to point to the sample (keep each file mapped correctly)
            if (currentResult.mesh) {
                const meshFile = _extractFilename(currentResult.mesh).split("?")[0];
                currentResult.mesh = `/api/samples/${sample.id}/files/${encodeURIComponent(meshFile)}`;
            }
            if (currentResult.pointcloud) {
                const pcFile = _extractFilename(currentResult.pointcloud).split("?")[0];
                currentResult.pointcloud = `/api/samples/${sample.id}/files/${encodeURIComponent(pcFile)}`;
            }

            clearDirtyFlag();
            alert(`Saved as sample "${name}"!`);
        } else {
            alert("Cannot determine save location.");
        }
    } catch (err) {
        alert("Save failed: " + err.message);
    } finally {
        btnSaveEdit.disabled = false;
        btnSaveEdit.textContent = "Save Changes";
    }
});

btnResetOriginal.addEventListener("click", async () => {
    if (!currentSampleId) {
        alert("Reset chi ho tro khi dang mo Sample.");
        return;
    }
    if (!confirm("Reset ve du lieu goc? Ban chinh sua da luu tren sample se bi bo.")) {
        return;
    }

    btnResetOriginal.disabled = true;
    const oldText = btnResetOriginal.textContent;
    btnResetOriginal.textContent = "Resetting...";

    try {
        const res = await fetch(`/api/samples/${currentSampleId}/reset`, {
            method: "POST",
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Reset failed");
        }

        // Reload current viewer source; sample route will now fall back to source_path/output
        if (currentResult?.mesh) {
            loadMesh(currentResult.mesh);
            btnMesh.classList.add("active");
            btnPointcloud.classList.remove("active");
        } else if (currentResult?.pointcloud) {
            loadPointCloud(currentResult.pointcloud);
            btnPointcloud.classList.add("active");
            btnMesh.classList.remove("active");
        }
        clearDirtyFlag();
        alert("Da reset ve du lieu goc.");
    } catch (err) {
        alert("Reset failed: " + err.message);
    } finally {
        btnResetOriginal.disabled = false;
        btnResetOriginal.textContent = oldText;
    }
});

function _extractFilename(url) {
    const parts = url.split("/");
    return decodeURIComponent(parts[parts.length - 1]);
}

// --- Toolbar: Refine ---
btnRefine.addEventListener("click", async () => {
    if (!currentJobId) return;

    const blob = exportCleanedPLY();
    if (!blob) {
        alert("No geometry to refine.");
        return;
    }

    // Exit edit mode before refining
    exitEditMode();
    _exitEditUI();

    // Show progress
    showSection(progressSection);
    updateProgress("Uploading cleaned mesh...", 0, "Sending to server...");

    try {
        const formData = new FormData();
        formData.append("file", blob, "cleaned.ply");
        formData.append("job_id", currentJobId);

        const res = await fetch("/api/refine", {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Refine request failed");
        }

        const data = await res.json();

        // Connect to WebSocket for refine progress
        connectWebSocket(data.job_id, async (result) => {
            // Refine complete — update currentResult and show viewer
            if (result.mesh) {
                currentResult.refined_mesh = result.mesh;
                currentResult.mesh = result.mesh;
            }
            showSection(viewerSection);
            initViewer(document.getElementById("viewer-container"));
            loadMesh(result.mesh);
            btnMesh.classList.add("active");
            btnPointcloud.classList.remove("active");

            // Auto-save refined mesh to sample if working in sample context
            if (currentSampleId && result.mesh) {
                try {
                    const meshRes = await fetch(result.mesh);
                    const meshBlob = await meshRes.blob();
                    await saveFileToSample(currentSampleId, "mesh_refined.ply", meshBlob);
                } catch (err) {
                    console.warn("Auto-save to sample failed:", err);
                }
            }
        });
    } catch (err) {
        showError(err.message);
    }
});

// --- Error ---
function showError(msg) {
    errorMessage.textContent = msg;
    showSection(errorSection);
}

btnRetry.addEventListener("click", () => {
    fileInput.value = "";
    currentResult = null;
    currentJobId = null;
    hideSaveBar();
    showSection(uploadSection);
});

// --- Initial state: hide scan tab content (dashboard is default) ---
tabContentScan.classList.add("hidden");
tabContentSamples.classList.add("hidden");
_switchPlyMode("sample");
refreshPlyLoaderOptions();
