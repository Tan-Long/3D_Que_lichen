/**
 * Samples tab – list, rename, delete samples and their files.
 */

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const samplesList = document.getElementById("samples-list");
const saveSampleBar = document.getElementById("save-sample-bar");
const sampleNameInput = document.getElementById("sample-name-input");
const btnSaveSample = document.getElementById("btn-save-sample");
const sortOrderBtn = document.getElementById("samples-sort-order");
const sortKeyButtons = Array.from(document.querySelectorAll(".samples-sort-key"));

let _currentJobId = null;
let _sortKey = "name";
let _sortAsc = true;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Call when a job finishes processing – enables "Save as Sample". */
export function onJobComplete(jobId) {
    _currentJobId = jobId;
    saveSampleBar.classList.remove("hidden");
    sampleNameInput.value = "";
    sampleNameInput.focus();
}

/** Hide the save bar (e.g. when switching tabs without a job). */
export function hideSaveBar() {
    saveSampleBar.classList.add("hidden");
}

/** Refresh the sample list from the server. */
export async function loadSamples() {
    try {
        const res = await fetch("/api/samples");
        const samples = await res.json();
        renderSamples(sortSamples(samples));
    } catch (err) {
        console.error("Failed to load samples:", err);
    }
}

/**
 * Save the current job's edited PLY into an existing sample.
 * Called from the main app when user saves after editing.
 */
export async function saveFileToSample(sampleId, filename, blob) {
    const formData = new FormData();
    formData.append("file", blob, filename);
    const res = await fetch(`/api/samples/${sampleId}/files`, {
        method: "POST",
        body: formData,
    });
    if (!res.ok) throw new Error("Failed to save file to sample");
    return res.json();
}

// ---------------------------------------------------------------------------
// Save as Sample
// ---------------------------------------------------------------------------

btnSaveSample.addEventListener("click", async () => {
    const name = sampleNameInput.value.trim();
    if (!name) {
        sampleNameInput.focus();
        return;
    }
    if (!_currentJobId) return;

    btnSaveSample.disabled = true;
    btnSaveSample.textContent = "Saving...";

    try {
        const res = await fetch("/api/samples", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, job_id: _currentJobId }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Save failed");
        }
        sampleNameInput.value = "";
        await loadSamples();
    } catch (err) {
        alert(err.message);
    } finally {
        btnSaveSample.disabled = false;
        btnSaveSample.textContent = "Save as Sample";
    }
});

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function renderSamples(samples) {
    if (!samples.length) {
        samplesList.innerHTML =
            '<p class="empty-hint">No samples saved yet. Process a scan and save it here.</p>';
        return;
    }

    samplesList.innerHTML = samples.map((s) => renderSampleCard(s)).join("");

    // Attach event listeners
    samplesList.querySelectorAll(".sample-header").forEach((hdr) => {
        hdr.addEventListener("click", (e) => {
            // Don't toggle when clicking action buttons
            if (e.target.closest(".sample-actions")) return;
            hdr.closest(".sample-card").classList.toggle("expanded");
        });
    });

    // Sample rename
    samplesList.querySelectorAll("[data-action='rename-sample']").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            startRenameSample(btn.dataset.id);
        });
    });

    // Sample delete
    samplesList.querySelectorAll("[data-action='delete-sample']").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            deleteSample(btn.dataset.id, btn.dataset.name);
        });
    });

    // File rename
    samplesList.querySelectorAll("[data-action='rename-file']").forEach((btn) => {
        btn.addEventListener("click", () => {
            startRenameFile(btn.dataset.sampleId, btn.dataset.filename);
        });
    });

    // File delete
    samplesList.querySelectorAll("[data-action='delete-file']").forEach((btn) => {
        btn.addEventListener("click", () => {
            deleteFile(btn.dataset.sampleId, btn.dataset.filename);
        });
    });
}

function renderSampleCard(sample) {
    const date = new Date(sample.created_at * 1000).toLocaleString();
    const fileCount = sample.files.length;

    const filesHtml = sample.files
        .map((f) => renderFileCard(sample.id, f))
        .join("");

    return `
    <div class="sample-card" data-sample-id="${sample.id}">
        <div class="sample-header">
            <span class="expand-icon">&#9654;</span>
            <span class="sample-name" data-sample-id="${sample.id}">${esc(sample.name)}</span>
            <span class="sample-meta">${fileCount} file${fileCount !== 1 ? "s" : ""} &middot; ${date}</span>
            <div class="sample-actions">
                <button class="btn-icon" data-action="rename-sample" data-id="${sample.id}" title="Rename">&#9998;</button>
                <button class="btn-icon danger" data-action="delete-sample" data-id="${sample.id}" data-name="${esc(sample.name)}" title="Delete">&#10005;</button>
            </div>
        </div>
        <div class="sample-files">
            <div class="files-grid">${filesHtml}</div>
        </div>
    </div>`;
}

function sortSamples(samples) {
    const sorted = [...samples];
    sorted.sort((a, b) => compareSample(a, b, _sortKey, _sortAsc));
    return sorted;
}

function compareSample(a, b, key, asc) {
    const direction = asc ? 1 : -1;
    let cmp = 0;

    if (key === "name") {
        cmp = naturalCompare(a.name || "", b.name || "");
    } else if (key === "status") {
        cmp = String(a.status || "").localeCompare(String(b.status || ""), undefined, { sensitivity: "base" });
        if (cmp === 0) cmp = naturalCompare(a.name || "", b.name || "");
    } else if (key === "files") {
        cmp = (a.files?.length || 0) - (b.files?.length || 0);
        if (cmp === 0) cmp = naturalCompare(a.name || "", b.name || "");
    } else if (key === "created") {
        cmp = (a.created_at || 0) - (b.created_at || 0);
        if (cmp === 0) cmp = naturalCompare(a.name || "", b.name || "");
    }

    return cmp * direction;
}

function naturalCompare(a, b) {
    const pa = parseNameParts(a);
    const pb = parseNameParts(b);

    // Compare prefix text first (case-insensitive)
    const prefixCmp = pa.prefix.localeCompare(pb.prefix, undefined, { sensitivity: "base" });
    if (prefixCmp !== 0) return prefixCmp;

    // Then compare numeric part (if present)
    if (pa.num !== null && pb.num !== null && pa.num !== pb.num) {
        return pa.num - pb.num;
    }
    if (pa.num !== null && pb.num === null) return -1;
    if (pa.num === null && pb.num !== null) return 1;

    // Fallback to plain text
    return String(a).localeCompare(String(b), undefined, { sensitivity: "base" });
}

function parseNameParts(value) {
    const s = String(value || "").trim();
    // Examples: 26Q1, Q12, sample-03
    const m = s.match(/^(.*?)(\d+)\s*$/);
    if (!m) {
        return { prefix: s, num: null };
    }
    return {
        prefix: (m[1] || "").trim(),
        num: Number.parseInt(m[2], 10),
    };
}

function renderFileCard(sampleId, file) {
    let preview;
    if (file.type === "image") {
        preview = `<img src="/api/samples/${sampleId}/files/${file.filename}" alt="${esc(file.filename)}" loading="lazy">`;
    } else if (file.type === "mesh") {
        preview = `<span class="file-icon">&#9651;</span>`;
    } else {
        preview = `<span class="file-icon">&#128196;</span>`;
    }

    const size = formatSize(file.size);

    return `
    <div class="file-card" data-filename="${esc(file.filename)}">
        <div class="file-preview">${preview}</div>
        <div class="file-info">
            <div class="file-name" data-sample-id="${sampleId}" data-filename="${esc(file.filename)}">${esc(file.filename)}</div>
            <div class="file-size">${size}</div>
            <div class="file-actions">
                <button class="btn-icon" data-action="rename-file" data-sample-id="${sampleId}" data-filename="${esc(file.filename)}" title="Rename">&#9998;</button>
                <button class="btn-icon danger" data-action="delete-file" data-sample-id="${sampleId}" data-filename="${esc(file.filename)}" title="Delete">&#10005;</button>
            </div>
        </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

function startRenameSample(sampleId) {
    const card = samplesList.querySelector(`.sample-card[data-sample-id="${sampleId}"]`);
    const nameEl = card.querySelector(".sample-name");
    const currentName = nameEl.textContent;

    const input = document.createElement("input");
    input.type = "text";
    input.className = "sample-name-edit";
    input.value = currentName;
    nameEl.replaceWith(input);
    input.focus();
    input.select();

    const commit = async () => {
        const newName = input.value.trim();
        if (!newName || newName === currentName) {
            // Revert
            const span = document.createElement("span");
            span.className = "sample-name";
            span.dataset.sampleId = sampleId;
            span.textContent = currentName;
            input.replaceWith(span);
            return;
        }
        try {
            await fetch(`/api/samples/${sampleId}/rename`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: newName }),
            });
            await loadSamples();
        } catch (err) {
            alert("Rename failed: " + err.message);
        }
    };

    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") input.blur();
        if (e.key === "Escape") {
            input.value = currentName;
            input.blur();
        }
    });
}

async function deleteSample(sampleId, name) {
    if (!confirm(`Delete sample "${name}" and all its files?`)) return;
    try {
        await fetch(`/api/samples/${sampleId}`, { method: "DELETE" });
        await loadSamples();
    } catch (err) {
        alert("Delete failed: " + err.message);
    }
}

function startRenameFile(sampleId, filename) {
    const nameEl = samplesList.querySelector(
        `.file-name[data-sample-id="${sampleId}"][data-filename="${filename}"]`
    );
    if (!nameEl) return;

    const input = document.createElement("input");
    input.type = "text";
    input.className = "file-name-edit";
    input.value = filename;
    nameEl.replaceWith(input);
    input.focus();

    // Select only the name part (before extension)
    const dotIdx = filename.lastIndexOf(".");
    if (dotIdx > 0) {
        input.setSelectionRange(0, dotIdx);
    } else {
        input.select();
    }

    const commit = async () => {
        const newName = input.value.trim();
        if (!newName || newName === filename) {
            await loadSamples(); // revert
            return;
        }
        try {
            await fetch(`/api/samples/${sampleId}/files/${encodeURIComponent(filename)}/rename`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ new_filename: newName }),
            });
            await loadSamples();
        } catch (err) {
            alert("Rename failed: " + err.message);
        }
    };

    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") input.blur();
        if (e.key === "Escape") {
            input.value = filename;
            input.blur();
        }
    });
}

async function deleteFile(sampleId, filename) {
    if (!confirm(`Delete file "${filename}"?`)) return;
    try {
        await fetch(`/api/samples/${sampleId}/files/${encodeURIComponent(filename)}`, {
            method: "DELETE",
        });
        await loadSamples();
    } catch (err) {
        alert("Delete failed: " + err.message);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function esc(str) {
    const el = document.createElement("span");
    el.textContent = str;
    return el.innerHTML;
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function refreshSortControls() {
    sortKeyButtons.forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.sortKey === _sortKey);
    });
    if (sortOrderBtn) {
        sortOrderBtn.textContent = _sortAsc ? "↑" : "↓";
        sortOrderBtn.title = _sortAsc ? "Ascending" : "Descending";
    }
}

sortKeyButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
        const selectedKey = btn.dataset.sortKey;
        if (!selectedKey) return;
        if (_sortKey === selectedKey) {
            _sortAsc = !_sortAsc;
        } else {
            _sortKey = selectedKey;
            _sortAsc = true;
        }
        refreshSortControls();
        await loadSamples();
    });
});

if (sortOrderBtn) {
    sortOrderBtn.addEventListener("click", async () => {
        _sortAsc = !_sortAsc;
        refreshSortControls();
        await loadSamples();
    });
}

refreshSortControls();
