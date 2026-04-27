/**
 * Dashboard – overview of all samples with status, filtering, and bulk import.
 */

const STATUS_CONFIG = {
    unprocessed:   { label: "Unprocessed",   color: "#6e7681", bg: "#21262d" },
    rgb_extracted: { label: "RGB Extracted", color: "#d29922", bg: "#2d2a1d" },
    meshed:        { label: "Meshed",        color: "#58a6ff", bg: "#1a2233" },
    filtered:      { label: "Filtered",      color: "#a371f7", bg: "#271f3d" },
    labeled:       { label: "Labeled",       color: "#3fb950", bg: "#1b2e1f" },
    complete:      { label: "Complete",      color: "#238636", bg: "#1b2e1f" },
};

let allSamples = [];
let currentFilter = "all";
let searchQuery = "";
let sortKey = "name";
let sortAsc = true;

// DOM refs (set after init)
let dashboardEl = null;

export function initDashboard(container) {
    dashboardEl = container;
    dashboardEl.innerHTML = `
        <div class="dash-toolbar">
            <div class="dash-filters">
                <button class="dash-filter active" data-filter="all">All</button>
                <button class="dash-filter" data-filter="unprocessed">Unprocessed</button>
                <button class="dash-filter" data-filter="meshed">Meshed</button>
                <button class="dash-filter" data-filter="filtered">Filtered</button>
                <button class="dash-filter" data-filter="labeled">Labeled</button>
                <button class="dash-filter" data-filter="complete">Complete</button>
            </div>
            <input type="text" class="input dash-search" placeholder="Search samples...">
        </div>
        <div class="dash-summary"></div>
        <div class="dash-actions-bar">
            <button class="btn btn-primary" id="btn-bulk-import">Bulk Import</button>
            <span class="dash-count"></span>
        </div>
        <div class="dash-table-wrap">
            <table class="dash-table">
                <thead>
                    <tr>
                        <th><button class="dash-sort-btn active" data-sort="name">Name <span class="dash-sort-ind">↑</span></button></th>
                        <th><button class="dash-sort-btn" data-sort="status">Status <span class="dash-sort-ind"></span></button></th>
                        <th>Assigned To</th>
                        <th><button class="dash-sort-btn" data-sort="files">Files <span class="dash-sort-ind"></span></button></th>
                        <th><button class="dash-sort-btn" data-sort="created">Created <span class="dash-sort-ind"></span></button></th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="dash-tbody"></tbody>
            </table>
        </div>
        <div id="import-modal" class="modal hidden">
            <div class="modal-content">
                <h3>Bulk Import</h3>
                <p class="modal-desc">Import dataset directories from the server.</p>
                <div class="modal-field">
                    <label>Base directory path:</label>
                    <input type="text" id="import-path" class="input" placeholder="/path/to/datasets">
                </div>
                <div id="import-preview" class="import-preview hidden"></div>
                <div class="modal-actions">
                    <button class="btn" id="btn-import-preview">Preview</button>
                    <button class="btn btn-primary" id="btn-import-confirm" disabled>Import</button>
                    <button class="btn" id="btn-import-cancel">Cancel</button>
                </div>
            </div>
        </div>
    `;

    // Filter buttons
    dashboardEl.querySelectorAll(".dash-filter").forEach(btn => {
        btn.addEventListener("click", () => {
            dashboardEl.querySelectorAll(".dash-filter").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentFilter = btn.dataset.filter;
            renderTable();
        });
    });

    // Search
    const searchInput = dashboardEl.querySelector(".dash-search");
    searchInput.addEventListener("input", () => {
        searchQuery = searchInput.value.trim().toLowerCase();
        renderTable();
    });

    // Sort controls
    dashboardEl.querySelectorAll(".dash-sort-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const selected = btn.dataset.sort;
            if (!selected) return;
            if (sortKey === selected) {
                sortAsc = !sortAsc;
            } else {
                sortKey = selected;
                sortAsc = true;
            }
            updateSortIndicators();
            renderTable();
        });
    });
    updateSortIndicators();

    // Bulk import
    const importModal = dashboardEl.querySelector("#import-modal");
    dashboardEl.querySelector("#btn-bulk-import").addEventListener("click", () => {
        importModal.classList.remove("hidden");
    });
    dashboardEl.querySelector("#btn-import-cancel").addEventListener("click", () => {
        importModal.classList.add("hidden");
    });
    dashboardEl.querySelector("#btn-import-preview").addEventListener("click", onPreviewImport);
    dashboardEl.querySelector("#btn-import-confirm").addEventListener("click", onConfirmImport);
}

export async function loadDashboard() {
    if (!dashboardEl) return;
    try {
        const res = await fetch("/api/samples");
        allSamples = await res.json();
        renderSummary();
        renderTable();
    } catch (err) {
        console.error("Failed to load dashboard:", err);
    }
}

function renderSummary() {
    const summary = dashboardEl.querySelector(".dash-summary");
    const counts = {};
    for (const s of allSamples) {
        counts[s.status] = (counts[s.status] || 0) + 1;
    }

    summary.innerHTML = Object.entries(STATUS_CONFIG).map(([key, cfg]) => {
        const count = counts[key] || 0;
        return `<span class="dash-stat" style="background:${cfg.bg};color:${cfg.color}">
            ${cfg.label}: <strong>${count}</strong>
        </span>`;
    }).join("");

    dashboardEl.querySelector(".dash-count").textContent = `${allSamples.length} total samples`;
}

function renderTable() {
    const tbody = dashboardEl.querySelector("#dash-tbody");
    let filtered = allSamples;

    if (currentFilter !== "all") {
        filtered = filtered.filter(s => s.status === currentFilter);
    }
    if (searchQuery) {
        filtered = filtered.filter(s => s.name.toLowerCase().includes(searchQuery));
    }
    filtered = sortSamples(filtered);

    if (!filtered.length) {
        tbody.innerHTML = `<tr><td colspan="6" class="dash-empty">No samples found</td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.map(s => {
        const cfg = STATUS_CONFIG[s.status] || STATUS_CONFIG.unprocessed;
        const fileCount = s.files ? s.files.length : 0;
        const date = new Date(s.created_at * 1000).toLocaleDateString();
        const assignee = s.assigned_to || "-";

        return `<tr data-id="${s.id}">
            <td class="dash-name">${esc(s.name)}</td>
            <td><span class="status-badge" style="background:${cfg.bg};color:${cfg.color}">${cfg.label}</span></td>
            <td class="dash-assignee">${esc(assignee)}</td>
            <td>${fileCount}</td>
            <td class="dash-date">${date}</td>
            <td class="dash-row-actions">
                <button class="btn-icon" data-action="view" data-id="${s.id}" title="View">&#128065;</button>
            </td>
        </tr>`;
    }).join("");

    // Attach view action
    tbody.querySelectorAll("[data-action='view']").forEach(btn => {
        btn.addEventListener("click", () => {
            const sampleId = btn.dataset.id;
            const sample = allSamples.find(s => s.id === sampleId);
            if (sample && window._onDashboardViewSample) {
                window._onDashboardViewSample(sample);
            }
        });
    });
}

function sortSamples(samples) {
    const out = [...samples];
    out.sort((a, b) => {
        let cmp = 0;
        if (sortKey === "name") {
            cmp = naturalCompare(a.name || "", b.name || "");
        } else if (sortKey === "status") {
            cmp = String(a.status || "").localeCompare(String(b.status || ""), undefined, { sensitivity: "base" });
            if (cmp === 0) cmp = naturalCompare(a.name || "", b.name || "");
        } else if (sortKey === "files") {
            cmp = (a.files?.length || 0) - (b.files?.length || 0);
            if (cmp === 0) cmp = naturalCompare(a.name || "", b.name || "");
        } else if (sortKey === "created") {
            cmp = (a.created_at || 0) - (b.created_at || 0);
            if (cmp === 0) cmp = naturalCompare(a.name || "", b.name || "");
        }
        return sortAsc ? cmp : -cmp;
    });
    return out;
}

function updateSortIndicators() {
    dashboardEl.querySelectorAll(".dash-sort-btn").forEach((btn) => {
        const active = btn.dataset.sort === sortKey;
        btn.classList.toggle("active", active);
        const ind = btn.querySelector(".dash-sort-ind");
        if (!ind) return;
        ind.textContent = active ? (sortAsc ? "↑" : "↓") : "";
    });
}

function naturalCompare(a, b) {
    const pa = parseNameParts(a);
    const pb = parseNameParts(b);
    const prefixCmp = pa.prefix.localeCompare(pb.prefix, undefined, { sensitivity: "base" });
    if (prefixCmp !== 0) return prefixCmp;
    if (pa.num !== null && pb.num !== null && pa.num !== pb.num) return pa.num - pb.num;
    if (pa.num !== null && pb.num === null) return -1;
    if (pa.num === null && pb.num !== null) return 1;
    return String(a).localeCompare(String(b), undefined, { sensitivity: "base" });
}

function parseNameParts(value) {
    const s = String(value || "").trim();
    const m = s.match(/^(.*?)(\d+)\s*$/);
    if (!m) return { prefix: s, num: null };
    return { prefix: (m[1] || "").trim(), num: Number.parseInt(m[2], 10) };
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

async function onPreviewImport() {
    const pathInput = dashboardEl.querySelector("#import-path");
    const previewDiv = dashboardEl.querySelector("#import-preview");
    const confirmBtn = dashboardEl.querySelector("#btn-import-confirm");
    const basePath = pathInput.value.trim();

    if (!basePath) {
        alert("Please enter a directory path.");
        return;
    }

    try {
        const res = await fetch(`/api/import/preview?base_path=${encodeURIComponent(basePath)}`);
        const data = await res.json();

        if (!data.directories.length) {
            previewDiv.innerHTML = "<p>No valid datasets found in this directory.</p>";
            previewDiv.classList.remove("hidden");
            confirmBtn.disabled = true;
            return;
        }

        const newCount = data.directories.filter(d => !d.already_imported).length;
        previewDiv.innerHTML = `
            <p>Found <strong>${data.directories.length}</strong> datasets,
               <strong>${newCount}</strong> new to import:</p>
            <ul class="import-list">
                ${data.directories.map(d => `
                    <li class="${d.already_imported ? 'imported' : 'new'}">
                        ${esc(d.name)}
                        ${d.already_imported ? '<span class="tag">already imported</span>' : ''}
                        ${d.has_output ? '<span class="tag tag-ok">has output</span>' : ''}
                    </li>
                `).join("")}
            </ul>
        `;
        previewDiv.classList.remove("hidden");
        confirmBtn.disabled = newCount === 0;
    } catch (err) {
        previewDiv.innerHTML = `<p class="error-text">Error: ${esc(err.message)}</p>`;
        previewDiv.classList.remove("hidden");
        confirmBtn.disabled = true;
    }
}

async function onConfirmImport() {
    const pathInput = dashboardEl.querySelector("#import-path");
    const confirmBtn = dashboardEl.querySelector("#btn-import-confirm");
    const basePath = pathInput.value.trim();

    confirmBtn.disabled = true;
    confirmBtn.textContent = "Importing...";

    try {
        const res = await fetch("/api/import/bulk", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ base_path: basePath }),
        });
        const data = await res.json();
        alert(`Imported ${data.imported_count} samples successfully!`);
        dashboardEl.querySelector("#import-modal").classList.add("hidden");
        await loadDashboard();
    } catch (err) {
        alert("Import failed: " + err.message);
    } finally {
        confirmBtn.disabled = false;
        confirmBtn.textContent = "Import";
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
