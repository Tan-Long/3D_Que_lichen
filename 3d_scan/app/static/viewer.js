import * as THREE from "three";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

let renderer, scene, camera, controls, container;
let currentObject = null;
let animFrameId = null;
let currentType = null; // 'mesh' | 'points'
let loadInProgress = false;

// Edit state
let editActive = false;
let selecting = false;
let selStart = { x: 0, y: 0 };
let selDiv = null;
let selectedSet = new Set();
let origColors = null; // Float32Array backup before highlight
let centerShift = new THREE.Vector3(); // reverse centering on export
let originalHadColors = false; // whether geometry originally had per-vertex colors
let tempColorsInjected = false; // whether we created a temporary color attribute

// Track if geometry was modified (dirty flag)
let geometryDirty = false;

const SEL_COLOR = [1.0, 0.15, 0.15];

// ===================== Core =====================

export function initViewer(el) {
    if (renderer) {
        if (el !== container) {
            el.appendChild(renderer.domElement);
            container = el;
            _onResize();
        }
        return;
    }
    container = el;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0d1117);
    container.appendChild(renderer.domElement);

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
    camera.position.set(0, 1, 2);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const d1 = new THREE.DirectionalLight(0xffffff, 0.8);
    d1.position.set(5, 10, 7);
    scene.add(d1);
    const d2 = new THREE.DirectionalLight(0xffffff, 0.3);
    d2.position.set(-5, -3, -5);
    scene.add(d2);

    scene.add(new THREE.GridHelper(4, 20, 0x30363d, 0x21262d));

    _onResize();
    window.addEventListener("resize", _onResize);
    _animate();
}

function _onResize() {
    if (!container || !renderer) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}

function _animate() {
    animFrameId = requestAnimationFrame(_animate);
    controls.update();
    renderer.render(scene, camera);
}

function _clearCurrent() {
    exitEditMode();
    if (currentObject) {
        scene.remove(currentObject);
        if (currentObject.geometry) currentObject.geometry.dispose();
        if (currentObject.material) {
            if (currentObject.material.map) currentObject.material.map.dispose();
            currentObject.material.dispose();
        }
        currentObject = null;
    }
    currentType = null;
    centerShift.set(0, 0, 0);
}

function _centerAndFit(geometry) {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);
    centerShift.copy(center);

    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * 1.5;

    camera.position.set(dist * 0.5, dist * 0.5, dist);
    controls.target.set(0, 0, 0);
    controls.update();
}

// ===================== Load =====================

export function loadMesh(url) {
    // Avoid concurrent/redundant loads that can leave colors/material in a bad state.
    if (loadInProgress && currentType === "mesh") return;
    loadInProgress = true;
    _clearCurrent();
    const loader = new PLYLoader();
    currentType = "mesh"; // set early so click spam won't trigger reload
    window.dispatchEvent(new CustomEvent("viewer-load-start", { detail: { type: "mesh", url } }));
    loader.load(
        url,
        (geometry) => {
        geometry.computeVertexNormals();
        _centerAndFit(geometry);

        const hasColor = geometry.hasAttribute("color");
        const material = new THREE.MeshStandardMaterial({
            vertexColors: hasColor,
            color: hasColor ? undefined : 0x58a6ff,
            side: THREE.DoubleSide,
            toneMapped: false,
        });

        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
        currentObject = mesh;
        loadInProgress = false;
        window.dispatchEvent(new CustomEvent("viewer-load-success", { detail: { type: "mesh", url } }));
        },
        undefined,
        (err) => {
            loadInProgress = false;
            window.dispatchEvent(
                new CustomEvent("viewer-load-error", {
                    detail: {
                        type: "mesh",
                        url,
                        message: err?.message || "Failed to load mesh.",
                    },
                })
            );
        }
    );
}

export function loadPointCloud(url) {
    if (loadInProgress && currentType === "points") return;
    loadInProgress = true;
    _clearCurrent();
    const loader = new PLYLoader();
    currentType = "points"; // set early so click spam won't trigger reload
    window.dispatchEvent(new CustomEvent("viewer-load-start", { detail: { type: "points", url } }));
    loader.load(
        url,
        (geometry) => {
        _centerAndFit(geometry);

        const hasColor = geometry.hasAttribute("color");
        const material = new THREE.PointsMaterial({
            size: 0.005,
            vertexColors: hasColor,
            color: hasColor ? undefined : 0xffffff, // default to white when no vertex colors
            sizeAttenuation: true,
            toneMapped: false,
        });

        const points = new THREE.Points(geometry, material);
        scene.add(points);
        currentObject = points;
        loadInProgress = false;
        window.dispatchEvent(new CustomEvent("viewer-load-success", { detail: { type: "points", url } }));
        },
        undefined,
        (err) => {
            loadInProgress = false;
            window.dispatchEvent(
                new CustomEvent("viewer-load-error", {
                    detail: {
                        type: "points",
                        url,
                        message: err?.message || "Failed to load point cloud.",
                    },
                })
            );
        }
    );
}

export function resetCamera() {
    if (!currentObject?.geometry) return;
    const geo = currentObject.geometry;
    geo.computeBoundingBox();
    const size = new THREE.Vector3();
    geo.boundingBox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * 1.5;
    camera.position.set(dist * 0.5, dist * 0.5, dist);
    controls.target.set(0, 0, 0);
    controls.update();
}

// ===================== Quality =====================

let lowQuality = false;

export function setLowQuality(enabled) {
    if (!renderer) return;
    lowQuality = enabled;
    if (enabled) {
        renderer.setPixelRatio(1);
        const w = container.clientWidth;
        const h = container.clientHeight;
        renderer.setSize(Math.floor(w * 0.5), Math.floor(h * 0.5), false);
        renderer.domElement.style.width = w + "px";
        renderer.domElement.style.height = h + "px";
        // Simpler material for mesh
        if (currentObject?.material && currentType === "mesh") {
            currentObject.material.flatShading = true;
            currentObject.material.needsUpdate = true;
        }
    } else {
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.domElement.style.width = "";
        renderer.domElement.style.height = "";
        _onResize();
        if (currentObject?.material && currentType === "mesh") {
            currentObject.material.flatShading = false;
            currentObject.material.needsUpdate = true;
        }
    }
}

export function isLowQuality() {
    return lowQuality;
}

export function getCurrentViewType() {
    return currentType;
}

// ===================== Edit Mode =====================

export function enterEditMode() {
    if (!currentObject || editActive) return;
    editActive = true;
    selectedSet.clear();

    const geo = currentObject.geometry;
    originalHadColors = geo.hasAttribute("color");
    tempColorsInjected = false;
    if (originalHadColors) {
        origColors = new Float32Array(geo.getAttribute("color").array);
    } else {
        const n = geo.getAttribute("position").count;
        const cols = new Float32Array(n * 3).fill(0.7);
        geo.setAttribute("color", new THREE.BufferAttribute(cols, 3));
        currentObject.material.vertexColors = true;
        currentObject.material.needsUpdate = true;
        origColors = new Float32Array(cols);
        tempColorsInjected = true;
    }

    selDiv = document.createElement("div");
    selDiv.className = "selection-box";
    selDiv.style.display = "none";
    container.appendChild(selDiv);

    container.addEventListener("pointerdown", _onPointerDown);
    container.addEventListener("pointermove", _onPointerMove);
    container.addEventListener("pointerup", _onPointerUp);
}

export function exitEditMode() {
    if (!editActive) return;
    editActive = false;
    selectedSet.clear();
    selecting = false;

    if (selDiv) {
        selDiv.remove();
        selDiv = null;
    }

    container?.removeEventListener("pointerdown", _onPointerDown);
    container?.removeEventListener("pointermove", _onPointerMove);
    container?.removeEventListener("pointerup", _onPointerUp);

    if (currentObject?.geometry) {
        const geo = currentObject.geometry;
        if (origColors && geo.hasAttribute("color")) {
            const ca = geo.getAttribute("color");
            ca.array.set(origColors);
            ca.needsUpdate = true;
        }
        // If we injected temporary colors for highlighting, remove them on exit
        if (!originalHadColors && tempColorsInjected) {
            geo.deleteAttribute?.("color");
            if (currentObject.material) {
                currentObject.material.vertexColors = false;
                currentObject.material.needsUpdate = true;
            }
        }
    }
    origColors = null;
    originalHadColors = false;
    tempColorsInjected = false;
    if (controls) controls.enabled = true;
}

export function isEditActive() {
    return editActive;
}

function _canvasPos(e) {
    const r = container.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function _onPointerDown(e) {
    if (!editActive || e.button !== 0) return;
    selecting = true;
    controls.enabled = false;
    selStart = _canvasPos(e);
    selDiv.style.left = selStart.x + "px";
    selDiv.style.top = selStart.y + "px";
    selDiv.style.width = "0";
    selDiv.style.height = "0";
    selDiv.style.display = "block";
    e.preventDefault();
}

function _onPointerMove(e) {
    if (!selecting) return;
    const p = _canvasPos(e);
    const l = Math.min(p.x, selStart.x);
    const t = Math.min(p.y, selStart.y);
    selDiv.style.left = l + "px";
    selDiv.style.top = t + "px";
    selDiv.style.width = Math.abs(p.x - selStart.x) + "px";
    selDiv.style.height = Math.abs(p.y - selStart.y) + "px";
}

function _onPointerUp(e) {
    if (!selecting) return;
    selecting = false;
    controls.enabled = true;
    selDiv.style.display = "none";

    const p = _canvasPos(e);
    const x0 = Math.min(selStart.x, p.x),
        x1 = Math.max(selStart.x, p.x);
    const y0 = Math.min(selStart.y, p.y),
        y1 = Math.max(selStart.y, p.y);

    if (x1 - x0 < 4 && y1 - y0 < 4) return;

    _selectInRect(x0, y0, x1, y1);
}

function _selectInRect(x0, y0, x1, y1) {
    if (!currentObject?.geometry) return;
    const pos = currentObject.geometry.getAttribute("position");
    const n = pos.count;
    const cW = container.clientWidth;
    const cH = container.clientHeight;

    const v = new THREE.Vector3();

    for (let i = 0; i < n; i++) {
        v.set(pos.array[i * 3], pos.array[i * 3 + 1], pos.array[i * 3 + 2]);
        v.applyMatrix4(currentObject.matrixWorld);
        v.project(camera);

        const sx = (v.x + 1) * 0.5 * cW;
        const sy = (1 - v.y) * 0.5 * cH;

        if (
            sx >= x0 &&
            sx <= x1 &&
            sy >= y0 &&
            sy <= y1 &&
            v.z > -1 &&
            v.z < 1
        ) {
            selectedSet.add(i);
        }
    }
    _highlightSelection();
}

function _highlightSelection() {
    if (!currentObject?.geometry || !origColors) return;
    const ca = currentObject.geometry.getAttribute("color");
    if (!ca) return;

    ca.array.set(origColors);
    for (const i of selectedSet) {
        ca.array[i * 3] = SEL_COLOR[0];
        ca.array[i * 3 + 1] = SEL_COLOR[1];
        ca.array[i * 3 + 2] = SEL_COLOR[2];
    }
    ca.needsUpdate = true;
}

export function clearSelection() {
    selectedSet.clear();
    if (origColors && currentObject?.geometry?.hasAttribute("color")) {
        const ca = currentObject.geometry.getAttribute("color");
        ca.array.set(origColors);
        ca.needsUpdate = true;
    }
}

export function getSelectionCount() {
    return selectedSet.size;
}

// ===================== Delete =====================

export function deleteSelected() {
    if (!currentObject?.geometry || selectedSet.size === 0) return 0;

    const geo = currentObject.geometry;
    const pos = geo.getAttribute("position");
    const n = pos.count;
    const deleted = selectedSet.size;

    if (currentType === "points") {
        const keep = [];
        for (let i = 0; i < n; i++) {
            if (!selectedSet.has(i)) keep.push(i);
        }

        const newPos = new Float32Array(keep.length * 3);
        const newCol = origColors ? new Float32Array(keep.length * 3) : null;

        for (let j = 0; j < keep.length; j++) {
            const i = keep[j];
            newPos[j * 3] = pos.array[i * 3];
            newPos[j * 3 + 1] = pos.array[i * 3 + 1];
            newPos[j * 3 + 2] = pos.array[i * 3 + 2];
            if (newCol) {
                newCol[j * 3] = origColors[i * 3];
                newCol[j * 3 + 1] = origColors[i * 3 + 1];
                newCol[j * 3 + 2] = origColors[i * 3 + 2];
            }
        }

        geo.setAttribute("position", new THREE.BufferAttribute(newPos, 3));
        if (newCol) {
            geo.setAttribute("color", new THREE.BufferAttribute(newCol, 3));
            origColors = new Float32Array(newCol);
        }
    } else if (currentType === "mesh") {
        const idx = geo.getIndex();
        if (idx) {
            // 1) Collect faces that do NOT touch any selected vertex
            const survivingFaces = [];
            for (let i = 0; i < idx.count; i += 3) {
                const a = idx.array[i],
                    b = idx.array[i + 1],
                    c = idx.array[i + 2];
                if (
                    !selectedSet.has(a) &&
                    !selectedSet.has(b) &&
                    !selectedSet.has(c)
                ) {
                    survivingFaces.push(a, b, c);
                }
            }

            // 2) Find which vertices are still used by surviving faces
            const usedSet = new Set(survivingFaces);
            const sortedUsed = Array.from(usedSet).sort((a, b) => a - b);
            const oldToNew = new Map();
            sortedUsed.forEach((old, nw) => oldToNew.set(old, nw));

            // 3) Rebuild compacted position + color arrays
            const nV = sortedUsed.length;
            const newPos = new Float32Array(nV * 3);
            const newCol = origColors
                ? new Float32Array(nV * 3)
                : null;
            const hasNormals = geo.hasAttribute("normal");
            const oldNormals = hasNormals
                ? geo.getAttribute("normal").array
                : null;
            const newNormals = hasNormals
                ? new Float32Array(nV * 3)
                : null;

            for (let j = 0; j < nV; j++) {
                const i = sortedUsed[j];
                newPos[j * 3] = pos.array[i * 3];
                newPos[j * 3 + 1] = pos.array[i * 3 + 1];
                newPos[j * 3 + 2] = pos.array[i * 3 + 2];
                if (newCol) {
                    newCol[j * 3] = origColors[i * 3];
                    newCol[j * 3 + 1] = origColors[i * 3 + 1];
                    newCol[j * 3 + 2] = origColors[i * 3 + 2];
                }
                if (newNormals) {
                    newNormals[j * 3] = oldNormals[i * 3];
                    newNormals[j * 3 + 1] = oldNormals[i * 3 + 1];
                    newNormals[j * 3 + 2] = oldNormals[i * 3 + 2];
                }
            }

            // 4) Remap face indices
            const newIdx = new Uint32Array(survivingFaces.length);
            for (let i = 0; i < survivingFaces.length; i++) {
                newIdx[i] = oldToNew.get(survivingFaces[i]);
            }

            // 5) Apply compacted geometry
            geo.setAttribute(
                "position",
                new THREE.BufferAttribute(newPos, 3)
            );
            if (newCol) {
                geo.setAttribute(
                    "color",
                    new THREE.BufferAttribute(newCol, 3)
                );
                origColors = new Float32Array(newCol);
            }
            if (newNormals) {
                geo.setAttribute(
                    "normal",
                    new THREE.BufferAttribute(newNormals, 3)
                );
            }
            geo.setIndex(
                new THREE.BufferAttribute(newIdx, 1)
            );
        }
    }

    selectedSet.clear();
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    geometryDirty = true;
    return deleted;
}

export function isGeometryDirty() {
    return geometryDirty;
}

export function clearDirtyFlag() {
    geometryDirty = false;
}

// ===================== PLY Export =====================

export function exportCleanedPLY() {
    if (!currentObject?.geometry) return null;

    const geo = currentObject.geometry;
    const pos = geo.getAttribute("position");
    const col = geo.hasAttribute("color") ? geo.getAttribute("color") : null;
    const idx = geo.getIndex();
    const isMesh = currentType === "mesh";
    // Only export colors if they originally existed; avoid persisting temporary highlight colors
    const colSrc = originalHadColors
        ? (origColors || (col ? col.array : null))
        : null;

    if (isMesh && idx) {
        // Compact mesh: only keep vertices referenced by remaining faces
        const usedSet = new Set();
        for (let i = 0; i < idx.count; i++) usedSet.add(idx.array[i]);

        const sortedUsed = Array.from(usedSet).sort((a, b) => a - b);
        const oldToNew = new Map();
        sortedUsed.forEach((old, nw) => oldToNew.set(old, nw));

        const nV = sortedUsed.length;
        const nF = idx.count / 3;

        const expPos = new Float32Array(nV * 3);
        const expCol = colSrc ? new Float32Array(nV * 3) : null;

        for (let j = 0; j < nV; j++) {
            const i = sortedUsed[j];
            expPos[j * 3] = pos.array[i * 3] + centerShift.x;
            expPos[j * 3 + 1] = pos.array[i * 3 + 1] + centerShift.y;
            expPos[j * 3 + 2] = pos.array[i * 3 + 2] + centerShift.z;
            if (expCol) {
                expCol[j * 3] = colSrc[i * 3];
                expCol[j * 3 + 1] = colSrc[i * 3 + 1];
                expCol[j * 3 + 2] = colSrc[i * 3 + 2];
            }
        }

        const expIdx = new Int32Array(idx.count);
        for (let i = 0; i < idx.count; i++) {
            expIdx[i] = oldToNew.get(idx.array[i]);
        }

        return _buildPLY(expPos, expCol, expIdx, nV, nF);
    } else {
        // Point cloud
        const n = pos.count;
        const expPos = new Float32Array(n * 3);
        const expCol = colSrc ? new Float32Array(n * 3) : null;

        for (let i = 0; i < n; i++) {
            expPos[i * 3] = pos.array[i * 3] + centerShift.x;
            expPos[i * 3 + 1] = pos.array[i * 3 + 1] + centerShift.y;
            expPos[i * 3 + 2] = pos.array[i * 3 + 2] + centerShift.z;
            if (expCol) {
                expCol[i * 3] = colSrc[i * 3];
                expCol[i * 3 + 1] = colSrc[i * 3 + 1];
                expCol[i * 3 + 2] = colSrc[i * 3 + 2];
            }
        }

        return _buildPLY(expPos, expCol, null, n, 0);
    }
}

function _buildPLY(positions, colors, indices, nVerts, nFaces) {
    let header = "ply\nformat binary_little_endian 1.0\n";
    header += `element vertex ${nVerts}\n`;
    header += "property float x\nproperty float y\nproperty float z\n";
    if (colors)
        header +=
            "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    if (nFaces > 0) {
        header += `element face ${nFaces}\n`;
        header += "property list uchar int vertex_indices\n";
    }
    header += "end_header\n";

    const hdr = new TextEncoder().encode(header);
    const vStride = 12 + (colors ? 3 : 0);
    const vBuf = new ArrayBuffer(nVerts * vStride);
    const vView = new DataView(vBuf);

    let off = 0;
    for (let i = 0; i < nVerts; i++) {
        vView.setFloat32(off, positions[i * 3], true);
        off += 4;
        vView.setFloat32(off, positions[i * 3 + 1], true);
        off += 4;
        vView.setFloat32(off, positions[i * 3 + 2], true);
        off += 4;
        if (colors) {
            const clamp = (v) => Math.round(Math.max(0, Math.min(1, v)) * 255);
            vView.setUint8(off++, clamp(colors[i * 3]));
            vView.setUint8(off++, clamp(colors[i * 3 + 1]));
            vView.setUint8(off++, clamp(colors[i * 3 + 2]));
        }
    }

    let fBuf = new ArrayBuffer(0);
    if (nFaces > 0 && indices) {
        fBuf = new ArrayBuffer(nFaces * 13);
        const fView = new DataView(fBuf);
        let fo = 0;
        for (let i = 0; i < nFaces; i++) {
            fView.setUint8(fo++, 3);
            fView.setInt32(fo, indices[i * 3], true);
            fo += 4;
            fView.setInt32(fo, indices[i * 3 + 1], true);
            fo += 4;
            fView.setInt32(fo, indices[i * 3 + 2], true);
            fo += 4;
        }
    }

    return new Blob([hdr, vBuf, fBuf], { type: "application/x-ply" });
}
