import * as THREE from 'three'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { renderLichenChart } from '../charts/polar_lichen.js'
import { renderSolarChart } from '../charts/solar_chart.js'
import { renderWindRose } from '../charts/wind_rose.js'

// Global cleanup state
let _map = null
let _renderer = null
let _animFrame = null
let _controls = null
let _container = null

export function render(container, sampleId) {
  _container = container

  if (!sampleId) {
    container.innerHTML = `<div class="info-panel">No sample selected.</div>`
    return
  }

  // Skeleton layout
  container.innerHTML = `
    <a class="back-link" href="#samples" onclick="window.navigate('#samples');return false">← Back to samples</a>
    <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:20px;flex-wrap:wrap">
      <div class="view-title" style="margin-bottom:0" id="detail-title">
        <div class="skeleton" style="height:28px;width:160px"></div>
      </div>
      <span style="font-size:13px;color:var(--text-secondary)" id="detail-meta"></span>
    </div>

    <div class="grid-4" style="margin-bottom:16px" id="stat-cards">
      ${[0,1,2,3].map(() => `<div class="card stat-card"><div class="skeleton" style="height:44px"></div></div>`).join('')}
    </div>

    <div class="grid-2" style="margin-bottom:12px">
      <div class="card">
        <div class="panel-title">3D Mesh</div>
        <div class="mesh-viewer-container" id="mesh-viewer">
          <div class="viewer-overlay">Loading mesh...</div>
        </div>
        <div style="font-size:11px;color:var(--text-secondary);margin-top:4px;text-align:center">Drag: rotate · Scroll: zoom · Right-drag: pan</div>
      </div>
      <div class="card">
        <div class="panel-title">Lichen by Slope Side</div>
        <div class="chart-wrap" style="height:220px"><canvas id="chart-lichen"></canvas></div>
        <div id="lichen-ratio-text" style="font-size:12px;color:var(--text-secondary);margin-top:8px;text-align:center"></div>
      </div>
    </div>

    <div class="grid-2" style="margin-bottom:12px">
      <div class="card">
        <div class="panel-title">Solar Exposure</div>
        <div id="solar-badge" style="margin-bottom:8px"></div>
        <div class="chart-wrap" style="height:160px"><canvas id="chart-solar"></canvas></div>
      </div>
      <div class="card">
        <div class="panel-title">Wind Rose</div>
        <div class="chart-wrap" style="height:160px"><canvas id="chart-wind"></canvas></div>
        <div id="wind-meta" style="font-size:12px;color:var(--text-secondary);margin-top:8px"></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="panel-title">Location</div>
        <div id="map-container"></div>
      </div>
      <div class="card">
        <div class="panel-title">Slope &amp; Metadata</div>
        <div id="slope-panel"><div class="skeleton" style="height:120px"></div></div>
      </div>
    </div>
  `

  // Kick off all fetches concurrently
  Promise.allSettled([
    window.API.getSample(sampleId),
    window.API.getStats(sampleId),
    window.API.getSlope(sampleId),
    window.API.getSolar(sampleId),
    window.API.getLocation(sampleId),
    window.API.getWind(sampleId),
  ]).then(([sampleR, statsR, slopeR, solarR, locationR, windR]) => {
    if (!_container) return

    const sample   = sampleR.status === 'fulfilled'   ? sampleR.value   : null
    const stats    = statsR.status === 'fulfilled'    ? statsR.value    : null
    const slope    = slopeR.status === 'fulfilled'    ? slopeR.value    : null
    const solar    = solarR.status === 'fulfilled'    ? solarR.value    : null
    const location = locationR.status === 'fulfilled' ? locationR.value : null
    const wind     = windR.status === 'fulfilled'     ? windR.value     : null

    renderHeader(sampleId, sample, location)
    renderStatCards(sample, stats, slope, location)
    renderMeshViewer(sampleId)
    renderLichenPanel(stats)
    renderSolarPanel(solar)
    renderWindPanel(wind)
    renderMapPanel(location, sampleId)
    renderSlopePanel(slope)
  })
}

export function destroy() {
  // Cleanup Leaflet
  if (_map) { try { _map.remove() } catch (_) {} _map = null }

  // Cleanup Three.js
  if (_animFrame) { cancelAnimationFrame(_animFrame); _animFrame = null }
  if (_controls) { _controls.dispose(); _controls = null }
  if (_renderer) { _renderer.dispose(); _renderer = null }

  // Destroy Chart.js instances
  ;['_lichenChart','_solarChart','_windChart'].forEach(key => {
    if (window[key]) { try { window[key].destroy() } catch (_) {} window[key] = null }
  })

  _container = null
}

// ── Header ────────────────────────────────────────────────────────────────────

function renderHeader(sampleId, sample, location) {
  if (!_container) return
  const titleEl = _container.querySelector('#detail-title')
  const metaEl  = _container.querySelector('#detail-meta')
  if (!titleEl) return

  if (!sample) {
    titleEl.textContent = 'Sample not found'
    return
  }

  titleEl.textContent = sample.name || sampleId

  const parts = []
  if (sample.created_at) parts.push(formatDate(sample.created_at * 1000))
  if (location) {
    parts.push(`${fmtCoord(location.anchor_lat)}°N, ${fmtCoord(location.anchor_lon)}°E`)
  }
  if (metaEl) metaEl.textContent = parts.join(' · ')
}

// ── Summary cards ─────────────────────────────────────────────────────────────

function renderStatCards(sample, stats, slope, location) {
  if (!_container) return
  const el = _container.querySelector('#stat-cards')
  if (!el) return

  const lichenPct = stats?.overall?.lichen_ratio_pct ?? null
  const altitude  = location?.anchor_alt ?? null
  const dbh       = slope?.dbh_cm ?? null

  // Aspect from trunk_axis
  let aspect = '—'
  if (slope?.trunk_axis) {
    aspect = axisToCompass(slope.trunk_axis)
  }

  el.innerHTML = `
    ${statCard('Lichen Coverage', lichenPct != null ? fmtNum(lichenPct, 1) : '—', lichenPct != null ? '%' : '')}
    ${statCard('DBH', dbh != null ? fmtNum(dbh, 1) : '—', dbh != null ? 'cm' : '')}
    ${statCard('Altitude', altitude != null ? Math.round(altitude) : '—', altitude != null ? 'm a.s.l.' : '')}
    ${statCard('Aspect', aspect, '')}
  `
}

function statCard(label, value, unit) {
  return `
    <div class="card stat-card">
      <div class="stat-label">${label}</div>
      <div class="stat-value">${value}<span class="stat-unit"> ${unit}</span></div>
    </div>
  `
}

// ── 3D Mesh viewer ────────────────────────────────────────────────────────────

function renderMeshViewer(sampleId) {
  if (!_container) return
  const viewerEl = _container.querySelector('#mesh-viewer')
  if (!viewerEl) return

  const url = window.API.meshUrl(sampleId)

  // Cleanup previous renderer
  if (_animFrame) { cancelAnimationFrame(_animFrame); _animFrame = null }
  if (_controls) { _controls.dispose(); _controls = null }
  if (_renderer) { _renderer.dispose(); _renderer = null }

  viewerEl.innerHTML = '<div class="viewer-overlay">Loading mesh...</div>'

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setClearColor(0x0d1117)
  _renderer = renderer

  const scene = new THREE.Scene()
  scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const d1 = new THREE.DirectionalLight(0xffffff, 0.9)
  d1.position.set(5, 10, 7)
  scene.add(d1)
  const d2 = new THREE.DirectionalLight(0xffffff, 0.25)
  d2.position.set(-4, -3, -5)
  scene.add(d2)

  const w = viewerEl.clientWidth || 400
  const h = viewerEl.clientHeight || 280
  renderer.setSize(w, h)

  const camera = new THREE.PerspectiveCamera(55, w / h, 0.01, 200)
  camera.position.set(0, 0.5, 2)

  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  _controls = controls

  viewerEl.innerHTML = ''
  viewerEl.appendChild(renderer.domElement)

  const overlay = document.createElement('div')
  overlay.className = 'viewer-overlay'
  overlay.textContent = 'Loading mesh...'
  viewerEl.appendChild(overlay)

  const onResize = () => {
    if (!viewerEl.isConnected) return
    const nw = viewerEl.clientWidth
    const nh = viewerEl.clientHeight
    renderer.setSize(nw, nh)
    camera.aspect = nw / nh
    camera.updateProjectionMatrix()
  }
  window.addEventListener('resize', onResize)

  const animate = () => {
    _animFrame = requestAnimationFrame(animate)
    controls.update()
    renderer.render(scene, camera)
  }
  animate()

  const loader = new PLYLoader()
  loader.load(url,
    geometry => {
      if (!_renderer) return
      overlay.style.display = 'none'
      geometry.computeVertexNormals()

      // Center and fit
      geometry.computeBoundingBox()
      const box = geometry.boundingBox
      const center = new THREE.Vector3()
      box.getCenter(center)
      geometry.translate(-center.x, -center.y, -center.z)
      const size = new THREE.Vector3()
      box.getSize(size)
      const maxDim = Math.max(size.x, size.y, size.z)
      const dist = maxDim * 1.6
      camera.position.set(dist * 0.4, dist * 0.3, dist)
      controls.target.set(0, 0, 0)
      controls.update()

      const hasColor = geometry.hasAttribute('color')
      const mat = new THREE.MeshStandardMaterial({
        vertexColors: hasColor,
        color: hasColor ? undefined : 0x888888,
        side: THREE.DoubleSide,
        toneMapped: false,
      })
      scene.add(new THREE.Mesh(geometry, mat))
    },
    undefined,
    () => {
      if (!_renderer) return
      overlay.textContent = 'Mesh not available'
      overlay.style.display = 'flex'
    }
  )
}

// ── Lichen chart ──────────────────────────────────────────────────────────────

function renderLichenPanel(stats) {
  if (!_container) return
  const canvas = _container.querySelector('#chart-lichen')
  const textEl = _container.querySelector('#lichen-ratio-text')
  if (!canvas) return

  if (!stats) {
    const wrap = canvas.parentElement
    wrap.innerHTML = `<div class="info-panel">Lichen stats not available</div>`
    return
  }

  if (window._lichenChart) { window._lichenChart.destroy(); window._lichenChart = null }
  window._lichenChart = renderLichenChart(canvas, stats)

  if (textEl) {
    const up = stats.upslope?.lichen_ratio_pct
    const dn = stats.downslope?.lichen_ratio_pct
    textEl.textContent = `Upslope: ${fmtNum(up, 1)}% · Downslope: ${fmtNum(dn, 1)}%`
  }
}

// ── Solar chart ───────────────────────────────────────────────────────────────

function renderSolarPanel(solar) {
  if (!_container) return
  const canvas  = _container.querySelector('#chart-solar')
  const badgeEl = _container.querySelector('#solar-badge')
  if (!canvas) return

  if (!solar) {
    canvas.parentElement.innerHTML = `<div class="info-panel">Solar exposure data not available</div>`
    return
  }

  if (badgeEl && solar.solar_classification) {
    const isSun = solar.solar_classification.toLowerCase().includes('sun')
    badgeEl.innerHTML = `<span class="badge ${isSun ? 'badge-processing' : 'badge-queued'}">${esc(solar.solar_classification)}</span>`
  }

  if (window._solarChart) { window._solarChart.destroy(); window._solarChart = null }
  window._solarChart = renderSolarChart(canvas, solar)
}

// ── Wind rose ─────────────────────────────────────────────────────────────────

function renderWindPanel(wind) {
  if (!_container) return
  const canvas = _container.querySelector('#chart-wind')
  const metaEl = _container.querySelector('#wind-meta')
  if (!canvas) return

  if (!wind) {
    canvas.parentElement.innerHTML = `<div class="info-panel">Wind data unavailable — Open-Meteo unreachable</div>`
    return
  }

  if (window._windChart) { window._windChart.destroy(); window._windChart = null }
  window._windChart = renderWindRose(canvas, wind)

  if (metaEl) {
    const windDir = wind.dominant_wind_dir_cardinal || '—'
    const windDeg = wind.dominant_wind_dir_deg != null ? fmtNum(wind.dominant_wind_dir_deg, 1) + '°' : '—'
    const fraction = wind.wind_facing_fraction != null ? fmtNum(wind.wind_facing_fraction * 100, 1) + '%' : '—'
    const speed = wind.mean_wind_speed_kmh != null ? fmtNum(wind.mean_wind_speed_kmh, 1) + ' km/h' : '—'
    const windward = wind.wind_facing_fraction != null ? (wind.wind_facing_fraction > 0.5 ? 'Windward' : 'Leeward') : ''
    metaEl.innerHTML = `
      Dominant: <strong>${windDir} (${windDeg})</strong> ·
      Wind-facing: ${fraction}${windward ? ' → ' + windward : ''} ·
      Mean speed: ${speed}
    `
  }
}

// ── Map ───────────────────────────────────────────────────────────────────────

function renderMapPanel(location, sampleId) {
  if (!_container) return
  const mapEl = _container.querySelector('#map-container')
  if (!mapEl) return

  if (!location || location.anchor_lat == null || location.anchor_lon == null) {
    mapEl.outerHTML = `<div class="info-panel" style="min-height:120px;display:flex;align-items:center;justify-content:center">Location data not available</div>`
    return
  }

  if (_map) { try { _map.remove() } catch (_) {} _map = null }

  const lat = location.anchor_lat
  const lon = location.anchor_lon
  const alt = location.anchor_alt || 0
  const acc = location.anchor_accuracy_m || 0

  // Needs to be visible first for Leaflet to init correctly
  requestAnimationFrame(() => {
    if (!_container || !mapEl.isConnected) return
    try {
      _map = L.map(mapEl, { zoomControl: true }).setView([lat, lon], 16)
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        maxZoom: 19,
      }).addTo(_map)

      const marker = L.circleMarker([lat, lon], {
        radius: 8, color: '#185FA5', fillColor: '#185FA5',
        fillOpacity: 0.9, weight: 2,
      }).addTo(_map)
      marker.bindPopup(`<b>${esc(sampleId)}</b><br>${fmtCoord(lat)}°N, ${fmtCoord(lon)}°E<br>${Math.round(alt)}m a.s.l. ± ${Math.round(acc)}m`).openPopup()

      if (acc > 0) {
        L.circle([lat, lon], {
          radius: acc, color: '#185FA5',
          fillColor: '#185FA5', fillOpacity: 0.08, weight: 1,
        }).addTo(_map)
      }
    } catch (err) {
      mapEl.innerHTML = `<div class="info-panel">Map failed to load</div>`
    }
  })
}

// ── Slope panel ───────────────────────────────────────────────────────────────

function renderSlopePanel(slope) {
  if (!_container) return
  const el = _container.querySelector('#slope-panel')
  if (!el) return

  if (!slope) {
    el.innerHTML = `<div class="info-panel">Slope data not available</div>`
    return
  }

  const slopeDeg = slope.terrain_slope_deg != null ? fmtNum(slope.terrain_slope_deg, 1) + '°' : '—'
  const nUp   = slope.n_upslope_pts || 0
  const nDn   = slope.n_downslope_pts || 0
  const total = nUp + nDn
  const upPct = total > 0 ? fmtNum(nUp / total * 100, 1) : '—'
  const dnPct = total > 0 ? fmtNum(nDn / total * 100, 1) : '—'

  // Detect gravity source: hardcoded fallback is approximately [0, -1, 0]
  let gravSource = '—'
  const g = slope.gravity_world
  if (g && Array.isArray(g)) {
    const isDefault = Math.abs(g[0]) < 0.05 && Math.abs(g[1] + 1) < 0.05 && Math.abs(g[2]) < 0.05
    gravSource = isDefault ? 'Hardcoded fallback [0, −1, 0]' : 'location.csv (real IMU)'
  }

  // Trunk lean from trunk_axis
  let trunkLean = '—'
  if (slope.trunk_axis) {
    const axis = slope.trunk_axis
    const horizAngle = Math.atan2(Math.sqrt(axis[0]*axis[0] + axis[2]*axis[2]), Math.abs(axis[1])) * 180 / Math.PI
    trunkLean = fmtNum(horizAngle, 1) + '°'
  }

  el.innerHTML = `
    <div class="meta-row"><span class="meta-key">Terrain slope</span><span class="meta-val">${slopeDeg}</span></div>
    <div class="meta-row"><span class="meta-key">Trunk lean</span><span class="meta-val">${trunkLean}</span></div>
    <div class="meta-row"><span class="meta-key">Upslope points</span><span class="meta-val">${nUp.toLocaleString()} (${upPct}%)</span></div>
    <div class="meta-row"><span class="meta-key">Downslope points</span><span class="meta-val">${nDn.toLocaleString()} (${dnPct}%)</span></div>
    <div class="meta-row"><span class="meta-key">Gravity source</span><span class="meta-val" style="font-size:11px">${gravSource}</span></div>
  `
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtNum(v, decimals = 1) {
  if (v == null || isNaN(Number(v))) return '—'
  return Number(v).toFixed(decimals)
}

function fmtCoord(v) {
  if (v == null) return '—'
  return Number(v).toFixed(6)
}

function formatDate(ms) {
  return new Date(ms).toLocaleDateString('en-US', { day: 'numeric', month: 'long', year: 'numeric' })
}

function axisToCompass(axis) {
  if (!Array.isArray(axis) || axis.length < 3) return '—'
  const angle = Math.atan2(axis[0], axis[2]) * 180 / Math.PI
  const normalized = ((angle % 360) + 360) % 360
  const dirs = ['N','NE','E','SE','S','SW','W','NW']
  return dirs[Math.round(normalized / 45) % 8]
}

function esc(str) {
  const d = document.createElement('div')
  d.textContent = String(str || '')
  return d.innerHTML
}
