import * as Upload from './views/upload.js'
import * as Jobs   from './views/jobs.js'
import * as Samples from './views/samples.js'
import * as Detail from './views/detail.js'

// ── API client ────────────────────────────────────────────────────────────────

async function _json(res) {
  if (!res.ok) {
    const err = new Error(`HTTP ${res.status}`)
    err.status = res.status
    throw err
  }
  return res.json()
}

window.API = {
  async upload(file, onProgress) {
    const form = new FormData()
    form.append('file', file)
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.upload.onprogress = e => { if (onProgress && e.lengthComputable) onProgress(e.loaded / e.total * 100) }
      xhr.onload = () => {
        try { resolve(JSON.parse(xhr.responseText)) }
        catch { reject(new Error('Invalid response')) }
      }
      xhr.onerror = () => reject(new Error('Upload failed — network error'))
      xhr.open('POST', '/api/upload')
      xhr.send(form)
    })
  },
  getJobs()      { return fetch('/api/jobs').then(_json) },
  getJob(id)     { return fetch(`/api/jobs/${id}`).then(_json) },
  getSamples()   { return fetch('/api/samples').then(_json) },
  getSample(id)  { return fetch(`/api/samples/${id}`).then(_json) },
  getStats(id)   { return fetch(`/api/samples/${id}/stats`).then(_json) },
  getSlope(id)   { return fetch(`/api/samples/${id}/slope`).then(_json) },
  getSolar(id)   { return fetch(`/api/samples/${id}/solar`).then(_json) },
  getLocation(id){ return fetch(`/api/samples/${id}/location`).then(_json) },
  getWind(id)    { return fetch(`/api/samples/${id}/wind`).then(_json) },
  meshUrl(id)    { return `/api/samples/${id}/files/trunk_mesh_detected.ply` },
  createSample(name, jobId) {
    return fetch('/api/samples', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, job_id: jobId }),
    }).then(_json)
  },
}

// ── Router ────────────────────────────────────────────────────────────────────

let _currentView = null

window.navigate = function navigate(hash, params = {}) {
  const query = Object.entries(params).map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join('&')
  const newHash = query ? `${hash}?${query}` : hash
  if (window.location.hash === newHash) {
    dispatch()
  } else {
    window.location.hash = newHash
  }
}

function parseLocation() {
  const raw = window.location.hash || '#upload'
  const [hash, qs] = raw.split('?')
  const params = {}
  if (qs) {
    qs.split('&').forEach(pair => {
      const [k, v] = pair.split('=')
      if (k) params[decodeURIComponent(k)] = decodeURIComponent(v || '')
    })
  }
  return { hash, params }
}

function setActiveNav(viewName) {
  document.querySelectorAll('.nav-link').forEach(a => {
    a.classList.toggle('active', a.dataset.view === viewName)
  })
}

function dispatch() {
  const { hash, params } = parseLocation()
  const viewName = hash.slice(1) // remove #

  // Cleanup previous view
  if (_currentView && typeof _currentView.destroy === 'function') {
    _currentView.destroy()
  }

  const main = document.getElementById('main')
  main.innerHTML = ''

  // Map view name to nav highlight and module
  const navMap = { upload: 'upload', jobs: 'jobs', samples: 'samples', detail: 'samples' }
  setActiveNav(navMap[viewName] || viewName)

  switch (viewName) {
    case 'upload':
      _currentView = Upload
      Upload.render(main)
      break
    case 'jobs':
      _currentView = Jobs
      Jobs.render(main, params)
      break
    case 'samples':
      _currentView = Samples
      Samples.render(main)
      break
    case 'detail':
      _currentView = Detail
      Detail.render(main, params.id)
      break
    default:
      _currentView = Upload
      Upload.render(main)
      break
  }
}

window.addEventListener('hashchange', dispatch)

// ── Boot ──────────────────────────────────────────────────────────────────────

dispatch()
