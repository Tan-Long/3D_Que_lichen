const STAGES = [
  { key: 'extract_rgb',   label: 'Extract RGB frames',  pctMin: 0,  pctMax: 5   },
  { key: 'quick_mesh',    label: 'Build quick mesh',     pctMin: 5,  pctMax: 50  },
  { key: 'tsdf',          label: 'TSDF fusion',          pctMin: 50, pctMax: 75  },
  { key: 'clean_pcd',     label: 'Clean point cloud',    pctMin: 75, pctMax: 78  },
  { key: 'recolor',       label: 'Recolor mesh',         pctMin: 78, pctMax: 82  },
  { key: 'trim',          label: 'Trim mesh',            pctMin: 82, pctMax: 85  },
  { key: 'finalize',      label: 'Finalize mesh',        pctMin: 85, pctMax: 88  },
  { key: 'detect_lichen', label: 'Detect lichen',        pctMin: 88, pctMax: 93  },
  { key: 'slope',         label: 'Slope analysis',       pctMin: 93, pctMax: 96  },
  { key: 'solar',         label: 'Solar exposure',       pctMin: 96, pctMax: 100 },
]

const ERROR_HINTS = [
  ['ffmpeg not found',    'Install ffmpeg: brew install ffmpeg (Mac) or sudo apt install ffmpeg (Linux)'],
  ['camera_matrix.csv',  'ZIP is missing camera_matrix.csv — check Stray Scanner export settings'],
  ['out of memory',       'Reduce TSDF_VOXEL_SIZE env var (try 0.010 instead of 0.008)'],
  ['No trunk points',     'Point cloud too sparse — rescan the tree from closer range'],
  ['camera_matrix',       'camera_matrix.csv missing or malformed in the uploaded ZIP'],
]

function currentStageIdx(pct) {
  for (let i = STAGES.length - 1; i >= 0; i--) {
    if (pct >= STAGES[i].pctMin) return i
  }
  return 0
}

function getHint(errorMsg) {
  if (!errorMsg) return ''
  const lower = errorMsg.toLowerCase()
  for (const [key, hint] of ERROR_HINTS) {
    if (lower.includes(key.toLowerCase())) return hint
  }
  return ''
}

function formatDuration(ms) {
  if (ms < 0) return ''
  const s = Math.floor(ms / 1000)
  if (s < 60) return s + 's'
  const m = Math.floor(s / 60)
  const rem = s % 60
  return m + 'm ' + rem + 's'
}

// Active WebSocket connections keyed by job_id
const _wsSockets = {}
// Elapsed time intervals
const _elapsedIntervals = {}
// Start times
const _startTimes = {}
// Cached jobs state
let _jobs = []
let _highlightJobId = null
// Poll interval for job list refresh
let _pollInterval = null
let _container = null

export function render(container, params = {}) {
  _container = container
  _highlightJobId = params.highlight || null

  container.innerHTML = `
    <div class="view-title">Pipeline Jobs</div>
    <div id="jobs-list"><div class="info-panel">Loading jobs...</div></div>
  `

  loadJobs()

  // Refresh job list every 5s (to catch new jobs created from other tabs/sessions)
  _pollInterval = setInterval(loadJobs, 5000)
}

export function destroy() {
  clearInterval(_pollInterval)
  _pollInterval = null

  // Close all WebSockets
  Object.values(_wsSockets).forEach(ws => {
    try { ws.close() } catch (_) {}
  })
  Object.keys(_wsSockets).forEach(k => delete _wsSockets[k])

  // Clear elapsed intervals
  Object.values(_elapsedIntervals).forEach(t => clearInterval(t))
  Object.keys(_elapsedIntervals).forEach(k => delete _elapsedIntervals[k])

  _container = null
}

async function loadJobs() {
  try {
    const jobs = await window.API.getJobs()
    _jobs = jobs || []
    renderJobList()

    // Connect WebSocket for any processing job that isn't already connected
    _jobs.forEach(job => {
      if (job.status === 'processing' && !_wsSockets[job.job_id]) {
        connectWS(job.job_id)
      }
    })
  } catch (err) {
    if (_container) {
      _container.querySelector('#jobs-list').innerHTML =
        `<div class="info-panel">Failed to load jobs: ${esc(err.message)}</div>`
    }
  }
}

function renderJobList() {
  if (!_container) return
  const el = _container.querySelector('#jobs-list')
  if (!el) return

  if (!_jobs.length) {
    el.innerHTML = `<div class="info-panel">No jobs yet. <a href="#upload" onclick="window.navigate('#upload');return false" style="color:var(--accent)">Upload a scan</a> to get started.</div>`
    return
  }

  // Sort newest first (job_id is a hex timestamp-ish uuid, but we just use array order from API)
  el.innerHTML = _jobs.map(job => renderJobCard(job)).join('')

  // Attach event delegation for save-as-sample
  el.querySelectorAll('[data-action="save-sample"]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation()
      const jobId = btn.dataset.jobId
      const input = el.querySelector(`#sample-name-${jobId}`)
      if (!input) return
      const name = (input.value || '').trim()
      if (!name) { input.focus(); return }
      saveSample(jobId, name, btn)
    })
  })

  el.querySelectorAll('[data-action="view-sample"]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation()
      window.navigate('#detail', { id: btn.dataset.sampleId })
    })
  })

  // Attach expand/collapse click on headers
  el.querySelectorAll('.job-header').forEach(hdr => {
    hdr.addEventListener('click', () => {
      const card = hdr.closest('[data-job-id]')
      if (!card) return
      const jobId = card.dataset.jobId
      const expanded = !card.classList.contains('expanded')
      card.classList.toggle('expanded', expanded)
      applyExpand(card, expanded)
      // Fetch full details for failed jobs to get the error message
      if (expanded) {
        const job = _jobs.find(j => j.job_id === jobId)
        if (job && job.status === 'error' && !job.error) {
          window.API.getJob(jobId).then(full => {
            const idx = _jobs.findIndex(j => j.job_id === jobId)
            if (idx >= 0) _jobs[idx] = { ..._jobs[idx], ...full }
            updateJobFromWS(jobId, full)
          }).catch(() => {})
        }
      }
    })
  })

  // Start elapsed counters for processing jobs
  _jobs.forEach(job => {
    if (job.status === 'processing') {
      startElapsedCounter(job.job_id)
    }
  })

  // Auto-expand highlighted job or any processing job
  const firstProcessing = _jobs.find(j => j.status === 'processing')
  const toExpand = _highlightJobId || (firstProcessing && firstProcessing.job_id)
  if (toExpand) {
    const card = el.querySelector(`[data-job-id="${toExpand}"]`)
    if (card) {
      card.classList.add('expanded')
      applyExpand(card, true)
    }
  }
}

function renderJobCard(job) {
  const pct = Math.round(job.progress * 100)
  const isProcessing = job.status === 'processing'
  const isDone = job.status === 'complete'
  const isError = job.status === 'error'
  const isHighlight = job.job_id === _highlightJobId

  const badge = badgeHtml(job.status)
  const stagesHtml = renderStages(pct, isProcessing, isDone)

  let actionHtml = ''
  if (isDone) {
    actionHtml = `
      <div style="margin-top:12px;padding:10px;background:var(--bg-page);border-radius:var(--radius-el);border:var(--border)">
        <div style="font-size:12px;color:var(--text-secondary);margin-bottom:6px">Save results as a named sample to view the full analysis:</div>
        <div style="display:flex;gap:8px">
          <input class="input" id="sample-name-${job.job_id}" placeholder="Sample name (e.g. cay_01)" style="flex:1">
          <button class="btn btn-primary btn-sm" data-action="save-sample" data-job-id="${job.job_id}">Save</button>
        </div>
        <div id="save-result-${job.job_id}" style="margin-top:6px;font-size:12px;display:none"></div>
      </div>
    `
  }

  if (isError) {
    const hint = getHint(job.error || '')
    actionHtml = `
      <div style="margin-top:12px">
        <div class="error-panel">
          <div class="error-panel-title">Pipeline failed${job.step ? ` at: ${esc(job.step)}` : ''}</div>
          <pre class="error-panel-log">${esc((job.error || 'Unknown error').slice(0, 1000))}</pre>
          ${hint ? `<div class="error-panel-hint">${esc(hint)}</div>` : ''}
        </div>
      </div>
    `
  }

  return `
    <div class="card" style="margin-bottom:12px;${isHighlight ? 'border-color:var(--accent)' : ''}" data-job-id="${job.job_id}">
      <div class="job-header" style="display:flex;align-items:center;gap:10px;cursor:pointer">
        <div style="flex:1;min-width:0">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
            <span style="font-weight:500;font-size:13px;font-family:monospace">${esc(job.job_id)}</span>
            ${badge}
            ${isProcessing ? `<span class="spinner"></span>` : ''}
          </div>
          <div style="font-size:12px;color:var(--text-secondary);margin-top:2px">${esc(job.step || '')}</div>
        </div>
        <div style="display:flex;align-items:center;gap:12px">
          ${isProcessing ? `<span id="elapsed-${job.job_id}" style="font-size:12px;color:var(--text-secondary);font-family:monospace">0s</span>` : ''}
          <span style="font-size:12px;color:var(--text-secondary)">${pct}%</span>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="transition:transform 0.2s;transform:var(--chevron-rot,rotate(0deg))" class="chevron"><polyline points="9 18 15 12 9 6"/></svg>
        </div>
      </div>

      <div style="margin-top:8px">
        <div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>
      </div>

      <div class="stages-wrap" style="display:none;margin-top:12px">
        <ul class="stage-list">${stagesHtml}</ul>
        ${actionHtml}
      </div>
    </div>
  `
}

function renderStages(pct, isProcessing, isDone) {
  const activeIdx = isDone ? STAGES.length : currentStageIdx(pct)
  return STAGES.map((stage, i) => {
    let icon, cls
    if (i < activeIdx || isDone) {
      icon = '✓'; cls = 'stage-done'
    } else if (isProcessing && i === activeIdx) {
      icon = '⟳'; cls = 'stage-running'
    } else {
      icon = '○'; cls = 'stage-waiting'
    }
    return `
      <li class="stage-item ${cls}">
        <span class="stage-icon">${icon}</span>
        <span class="stage-name">${i + 1}. ${stage.label}</span>
        <span class="stage-duration"></span>
      </li>
    `
  }).join('')
}

function badgeHtml(status) {
  const map = {
    queued:     ['badge-queued', 'Queued'],
    processing: ['badge-processing', 'Processing'],
    complete:   ['badge-done', 'Done'],
    error:      ['badge-error', 'Failed'],
  }
  const [cls, label] = map[status] || ['badge-queued', status]
  return `<span class="badge ${cls}">${label}</span>`
}

function startElapsedCounter(jobId) {
  if (_elapsedIntervals[jobId]) return
  if (!_startTimes[jobId]) _startTimes[jobId] = Date.now()

  _elapsedIntervals[jobId] = setInterval(() => {
    if (!_container) return
    const el = _container.querySelector(`#elapsed-${jobId}`)
    if (!el) {
      clearInterval(_elapsedIntervals[jobId])
      delete _elapsedIntervals[jobId]
      return
    }
    el.textContent = formatDuration(Date.now() - _startTimes[jobId])
  }, 1000)
}

function connectWS(jobId) {
  if (_wsSockets[jobId]) return
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const ws = new WebSocket(`${protocol}//${location.host}/ws/jobs/${jobId}`)
  _wsSockets[jobId] = ws

  ws.onmessage = event => {
    let data
    try { data = JSON.parse(event.data) } catch { return }
    updateJobFromWS(jobId, data)
  }

  ws.onclose = () => {
    delete _wsSockets[jobId]
    // If job is still processing, try to reconnect after 2s
    const job = _jobs.find(j => j.job_id === jobId)
    if (job && job.status === 'processing') {
      setTimeout(() => { if (_container) connectWS(jobId) }, 2000)
    }
  }

  ws.onerror = () => { ws.close() }
}

function updateJobFromWS(jobId, data) {
  // Update in-memory jobs array
  const idx = _jobs.findIndex(j => j.job_id === jobId)
  if (idx >= 0) {
    _jobs[idx] = { ..._jobs[idx], ...data }
  } else {
    _jobs.push(data)
  }

  // Stop elapsed counter if done
  if (data.status === 'complete' || data.status === 'error') {
    clearInterval(_elapsedIntervals[jobId])
    delete _elapsedIntervals[jobId]
  }

  // Re-render just this card
  if (!_container) return
  const card = _container.querySelector(`[data-job-id="${jobId}"]`)
  if (!card) {
    renderJobList()
    return
  }

  const wasExpanded = card.classList.contains('expanded')
  const pct = Math.round(data.progress * 100)

  // Update badge
  const badge = card.querySelector('.badge')
  if (badge) {
    badge.className = 'badge ' + (data.status === 'complete' ? 'badge-done' : data.status === 'error' ? 'badge-error' : data.status === 'processing' ? 'badge-processing' : 'badge-queued')
    badge.textContent = data.status === 'complete' ? 'Done' : data.status === 'error' ? 'Failed' : data.status === 'processing' ? 'Processing' : 'Queued'
  }

  // Update progress bar
  const fill = card.querySelector('.progress-fill')
  if (fill) fill.style.width = pct + '%'

  // Update step text
  const stepEl = card.querySelectorAll('div > div')[1]  // second child div
  // Update percentage text
  const pcts = card.querySelectorAll('span')
  pcts.forEach(s => { if (s.textContent.endsWith('%')) s.textContent = pct + '%' })

  // Update stages on significant changes
  const stagesWrap = card.querySelector('.stages-wrap')
  if (stagesWrap) {
    const isDone = data.status === 'complete'
    const isProcessing = data.status === 'processing'
    const stageList = stagesWrap.querySelector('.stage-list')
    if (stageList) {
      stageList.innerHTML = renderStages(pct, isProcessing, isDone)
    }

    // Show save form when done
    if (isDone && !stagesWrap.querySelector(`#sample-name-${jobId}`)) {
      const existingAction = stagesWrap.querySelector('.error-panel')
      if (!existingAction) {
        const div = document.createElement('div')
        div.style.marginTop = '12px'
        div.style.padding = '10px'
        div.style.background = 'var(--bg-page)'
        div.style.borderRadius = 'var(--radius-el)'
        div.style.border = 'var(--border)'
        div.innerHTML = `
          <div style="font-size:12px;color:var(--text-secondary);margin-bottom:6px">Save results as a named sample to view the full analysis:</div>
          <div style="display:flex;gap:8px">
            <input class="input" id="sample-name-${jobId}" placeholder="Sample name (e.g. cay_01)" style="flex:1">
            <button class="btn btn-primary btn-sm" data-action="save-sample" data-job-id="${jobId}">Save</button>
          </div>
          <div id="save-result-${jobId}" style="margin-top:6px;font-size:12px;display:none"></div>
        `
        stagesWrap.appendChild(div)
        div.querySelector('[data-action="save-sample"]').addEventListener('click', e => {
          e.stopPropagation()
          const input = div.querySelector(`#sample-name-${jobId}`)
          const name = (input.value || '').trim()
          if (!name) { input.focus(); return }
          saveSample(jobId, name, div.querySelector('[data-action="save-sample"]'))
        })
      }
    }
  }

  if (wasExpanded) {
    card.classList.add('expanded')
    applyExpand(card, true)
  }

  // Manage spinner visibility
  const spinner = card.querySelector('.spinner')
  if (spinner) spinner.style.display = (data.status === 'processing') ? 'inline-block' : 'none'

  // Re-attach header click after any update that may have mutated the DOM
  const hdr = card.querySelector('.job-header')
  if (hdr && !hdr._clickAttached) {
    hdr._clickAttached = true
    hdr.addEventListener('click', () => {
      const expanded = !card.classList.contains('expanded')
      card.classList.toggle('expanded', expanded)
      applyExpand(card, expanded)
    })
  }
}

function applyExpand(card, expanded) {
  const wrap = card.querySelector('.stages-wrap')
  const chevron = card.querySelector('.chevron')
  if (wrap) wrap.style.display = expanded ? 'block' : 'none'
  if (chevron) chevron.style.transform = expanded ? 'rotate(90deg)' : 'rotate(0deg)'
}

async function saveSample(jobId, name, btn) {
  if (!btn) return
  const origText = btn.textContent
  btn.disabled = true
  btn.textContent = 'Saving...'

  const resultEl = document.getElementById(`save-result-${jobId}`)

  try {
    const sample = await window.API.createSample(name, jobId)
    if (resultEl) {
      resultEl.style.display = 'block'
      resultEl.style.color = 'var(--badge-done-fg)'
      resultEl.innerHTML = `Saved as "${esc(sample.name)}" — <a href="#detail?id=${sample.id}" onclick="window.navigate('#detail',{id:'${sample.id}'});return false" style="color:var(--accent)">View analysis</a>`
    }
    btn.style.display = 'none'
    const input = document.getElementById(`sample-name-${jobId}`)
    if (input) input.disabled = true
  } catch (err) {
    if (resultEl) {
      resultEl.style.display = 'block'
      resultEl.style.color = 'var(--badge-error-fg)'
      resultEl.textContent = 'Save failed: ' + err.message
    }
    btn.disabled = false
    btn.textContent = origText
  }
}


function esc(str) {
  const d = document.createElement('div')
  d.textContent = String(str || '')
  return d.innerHTML
}
