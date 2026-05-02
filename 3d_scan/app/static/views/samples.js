let _container = null
let _allSamples = []
let _searchQuery = ''
let _sortKey = 'created'
let _sortDir = -1  // -1 = descending

export function render(container) {
  _container = container
  _allSamples = []

  container.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:8px">
      <div class="view-title" style="margin-bottom:0">Samples</div>
      <a class="btn btn-primary btn-sm" href="#upload" onclick="window.navigate('#upload');return false">+ Upload new</a>
    </div>

    <div class="filter-bar">
      <input class="input" id="search-input" placeholder="Search by name..." style="max-width:200px" value="${esc(_searchQuery)}">
      <select class="input" id="sort-select" style="width:auto">
        <option value="created" ${_sortKey==='created'?'selected':''}>Newest first</option>
        <option value="name" ${_sortKey==='name'?'selected':''}>Name A–Z</option>
        <option value="status" ${_sortKey==='status'?'selected':''}>Status</option>
      </select>
    </div>

    <div id="samples-grid">
      <div class="info-panel">Loading samples...</div>
    </div>
  `

  container.querySelector('#search-input').addEventListener('input', e => {
    _searchQuery = e.target.value
    renderGrid()
  })

  container.querySelector('#sort-select').addEventListener('change', e => {
    _sortKey = e.target.value
    _sortDir = _sortKey === 'name' ? 1 : -1
    renderGrid()
  })

  loadSamples()
}

export function destroy() {
  _container = null
}

async function loadSamples() {
  try {
    const samples = await window.API.getSamples()
    _allSamples = samples || []
    renderGrid()
  } catch (err) {
    if (_container) {
      _container.querySelector('#samples-grid').innerHTML =
        `<div class="info-panel">Failed to load samples: ${esc(err.message)}</div>`
    }
  }
}

function renderGrid() {
  if (!_container) return
  const grid = _container.querySelector('#samples-grid')
  if (!grid) return

  let filtered = _allSamples
  if (_searchQuery) {
    const q = _searchQuery.toLowerCase()
    filtered = filtered.filter(s => (s.name || '').toLowerCase().includes(q))
  }

  filtered = [...filtered].sort((a, b) => {
    if (_sortKey === 'name') return naturalCompare(a.name, b.name) * _sortDir
    if (_sortKey === 'status') return String(a.status).localeCompare(String(b.status)) * _sortDir
    // 'created' default
    return ((a.created_at || 0) - (b.created_at || 0)) * _sortDir
  })

  if (!filtered.length) {
    grid.innerHTML = _allSamples.length
      ? `<div class="info-panel">No samples match your search.</div>`
      : `<div class="info-panel">No samples yet. <a href="#upload" onclick="window.navigate('#upload');return false" style="color:var(--accent)">Upload a scan</a> to get started.</div>`
    return
  }

  grid.innerHTML = `<div class="sample-grid">${filtered.map(renderCard).join('')}</div>`

  grid.querySelectorAll('[data-action="view"]').forEach(btn => {
    btn.addEventListener('click', () => {
      window.navigate('#detail', { id: btn.dataset.id })
    })
  })
}

function renderCard(sample) {
  const date = sample.created_at ? formatDate(sample.created_at * 1000) : '—'
  const meshFile = (sample.files || []).find(f => f.file_type === 'mesh' || (f.filename || '').endsWith('.ply'))
  const hasMesh = !!meshFile

  const statusColors = {
    unprocessed: ['#f0f0f0','#555'],
    rgb_extracted: ['#FFF4E0','#b45309'],
    meshed: ['#E6F1FB','#185FA5'],
    filtered: ['#F0E8FF','#6d28d9'],
    labeled: ['#E6F4EA','#1a7f37'],
    complete: ['#E6F4EA','#1a7f37'],
  }
  const [sbg, sfg] = statusColors[sample.status] || ['#f0f0f0','#555']

  return `
    <div class="card" style="cursor:pointer" onclick="window.navigate('#detail',{id:'${sample.id}'})">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:10px">
        <div>
          <div style="font-weight:600;font-size:14px;margin-bottom:2px">${esc(sample.name)}</div>
          <div style="font-size:11px;color:var(--text-secondary)">${date}</div>
        </div>
        <span style="display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:500;background:${sbg};color:${sfg};white-space:nowrap">${esc(sample.status || 'unprocessed')}</span>
      </div>

      <hr class="sep" style="margin:8px 0">

      <div style="font-size:12px;color:var(--text-secondary);margin-bottom:8px">
        ${(sample.files || []).length} file${(sample.files||[]).length !== 1 ? 's' : ''}
        ${hasMesh ? ' · mesh ready' : ''}
      </div>

      <button class="btn btn-sm" data-action="view" data-id="${sample.id}" style="width:100%">
        View Analysis →
      </button>
    </div>
  `
}

function formatDate(ms) {
  const d = new Date(ms)
  return d.toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' })
}

function naturalCompare(a, b) {
  const sa = String(a || ''), sb = String(b || '')
  const ma = sa.match(/^(.*?)(\d+)\s*$/)
  const mb = sb.match(/^(.*?)(\d+)\s*$/)
  if (ma && mb) {
    const pc = (ma[1] || '').localeCompare(mb[1] || '', undefined, { sensitivity: 'base' })
    if (pc !== 0) return pc
    return parseInt(ma[2]) - parseInt(mb[2])
  }
  return sa.localeCompare(sb, undefined, { sensitivity: 'base' })
}

function esc(str) {
  const d = document.createElement('div')
  d.textContent = String(str || '')
  return d.innerHTML
}
