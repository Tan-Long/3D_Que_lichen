const REQUIRED_FILES = [
  { name: 'camera_matrix.csv', required: true, note: 'Camera intrinsics' },
  { name: 'odometry.csv',       required: true, note: 'Camera trajectory' },
  { name: 'imu.csv',            required: true, note: 'Accelerometer data' },
  { name: 'rgb.mp4',            required: true, note: 'RGB video' },
  { name: 'depth/',             required: true, note: '16-bit PNG depth maps' },
  { name: 'confidence/',        required: true, note: 'Confidence maps' },
  { name: 'location.csv',       required: false, note: 'GPS + IMU (patched app)' },
  { name: 'location.json',      required: false, note: 'GPS + IMU JSON (patched app)' },
  { name: 'tree_gps_anchor.json',required: false, note: 'GPS anchor (patched app)' },
  { name: 'frame_transforms.csv',required: false, note: 'Per-frame 4×4 transforms (patched app)' },
  { name: 'point_cloud_raw.ply', required: false, note: 'ARKit sparse points (optional)' },
]

let _dragListeners = []

export function render(container) {
  container.innerHTML = `
    <div class="view-title">Upload Scan</div>

    <div class="card" style="max-width:560px">
      <div id="drop-zone" class="drop-zone">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="color:var(--text-secondary);display:block;margin:0 auto 12px">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
        <div style="font-size:14px;margin-bottom:4px">Drag &amp; drop your <strong>.zip</strong> dataset here</div>
        <div style="font-size:12px;color:var(--text-secondary)">or click to browse</div>
        <input type="file" id="file-input" accept=".zip" style="display:none">
      </div>

      <div id="file-chosen" style="display:none;margin-top:10px;padding:8px 12px;background:var(--accent-light);border-radius:var(--radius-el);font-size:13px;display:flex;align-items:center;justify-content:space-between">
        <span id="file-name" style="font-weight:500"></span>
        <span id="file-size" style="color:var(--text-secondary)"></span>
      </div>

      <div id="upload-error" style="display:none;margin-top:10px"></div>

      <div id="upload-progress" style="display:none;margin-top:10px">
        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-secondary);margin-bottom:4px">
          <span id="upload-status">Uploading...</span>
          <span id="upload-pct">0%</span>
        </div>
        <div class="progress-track"><div class="progress-fill" id="upload-bar" style="width:0"></div></div>
      </div>

      <div style="margin-top:14px;display:flex;align-items:center;gap:8px">
        <button id="btn-submit" class="btn btn-primary" disabled>Upload &amp; Process</button>
      </div>
    </div>

    <div class="card" style="max-width:560px;margin-top:12px">
      <div class="panel-title">Required files in ZIP</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <colgroup><col style="width:160px"><col><col style="width:80px"></colgroup>
        <thead>
          <tr>
            <th style="text-align:left;padding:4px 0;color:var(--text-secondary);font-weight:500">File</th>
            <th style="text-align:left;padding:4px 0;color:var(--text-secondary);font-weight:500">Description</th>
            <th style="text-align:left;padding:4px 0;color:var(--text-secondary);font-weight:500">Required</th>
          </tr>
        </thead>
        <tbody>
          ${REQUIRED_FILES.map(f => `
          <tr style="border-top:var(--border)">
            <td style="padding:5px 0;font-family:monospace;font-size:11px">${f.name}</td>
            <td style="padding:5px 0;color:var(--text-secondary)">${f.note}</td>
            <td style="padding:5px 0">${f.required ? '<span style="color:var(--badge-error-fg)">required</span>' : '<span style="color:var(--text-secondary)">optional</span>'}</td>
          </tr>`).join('')}
        </tbody>
      </table>
    </div>
  `

  const dropZone = container.querySelector('#drop-zone')
  const fileInput = container.querySelector('#file-input')
  const btnSubmit = container.querySelector('#btn-submit')
  const fileChosen = container.querySelector('#file-chosen')
  const fileName = container.querySelector('#file-name')
  const fileSize = container.querySelector('#file-size')
  const uploadError = container.querySelector('#upload-error')
  const uploadProgress = container.querySelector('#upload-progress')
  const uploadBar = container.querySelector('#upload-bar')
  const uploadStatus = container.querySelector('#upload-status')
  const uploadPct = container.querySelector('#upload-pct')

  let selectedFile = null

  function showError(msg) {
    uploadError.style.display = 'block'
    uploadError.innerHTML = `<div class="error-panel"><div class="error-panel-title">Error</div><div class="error-panel-log">${esc(msg)}</div></div>`
  }

  function clearError() {
    uploadError.style.display = 'none'
    uploadError.innerHTML = ''
  }

  function selectFile(file) {
    if (!file.name.toLowerCase().endsWith('.zip')) {
      showError('Please select a .zip file.')
      return
    }
    selectedFile = file
    fileChosen.style.display = 'flex'
    fileName.textContent = file.name
    fileSize.textContent = formatSize(file.size)
    btnSubmit.disabled = false
    clearError()
  }

  // Drop zone events
  const onDragOver = e => { e.preventDefault(); dropZone.classList.add('dragover') }
  const onDragLeave = () => dropZone.classList.remove('dragover')
  const onDrop = e => {
    e.preventDefault()
    dropZone.classList.remove('dragover')
    const f = e.dataTransfer.files[0]
    if (f) selectFile(f)
  }
  const onClick = () => fileInput.click()

  dropZone.addEventListener('dragover', onDragOver)
  dropZone.addEventListener('dragleave', onDragLeave)
  dropZone.addEventListener('drop', onDrop)
  dropZone.addEventListener('click', onClick)
  _dragListeners = [
    [dropZone, 'dragover', onDragOver],
    [dropZone, 'dragleave', onDragLeave],
    [dropZone, 'drop', onDrop],
    [dropZone, 'click', onClick],
  ]

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) selectFile(fileInput.files[0])
  })

  btnSubmit.addEventListener('click', async () => {
    if (!selectedFile) return
    btnSubmit.disabled = true
    clearError()
    uploadProgress.style.display = 'block'
    uploadStatus.textContent = 'Uploading...'

    try {
      const result = await window.API.upload(selectedFile, pct => {
        uploadBar.style.width = pct.toFixed(1) + '%'
        uploadPct.textContent = Math.round(pct) + '%'
      })

      uploadBar.style.width = '100%'
      uploadPct.textContent = '100%'
      uploadStatus.textContent = 'Upload complete — starting pipeline...'

      if (result && result.job_id) {
        setTimeout(() => window.navigate('#jobs', { highlight: result.job_id }), 600)
      } else {
        showError('Unexpected response from server.')
        uploadProgress.style.display = 'none'
        btnSubmit.disabled = false
      }
    } catch (err) {
      uploadProgress.style.display = 'none'
      btnSubmit.disabled = false
      showError(err.message || 'Upload failed.')
    }
  })
}

export function destroy() {
  _dragListeners.forEach(([el, ev, fn]) => el && el.removeEventListener(ev, fn))
  _dragListeners = []
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / 1048576).toFixed(1) + ' MB'
}

function esc(str) {
  const d = document.createElement('div')
  d.textContent = str
  return d.innerHTML
}
