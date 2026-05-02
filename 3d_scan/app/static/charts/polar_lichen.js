export function renderLichenChart(canvas, stats) {
  const Chart = window.Chart
  if (!Chart || !canvas) return null

  const upslope   = stats.upslope   || {}
  const downslope = stats.downslope || {}

  const whitUp  = upslope.white_lichen_cm2   ?? upslope.lichen_cm2   ?? 0
  const greenUp = upslope.green_lichen_cm2  ?? 0
  const whitDn  = downslope.white_lichen_cm2 ?? downslope.lichen_cm2 ?? 0
  const greenDn = downslope.green_lichen_cm2 ?? 0

  // Fallback: if no color split, use total
  const totalUp = upslope.total_lichen_cm2 ?? (whitUp + greenUp)
  const totalDn = downslope.total_lichen_cm2 ?? (whitDn + greenDn)

  const hasColorSplit = (whitUp + greenUp + whitDn + greenDn) > 0

  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  const gridColor  = isDark ? '#2a2a2a' : '#e5e5e5'
  const labelColor = isDark ? '#888' : '#666'
  const tickColor  = isDark ? '#f0f0f0' : '#111'

  const datasets = hasColorSplit
    ? [
        {
          label: 'White lichen (cm²)',
          data: [Math.round(whitUp), Math.round(whitDn)],
          backgroundColor: '#B5D4F4',
        },
        {
          label: 'Green lichen (cm²)',
          data: [Math.round(greenUp), Math.round(greenDn)],
          backgroundColor: '#97C459',
        },
      ]
    : [
        {
          label: 'Lichen (cm²)',
          data: [Math.round(totalUp), Math.round(totalDn)],
          backgroundColor: '#B5D4F4',
        },
      ]

  return new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['Upslope', 'Downslope'],
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: labelColor, font: { size: 11 }, boxWidth: 12 },
        },
      },
      scales: {
        x: {
          ticks: { color: tickColor, font: { size: 12 } },
          grid: { color: gridColor },
        },
        y: {
          title: { display: true, text: 'Area (cm²)', color: labelColor, font: { size: 11 } },
          ticks: { color: tickColor, font: { size: 11 } },
          grid: { color: gridColor },
          beginAtZero: true,
        },
      },
    },
  })
}
