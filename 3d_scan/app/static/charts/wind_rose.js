const DIRS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

export function renderWindRose(canvas, wind) {
  const Chart = window.Chart
  if (!Chart || !canvas) return null

  const rose = wind.wind_rose || {}
  const dominant = wind.dominant_wind_dir_cardinal || ''
  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  const labelColor = isDark ? '#888' : '#666'
  const tickColor  = isDark ? '#f0f0f0' : '#111'

  const values = DIRS.map(d => rose[d] || 0)
  const bgColors = DIRS.map(d => d === dominant ? '#185FA5' : '#B5D4F4')
  const borderColors = DIRS.map(d => d === dominant ? '#0d3d6b' : '#93b8d4')

  return new Chart(canvas, {
    type: 'polarArea',
    data: {
      labels: DIRS,
      datasets: [{
        data: values,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.label}: ${ctx.parsed.r.toLocaleString()} hrs`,
          },
        },
      },
      scales: {
        r: {
          ticks: { display: false },
          grid: { color: isDark ? '#2a2a2a' : '#e5e5e5' },
          pointLabels: { color: labelColor, font: { size: 11 } },
        },
      },
    },
  })
}
