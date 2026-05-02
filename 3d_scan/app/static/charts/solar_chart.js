export function renderSolarChart(canvas, solar) {
  const Chart = window.Chart
  if (!Chart || !canvas) return null

  const monthly = solar.monthly || []
  const mean    = solar.sun_facing_fraction_mean ?? solar.mean_sun_facing ?? 0
  const peak    = solar.peak_solar_month || ''

  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  const gridColor  = isDark ? '#2a2a2a' : '#e5e5e5'
  const labelColor = isDark ? '#888' : '#666'
  const tickColor  = isDark ? '#f0f0f0' : '#111'

  const labels = monthly.map(m => m.month || m.label || m.name || '')
  const values = monthly.map(m => {
    const v = m.sun_facing_fraction ?? m.fraction ?? m.value ?? 0
    return Math.round(v * 1000) / 1000  // 3 decimal places
  })
  const bgColors = labels.map(l => l === peak ? '#EF9F27' : '#B5D4F4')

  const datasets = [{
    label: 'Sun-facing fraction',
    data: values,
    backgroundColor: bgColors,
    borderRadius: 3,
  }]

  // Mean line as a dataset
  if (mean > 0) {
    datasets.push({
      label: 'Mean',
      data: new Array(labels.length).fill(Math.round(mean * 1000) / 1000),
      type: 'line',
      borderColor: '#185FA5',
      borderDash: [4, 4],
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
    })
  }

  return new Chart(canvas, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: labelColor, font: { size: 11 }, boxWidth: 12 },
        },
        tooltip: {
          callbacks: {
            label: ctx => `${(ctx.parsed.y * 100).toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: tickColor, font: { size: 11 } },
          grid: { color: gridColor },
        },
        y: {
          min: 0,
          max: 1,
          title: { display: true, text: 'Sun-facing fraction', color: labelColor, font: { size: 11 } },
          ticks: {
            color: tickColor,
            font: { size: 11 },
            callback: v => (v * 100).toFixed(0) + '%',
          },
          grid: { color: gridColor },
        },
      },
    },
  })
}
