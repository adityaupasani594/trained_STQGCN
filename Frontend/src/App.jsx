import { Canvas, useFrame } from '@react-three/fiber'
import { Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { motion } from 'framer-motion'
import { useEffect, useMemo, useRef, useState } from 'react'

const DEFAULTS = {
  rain: 36,
  wind: 24,
  temperature: 28,
  quantumDepth: 5,
  speedFactor: 1.2,
  timeOfDay: 14,
}

const LIVE_REFRESH_MS = 5_000

const ROAD_PATHS = [
  // Primary North-South Arterial (Curved)
  [[-15.2, 0.2, 8.5], [-14.8, 0.2, 4.2], [-14.5, 0.2, 0.1], [-15.0, 0.2, -4.5], [-15.5, 0.2, -8.2]],
  [[0, 0.2, 9.0], [1.2, 0.2, 4.5], [0, 0.2, 0], [-1.5, 0.2, -4.8], [0, 0.2, -9.2]],
  [[14.5, 0.2, 8.2], [15.0, 0.2, 3.8], [14.8, 0.2, -0.5], [14.2, 0.2, -4.2], [15.1, 0.2, -8.8]],

  // Primary East-West Arterial (Curved)
  [[-18.5, 0.2, 5.5], [-14.8, 0.2, 4.2], [-8.2, 0.2, 4.8], [1.2, 0.2, 4.5], [7.5, 0.2, 4.2], [15.0, 0.2, 3.8], [18.8, 0.2, 4.5]],
  [[-19.0, 0.2, -2.5], [-14.5, 0.2, 0.1], [-7.5, 0.2, -0.5], [0, 0.2, 0], [8.0, 0.2, 0.5], [14.8, 0.2, -0.5], [19.2, 0.2, -1.2]],
  [[-18.2, 0.2, -7.8], [-15.5, 0.2, -8.2], [-7.8, 0.2, -8.5], [0, 0.2, -9.2], [8.5, 0.2, -8.8], [15.1, 0.2, -8.8], [18.5, 0.2, -8.2]],

  // Diagonal & Radial Connectors
  [[-14.8, 0.2, 4.2], [-7.5, 0.2, -0.5]],
  [[0, 0.2, 0], [7.5, 0.2, 4.2]],
  [[15.0, 0.2, 3.8], [8.5, 0.2, -8.8]],
  [[-1.5, 0.2, -4.8], [-7.8, 0.2, -8.5]],

  // Intermediates for Complexity
  [[-8.2, 0.2, 4.8], [-7.5, 0.2, -0.5]],
  [[-7.5, 0.2, -0.5], [-7.8, 0.2, -8.5]],
  [[7.5, 0.2, 4.2], [8.0, 0.2, 0.5]],
  [[8.0, 0.2, 0.5], [8.5, 0.2, -8.8]],
]



function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function heatColor(t) {
  const v = clamp(t, 0, 1)
  if (v < 0.2) return '#22c55e'   // green: free flow
  if (v < 0.4) return '#fde047'   // yellow: light traffic
  if (v < 0.6) return '#fb923c'   // orange: moderate congestion
  if (v < 0.8) return '#ef4444'   // red: heavy congestion
  return '#050505'                // black: near standstill
}

function lerp(a, b, t) {
  return a + (b - a) * t
}

function lerpHexColor(hexA, hexB, t) {
  const a = hexA.replace('#', '')
  const b = hexB.replace('#', '')
  const ar = parseInt(a.slice(0, 2), 16)
  const ag = parseInt(a.slice(2, 4), 16)
  const ab = parseInt(a.slice(4, 6), 16)
  const br = parseInt(b.slice(0, 2), 16)
  const bg = parseInt(b.slice(2, 4), 16)
  const bb = parseInt(b.slice(4, 6), 16)
  const rr = Math.round(lerp(ar, br, t)).toString(16).padStart(2, '0')
  const rg = Math.round(lerp(ag, bg, t)).toString(16).padStart(2, '0')
  const rb = Math.round(lerp(ab, bb, t)).toString(16).padStart(2, '0')
  return `#${rr}${rg}${rb}`
}

function toDaylightFactor(hour) {
  const wrapped = ((hour % 24) + 24) % 24
  const solar = Math.cos(((wrapped - 12) / 12) * Math.PI)
  return clamp((solar + 0.12) / 1.12, 0, 1)
}

function edgePointDistance2D(px, pz, ax, az, bx, bz) {
  const abx = bx - ax
  const abz = bz - az
  const apx = px - ax
  const apz = pz - az
  const denom = abx * abx + abz * abz
  const t = denom === 0 ? 0 : clamp((apx * abx + apz * abz) / denom, 0, 1)
  const cx = ax + abx * t
  const cz = az + abz * t
  const dx = px - cx
  const dz = pz - cz
  return Math.sqrt(dx * dx + dz * dz)
}

function pointKey(point) {
  return `${point[0].toFixed(2)}|${point[2].toFixed(2)}`
}

function pointDist2D(a, b) {
  const dx = a[0] - b[0]
  const dz = a[2] - b[2]
  return Math.sqrt(dx * dx + dz * dz)
}

function buildComplexNetwork() {
  const degreeByKey = new Map()
  const pointByKey = new Map()
  const roadSegments = []

  ROAD_PATHS.forEach((path) => {
    for (let i = 1; i < path.length; i += 1) {
      const a = path[i - 1]
      const b = path[i]
      const ka = pointKey(a)
      const kb = pointKey(b)
      pointByKey.set(ka, a)
      pointByKey.set(kb, b)
      degreeByKey.set(ka, (degreeByKey.get(ka) ?? 0) + 1)
      degreeByKey.set(kb, (degreeByKey.get(kb) ?? 0) + 1)
      roadSegments.push({ a, b, length: pointDist2D(a, b) })
    }
  })

  const junctionKeys = Array.from(degreeByKey.entries())
    .filter(([, degree]) => degree >= 2)
    .map(([k]) => k)

  const junctionNodes = junctionKeys.map((k) => pointByKey.get(k))
  const junctionIndexByKey = new Map(junctionKeys.map((k, idx) => [k, idx]))

  const edgeMap = new Map()
  ROAD_PATHS.forEach((path) => {
    let prevJ = null
    let cumulative = 0
    for (let i = 1; i < path.length; i += 1) {
      cumulative += pointDist2D(path[i - 1], path[i])
      const key = pointKey(path[i])
      if (!junctionIndexByKey.has(key)) continue

      const curJ = junctionIndexByKey.get(key)
      if (prevJ !== null && prevJ !== curJ) {
        const minIdx = Math.min(prevJ, curJ)
        const maxIdx = Math.max(prevJ, curJ)
        const eKey = `${minIdx}|${maxIdx}`
        const existing = edgeMap.get(eKey)
        if (!existing || cumulative < existing.d) {
          edgeMap.set(eKey, { a: minIdx, b: maxIdx, d: cumulative })
        }
      }
      prevJ = curJ
      cumulative = 0
    }
  })

  const junctionEdges = Array.from(edgeMap.values())

  const nearestJunction = (point) => {
    let bestIdx = 0
    let bestDist = Number.POSITIVE_INFINITY
    for (let i = 0; i < junctionNodes.length; i += 1) {
      const d = pointDist2D(point, junctionNodes[i])
      if (d < bestDist) {
        bestDist = d
        bestIdx = i
      }
    }
    return bestIdx
  }

  const roadSegmentsWithJunction = roadSegments.map((seg) => ({
    ...seg,
    jA: nearestJunction(seg.a),
    jB: nearestJunction(seg.b),
  }))

  return {
    junctionNodes,
    junctionEdges,
    roadSegments: roadSegmentsWithJunction,
  }
}

function Buildings({ networkLoad, nodes, roadSegments }) {
  const blocks = useMemo(() => {
    const maxBuildings = 90
    const out = []
    let attempts = 0

    while (out.length < maxBuildings && attempts < 3500) {
      attempts += 1
      const x = -19 + Math.random() * 38
      const z = -9 + Math.random() * 17

      let blocked = false

      for (let i = 0; i < nodes.length; i += 1) {
        const dx = x - nodes[i][0]
        const dz = z - nodes[i][2]
        if (Math.sqrt(dx * dx + dz * dz) < 1.2) {
          blocked = true
          break
        }
      }
      if (blocked) continue

      for (let i = 0; i < roadSegments.length; i += 1) {
        const seg = roadSegments[i]
        const dist = edgePointDistance2D(x, z, seg.a[0], seg.a[2], seg.b[0], seg.b[2])
        if (dist < 0.65) {
          blocked = true
          break
        }
      }
      if (blocked) continue

      const h = 0.9 + Math.random() * 4.8
      out.push({
        pos: [x, h / 2, z],
        h,
        sx: 0.7 + Math.random() * 0.95,
        sz: 0.7 + Math.random() * 0.95,
      })
    }

    return out
  }, [nodes, roadSegments])

  const glow = clamp(networkLoad, 0, 1)

  return (
    <group>
      {blocks.map((blk, idx) => (
        <mesh key={idx} position={blk.pos} castShadow receiveShadow>
          <boxGeometry args={[blk.sx, blk.h, blk.sz]} />
          <meshStandardMaterial
            color="#666666"
            emissive="#f97316"
            emissiveIntensity={0.03 + glow * 0.12}
            roughness={0.58}
            metalness={0.22}
          />
        </mesh>
      ))}
    </group>
  )
}

function RainField({ rain, wind }) {
  const rainRef = useRef()

  const drops = useMemo(() => {
    if (rain <= 0) return []
    const count = Math.floor(rain * 7)
    const arr = []
    for (let i = 0; i < count; i += 1) {
      arr.push({
        x: (Math.random() - 0.5) * 40,
        y: Math.random() * 18 + 2,
        z: (Math.random() - 0.5) * 22,
      })
    }
    return arr
  }, [rain])

  useFrame((_, delta) => {
    if (!rainRef.current || rain <= 0) return
    const speed = 3 + rain * 0.045
    rainRef.current.children.forEach((drop) => {
      drop.position.y -= delta * speed
      drop.position.x += delta * wind * 0.012
      if (drop.position.y < 0) drop.position.y = 18
      if (drop.position.x > 20) drop.position.x = -20
      if (drop.position.x < -20) drop.position.x = 20
    })
  })

  if (rain <= 0) return null

  return (
    <group ref={rainRef}>
      {drops.map((d, idx) => (
        <mesh key={idx} position={[d.x, d.y, d.z]}>
          <cylinderGeometry args={[0.015, 0.015, 0.65, 5]} />
          <meshBasicMaterial color="#9fd8ff" transparent opacity={0.22 + rain * 0.0042} />
        </mesh>
      ))}
    </group>
  )
}

function NetworkScene({ params, selectedNode, onNodeSelect, network, nodeState, networkLoad }) {
  const nodes = network.junctionNodes
  const roadSegments = network.roadSegments

  const pulseRef = useRef([])
  const day = toDaylightFactor(params.timeOfDay)

  const bgColor = lerpHexColor('#041022', '#88d2ff', day)
  const fogColor = lerpHexColor('#07121f', '#9ac3de', day)
  const ambient = 0.2 + day * 0.55
  const sunIntensity = 0.12 + day * 1.05

  useFrame((state) => {
    const t = state.clock.getElapsedTime() * params.speedFactor
    pulseRef.current.forEach((mesh, idx) => {
      if (!mesh) return
      const wobble = 1 + Math.sin(t * 2 + idx * 0.22) * 0.1
      mesh.scale.set(wobble, wobble, wobble)
    })
  })

  return (
    <>
      <color attach="background" args={[bgColor]} />
      <fog attach="fog" args={[fogColor, 12, 44]} />

      <ambientLight intensity={ambient} color={day > 0.5 ? '#f8f4d6' : '#9ec5ff'} />
      <directionalLight position={[8 + day * 10, 8 + day * 16, 6]} intensity={sunIntensity} color="#fff3c2" castShadow />
      <pointLight position={[-12, 8, -7]} intensity={0.35 + (1 - day) * 0.35} color="#4ac3ff" />

      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[54, 30]} />
        <meshStandardMaterial color={lerpHexColor('#0c1b2f', '#6d7f8c', day * 0.3)} roughness={1} metalness={0.05} />
      </mesh>

      <Buildings networkLoad={networkLoad} nodes={nodes} roadSegments={roadSegments} />

      {roadSegments.map((seg, idx) => {
        const normalized = 1 - clamp(seg.length / 6, 0, 1)
        const edgeCongestion = (nodeState[seg.jA]?.congestion + nodeState[seg.jB]?.congestion) / 2
        const heat = clamp(networkLoad * 0.22 + normalized * 0.16 + edgeCongestion * 0.62, 0, 1)
        return (
          <Line
            key={idx}
            points={[seg.a, seg.b]}
            color={heatColor(heat)}
            lineWidth={1.05 + heat * 1.85}
            transparent
            opacity={0.38 + heat * 0.48}
          />
        )
      })}

      {nodes.map((n, idx) => {
        const congestion = nodeState[idx]?.congestion ?? 0
        const heat = clamp(congestion, 0, 1)
        const isSelected = selectedNode === idx

        return (
          <mesh
            key={idx}
            ref={(el) => {
              pulseRef.current[idx] = el
            }}
            position={n}
            castShadow
            onPointerDown={(e) => {
              e.stopPropagation()
              onNodeSelect(idx)
            }}
          >
            <sphereGeometry args={[0.3 + heat * 0.25 + (isSelected ? 0.1 : 0), 20, 20]} />
            <meshStandardMaterial
              color={isSelected ? '#d6f6ff' : heatColor(heat)}
              emissive={heatColor(heat)}
              emissiveIntensity={0.3 + heat * 0.7 + (isSelected ? 0.3 : 0)}
              roughness={0.35}
              metalness={0.2}
            />
          </mesh>
        )
      })}

      <RainField rain={params.rain} wind={params.wind} />

      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        maxPolarAngle={Math.PI / 2.07}
        minDistance={14}
        maxDistance={38}
      />
    </>
  )
}

function Slider({ label, value, min, max, step, onChange, unit }) {
  return (
    <label className="space-y-2">
      <div className="flex items-center justify-between text-sm font-semibold tracking-wide text-slate-100">
        <span>{label}</span>
        <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-cyan-200">
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-600/80 accent-cyan-400"
      />
    </label>
  )
}

function TrainingSparkline({ history }) {
  if (!history || history.length < 2) {
    return <p className="mt-2 text-xs text-slate-400">No training history available.</p>
  }

  const values = history
    .map((row) => Number(row?.val_mse ?? row?.val_loss ?? Number.NaN))
    .filter((v) => Number.isFinite(v))

  if (values.length < 2) {
    return <p className="mt-2 text-xs text-slate-400">No validation curve available.</p>
  }

  const width = 280
  const height = 74
  const minVal = Math.min(...values)
  const maxVal = Math.max(...values)
  const span = Math.max(maxVal - minVal, 1e-9)

  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width
      const y = height - ((v - minVal) / span) * (height - 4) - 2
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  return (
    <div className="mt-3 rounded-lg border border-cyan-400/20 bg-slate-900/60 p-2">
      <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-cyan-200">Validation MSE Trend</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-[74px] w-full">
        <polyline fill="none" stroke="#67e8f9" strokeWidth="2" points={points} />
      </svg>
    </div>
  )
}

function apiUrl(path) {
  const base = import.meta.env.VITE_API_BASE?.trim()
  if (base) {
    return `${base.replace(/\/$/, '')}${path}`
  }
  return path
}

// ── Congestion colour helper ───────────────────────────────────────────────────
const CONG_STYLE = {
  'Free Flow': { bg: 'rgba(34,197,94,0.18)', text: '#86efac', border: 'rgba(34,197,94,0.4)' },
  'Moderate': { bg: 'rgba(234,179,8,0.18)', text: '#fde047', border: 'rgba(234,179,8,0.4)' },
  'Heavy': { bg: 'rgba(249,115,22,0.18)', text: '#fb923c', border: 'rgba(249,115,22,0.4)' },
  'Standstill': { bg: 'rgba(239,68,68,0.18)', text: '#f87171', border: 'rgba(239,68,68,0.4)' },
}

function CongestionBadge({ level }) {
  const s = CONG_STYLE[level] ?? CONG_STYLE['Moderate']
  return (
    <span style={{
      background: s.bg,
      color: s.text,
      border: `1px solid ${s.border}`,
      borderRadius: '9999px',
      padding: '1px 8px',
      fontSize: '0.68rem',
      fontWeight: 700,
      letterSpacing: '0.04em',
      whiteSpace: 'nowrap',
    }}>
      {level}
    </span>
  )
}

function TrendArrow({ trend }) {
  const map = { '↑': '#22d3ee', '↓': '#f87171', '→': '#94a3b8' }
  return (
    <span style={{ color: map[trend] ?? '#94a3b8', fontWeight: 700, fontSize: '1rem' }}>
      {trend}
    </span>
  )
}

function MiniUtilBar({ flow, capacity }) {
  const util = Math.min(100, Math.round((flow / (capacity || 1)) * 100))
  const color = util > 74 ? '#ef4444' : util > 54 ? '#f97316' : util > 29 ? '#facc15' : '#22c55e'
  return (
    <div style={{ marginTop: 4, width: '100%', height: 3, background: 'rgba(255,255,255,0.08)', borderRadius: 4, overflow: 'hidden' }}>
      <div style={{ width: `${util}%`, height: '100%', background: color, transition: 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)' }} />
    </div>
  )
}

function NodeForecastTable({ selectedNode, onNodeSelect, nodeOverrides, onForecastData, params, nodes, isSimulating, simStep }) {
  const [forecast, setForecast] = useState(null)
  const [loading, setLoading] = useState(true)
  const [inferring, setInferring] = useState(false)
  const [lastTs, setLastTs] = useState('')
  const rowRefs = useRef({})
  const debounceRef = useRef(null)

  // POST overrides to the model and receive real predictions
  const fetchWithOverrides = async (overridesPayload, globalParams, stepOffset = 0) => {
    setInferring(true)
    try {
      // Translate nodeOverrides
      const apiOverrides = {}
      Object.entries(overridesPayload).forEach(([idx, vals]) => {
        const entry = {}
        if (vals.trafficFlow !== undefined) entry.traffic_flow = vals.trafficFlow
        if (vals.avgSpeed !== undefined) entry.avg_speed = vals.avgSpeed
        if (Object.keys(entry).length > 0) apiOverrides[String(idx)] = entry
      })

      // Map global params
      const apiGlobal = {
        rain: globalParams.rain,
        temp: globalParams.temperature,
        hour: globalParams.timeOfDay,
        wind: globalParams.wind,
      }

      const resp = await fetch(apiUrl('/api/nodes/forecast'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          overrides: apiOverrides,
          global_params: apiGlobal,
          step_offset: stepOffset,
          n_steps: 3
        }),
      })
      if (!resp.ok) return
      const data = await resp.json()
      const forecastNodes = data.nodes ?? []
      setForecast(forecastNodes)
      setLastTs(data.last_updated ?? '')
      if (onForecastData) onForecastData(forecastNodes)
    } catch (_) {
      // network error
    } finally {
      setLoading(false)
      setInferring(false)
    }
  }

  // Initial load — one request on mount
  useEffect(() => {
    fetchWithOverrides({}, params, 0)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Re-inference: fires when sliders change (params/overrides) OR when simulation
  // advances a step. Debounced so rapid slider drags only fire once.
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      fetchWithOverrides(nodeOverrides, params, simStep)
    }, 350)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeOverrides, params, simStep])

  // Scroll selected node into view whenever it changes
  useEffect(() => {
    const key = `N${String(selectedNode + 1).padStart(3, '0')}`
    const el = rowRefs.current[key]
    if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [selectedNode])

  const selectedNodeId = `N${String(selectedNode + 1).padStart(3, '0')}`
  const hasOverrides = Object.keys(nodeOverrides).length > 0

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '10px 16px 8px',
        borderBottom: '1px solid rgba(100,200,255,0.12)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: '#34d399',
            boxShadow: '0 0 10px #34d399',
            display: 'inline-block',
            animation: 'pulse 2s infinite',
          }} />
          <p style={{ fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.15em', textTransform: 'uppercase', color: '#67e8f9', margin: 0 }}>
            5-Second Ahead Forecast - Spatiotemporal QGCN Model
          </p>
        </div>
        <p style={{ fontSize: '0.65rem', color: inferring ? '#fbbf24' : '#64748b', margin: 0 }}>
          {loading ? 'Running model…' : inferring ? '⟳ Re-running inference…' : `Updated ${lastTs.slice(11, 16)} UTC`}
        </p>
      </div>

      {/* Table */}
      <div style={{ overflowY: 'auto', flex: 1, padding: '0 4px' }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '0.75rem',
        }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 2 }}>
            <tr style={{ background: '#0b1829' }}>
              {['Node', 'Zone', 'Current (t)', '+5s', '+10s', '+15s'].map((h) => (
                <th key={h} style={{
                  padding: '8px 10px',
                  textAlign: 'left',
                  fontSize: '0.63rem',
                  fontWeight: 700,
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  color: '#475569',
                  borderBottom: '1px solid rgba(100,200,255,0.1)',
                  whiteSpace: 'nowrap',
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(forecast ?? []).slice(0, nodes.length).map((node, idx) => {
              const isSelected = node.node_id === selectedNodeId
              return (
                <tr
                  key={node.node_id}
                  ref={(el) => { rowRefs.current[node.node_id] = el }}
                  onClick={() => onNodeSelect(idx)}
                  style={{
                    cursor: 'pointer',
                    background: isSelected
                      ? 'rgba(34,211,238,0.10)'
                      : idx % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent',
                    boxShadow: isSelected ? 'inset 0 0 0 1.5px rgba(34,211,238,0.6)' : 'none',
                    transition: 'background 0.2s, box-shadow 0.2s',
                  }}
                >
                  <td style={{
                    padding: '5px 10px',
                    fontFamily: 'monospace',
                    fontWeight: isSelected ? 800 : 500,
                    color: isSelected ? '#e0f7ff' : '#cbd5e1',
                    borderBottom: '1px solid rgba(255,255,255,0.03)',
                  }}>
                    {isSelected && <span style={{ color: '#22d3ee', marginRight: 4 }}>▶</span>}
                    {node.node_id}
                  </td>
                  <td style={{ padding: '5px 10px', color: '#94a3b8', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                    {node.zone}
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid rgba(255,255,255,0.03)', minWidth: 80 }}>
                    <div style={{ color: '#e2e8f0', fontWeight: 600 }}>{Math.round(node.flow_t).toLocaleString()}</div>
                    <MiniUtilBar flow={node.flow_t} capacity={node.capacity_veh_per_hr} />
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid rgba(255,255,255,0.03)', minWidth: 80 }}>
                    <div style={{ color: '#cbd5e1' }}>{Math.round(node.predictions?.[0] ?? 0).toLocaleString()}</div>
                    <MiniUtilBar flow={node.predictions?.[0] ?? 0} capacity={node.capacity_veh_per_hr} />
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid rgba(255,255,255,0.03)', minWidth: 80 }}>
                    <div style={{ color: '#94a3b8' }}>{Math.round(node.predictions?.[1] ?? 0).toLocaleString()}</div>
                    <MiniUtilBar flow={node.predictions?.[1] ?? 0} capacity={node.capacity_veh_per_hr} />
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid rgba(255,255,255,0.03)', minWidth: 80 }}>
                    <div style={{ color: '#64748b' }}>{Math.round(node.predictions?.[2] ?? 0).toLocaleString()}</div>
                    <MiniUtilBar flow={node.predictions?.[2] ?? 0} capacity={node.capacity_veh_per_hr} />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {loading && (
          <p style={{ textAlign: 'center', color: '#475569', fontSize: '0.75rem', padding: '16px 0' }}>
            Running ST-QGCN inference…
          </p>
        )}
      </div>
    </div>
  )
}

function ReportsView({ activeRun, setActiveRun, runs, isLoadingApi, apiError, history, onBack }) {
  const [plots, setPlots] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const fetchPlots = async () => {
      try {
        setLoading(true)
        const resp = await fetch(apiUrl(`/api/runs/${activeRun}/plots`))
        if (!resp.ok) throw new Error('Failed to fetch plots')
        const data = await resp.json()
        setPlots(data.plots || [])
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    if (activeRun) fetchPlots()
  }, [activeRun])

  return (
    <div className="mx-auto flex h-full w-full max-w-[1400px] flex-col px-5 pb-6 pt-6 md:px-8">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-cyan-300/90">Run Reports</p>
          <h2 className="mt-1 font-display text-3xl font-semibold text-white">Simulation Graphs</h2>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-6 overflow-hidden md:flex-row">
        <div className="glass-panel w-full flex-shrink-0 overflow-y-auto rounded-3xl border border-slate-200/15 p-6 md:w-[360px]">
          <h3 className="mb-4 font-display text-xl text-white">Run Details</h3>
          <div className="rounded-xl border border-cyan-300/20 bg-slate-900/50 p-4">
            {/* Connected Run selector removed for cleaner UI */}
            <TrainingSparkline history={history} />
          </div>

          <button
            onClick={onBack}
            className="mt-6 w-full rounded-xl border border-cyan-300/30 bg-cyan-500/10 px-4 py-3 font-semibold text-cyan-100 transition hover:bg-cyan-500/20"
          >
            &larr; Back to Simulation
          </button>
        </div>

        <div className="glass-panel flex-1 overflow-y-auto rounded-3xl border border-slate-200/15 p-6">
          {loading ? (
            <p className="text-slate-400">Loading graphs...</p>
          ) : error ? (
            <p className="text-rose-400">{error}</p>
          ) : plots.length === 0 ? (
            <p className="text-slate-400">No graphs found for this run.</p>
          ) : (
            <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
              {plots.map((plot) => (
                <div key={plot.name} className="flex flex-col rounded-2xl border border-slate-700/50 bg-slate-900/40 p-4">
                  <h3 className="mb-3 text-sm font-semibold capitalize text-cyan-100">{plot.name.replace(/_/g, ' ').replace(/\.[^/.]+$/, '')}</h3>
                  <div className="flex flex-1 items-center justify-center rounded-xl bg-slate-950 p-2">
                    <img src={apiUrl(plot.url)} alt={plot.name} className="max-h-[400px] w-full object-contain" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [params, setParams] = useState(DEFAULTS)
  const [runs, setRuns] = useState([])
  const [activeRun, setActiveRun] = useState('')
  const [metrics, setMetrics] = useState(null)
  const [history, setHistory] = useState([])
  const [apiError, setApiError] = useState('')
  const [isLoadingApi, setIsLoadingApi] = useState(true)

  const [selectedNode, setSelectedNode] = useState(0)
  const [nodeOverrides, setNodeOverrides] = useState({})
  const [showMetrics, setShowMetrics] = useState(true)
  const [currentView, setCurrentView] = useState('simulation')

  // Simulation State
  const [isSimulating, setIsSimulating] = useState(false)
  const [simStep, setSimStep] = useState(0)

  // Simulation ticker — only advances simStep (and therefore triggers inference)
  // while the user has pressed "Start Real-Time Simulation".
  useEffect(() => {
    if (!isSimulating) return
    const interval = setInterval(() => {
      setSimStep((s) => s + 1)
    }, LIVE_REFRESH_MS)
    return () => clearInterval(interval)
  }, [isSimulating])

  const network = useMemo(() => buildComplexNetwork(), [])
  const nodes = network.junctionNodes
  const edges = network.junctionEdges
  const roadSegments = network.roadSegments

  // forecastData holds the last model response; drives both 3-D scene and table
  const [forecastData, setForecastData] = useState([])

  // nodeState is derived entirely from the model's forecast output
  const nodeState = useMemo(() => {
    const n = nodes.length
    if (forecastData.length === 0) {
      // Before first model response: neutral state
      return Array.from({ length: n }, () => ({ flow: 0, speed: 0, congestion: 0 }))
    }
    // Build a map from node_index → forecast row
    const byIndex = {}
    forecastData.forEach((row) => {
      byIndex[row.node_index] = row
    })
    return Array.from({ length: n }, (_, idx) => {
      const row = byIndex[idx]
      if (!row) return { flow: 0, speed: 0, congestion: 0 }
      // Use current flow_t for the map visualization so it responds instantly to sliders
      const flow = row.flow_t
      const capacity = row.capacity_veh_per_hr
      const ratio = capacity > 0 ? clamp(flow / capacity, 0, 1) : 0
      return { flow, speed: 0, congestion: ratio }
    })
  }, [forecastData, nodes.length])

  const networkLoad = useMemo(() => {
    if (nodeState.length === 0) return 0
    return nodeState.reduce((sum, n) => sum + n.congestion, 0) / nodeState.length
  }, [nodeState])

  const setParam = (key) => (val) => {
    setParams((prev) => ({ ...prev, [key]: val }))
  }

  const setSelectedNodeParam = (key, value) => {
    if (selectedNode === null || selectedNode === undefined) return
    setNodeOverrides((prev) => {
      const forecastRow = forecastData.find((r) => r.node_index === selectedNode)
      const defaultFlow = forecastRow ? Math.round(forecastRow.flow_t) : 500
      const current = prev[selectedNode] ?? { trafficFlow: defaultFlow, avgSpeed: 42 }
      return {
        ...prev,
        [selectedNode]: {
          ...current,
          [key]: value,
        },
      }
    })
  }

  useEffect(() => {
    const loadRuns = async () => {
      try {
        setApiError('')
        setIsLoadingApi(true)
        const resp = await fetch(apiUrl('/api/runs'))
        if (!resp.ok) throw new Error(`Runs API failed (${resp.status})`)
        const data = await resp.json()
        const runNames = (data.runs ?? []).map((r) => r.name)
        setRuns(runNames)

        if (data.default_run) {
          setActiveRun(data.default_run)
        } else if (runNames.length > 0) {
          setActiveRun(runNames[0])
        } else {
          setApiError('No runs found in backend/runs.')
        }
      } catch (err) {
        setApiError(err.message || 'Failed to connect backend API.')
      } finally {
        setIsLoadingApi(false)
      }
    }

    loadRuns()
  }, [])

  useEffect(() => {
    if (!activeRun) return

    const loadRunPayload = async () => {
      try {
        setApiError('')
        const [metricsResp, historyResp] = await Promise.all([
          fetch(apiUrl(`/api/runs/${activeRun}/metrics`)),
          fetch(apiUrl(`/api/runs/${activeRun}/history`)),
        ])

        if (!metricsResp.ok) throw new Error(`Metrics API failed (${metricsResp.status})`)
        if (!historyResp.ok) throw new Error(`History API failed (${historyResp.status})`)

        const metricsData = await metricsResp.json()
        const historyData = await historyResp.json()

        setMetrics(metricsData.metrics ?? null)
        setHistory(historyData.history ?? [])
      } catch (err) {
        setApiError(err.message || 'Failed to load run data.')
      }
    }

    loadRunPayload()
    const timer = setInterval(loadRunPayload, LIVE_REFRESH_MS)
    return () => clearInterval(timer)
  }, [activeRun])

  // Default slider values for a node come from the latest model forecast
  const selectedForecastRow = forecastData.find((r) => r.node_index === selectedNode)
  const selectedState = nodeOverrides[selectedNode] ?? {
    trafficFlow: selectedForecastRow ? Math.round(selectedForecastRow.flow_t) : 0,
    avgSpeed: 0,
  }
  const selectedNodeName = selectedNode !== null && selectedNode !== undefined
    ? `JUNC-${String(selectedNode + 1).padStart(3, '0')}`
    : 'None'

  const selectedUtil = selectedForecastRow ? (selectedForecastRow.predictions?.[0] ?? selectedForecastRow.flow_t) / (selectedForecastRow.capacity_veh_per_hr || 1) * 100 : 0
  const liveHeatPressure = Math.round(clamp(selectedUtil, 0, 100))
  const liveWeatherSeverity = Math.round((params.rain + params.wind) / 2)
  const bestVal = Number(metrics?.best_val_mse)
  const testMse = Number(metrics?.test_mse)
  const testMae = Number(metrics?.test_mae)
  const bestEpoch = Number(metrics?.best_epoch)
  const dayFactor = toDaylightFactor(params.timeOfDay)

  const resetAll = () => {
    setParams(DEFAULTS)
    setNodeOverrides({})
    setIsSimulating(false)
    setSimStep(0)
  }

  return (
    <div className="h-screen overflow-hidden bg-night-grid text-slate-100">
      {currentView === 'reports' ? (
        <ReportsView
          activeRun={activeRun}
          setActiveRun={setActiveRun}
          runs={runs}
          isLoadingApi={isLoadingApi}
          apiError={apiError}
          history={history}
          onBack={() => setCurrentView('simulation')}
        />
      ) : (
        <>
          <motion.header
            initial={{ y: -28, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.55, ease: 'easeOut' }}
            className="mx-auto flex w-full max-w-[1400px] items-center justify-between px-5 pb-4 pt-6 md:px-8"
          >
            <div>
              <p className="font-mono text-xs uppercase tracking-[0.2em] text-cyan-300/90">Spatiotemporal-QGCN Urban Digital Twin</p>
              <h1 className="mt-2 font-display text-3xl font-semibold leading-tight text-white md:text-4xl">
                Chembur Network Command Deck
              </h1>
            </div>
          </motion.header>

          <main className="mx-auto grid h-[calc(100vh-118px)] w-full max-w-[1400px] gap-5 px-5 pb-6 md:grid-cols-[360px_1fr] md:px-8">
            <motion.aside
              initial={{ x: -24, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="glass-panel h-full overflow-y-auto rounded-3xl p-5 shadow-2xl shadow-black/30"
            >
              <h2 className="font-display text-xl text-white">Simulation Controls</h2>
              <p className="mt-1 text-sm text-slate-300">

              </p>

              <div className="mt-6">
                <button
                  type="button"
                  onClick={() => setIsSimulating(!isSimulating)}
                  className={`w-full rounded-xl px-4 py-3 font-semibold transition ${isSimulating ? 'bg-rose-500/20 border border-rose-300/30 text-rose-100' : 'bg-cyan-500/10 border border-cyan-300/30 text-cyan-100'}`}
                >
                  {isSimulating ? `Stop Simulation (Step ${simStep})` : 'Start Real-Time Simulation'}
                </button>
              </div>

              <div className="mt-6 space-y-5">
                <Slider label="Rain" value={params.rain} min={0} max={100} step={1} unit="%" onChange={setParam('rain')} />
                <Slider label="Wind" value={params.wind} min={0} max={60} step={1} unit=" km/h" onChange={setParam('wind')} />
                <Slider label="Time of Day" value={params.timeOfDay} min={0} max={23} step={1} unit=":00" onChange={setParam('timeOfDay')} />
                <Slider label="Temperature" value={params.temperature} min={-5} max={45} step={1} unit=" C" onChange={setParam('temperature')} />
              </div>

              <div className="mt-6 rounded-xl border border-amber-300/25 bg-slate-900/60 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-amber-200">Selected Node</p>
                <p className="mt-1 text-sm font-semibold text-white">{selectedNodeName}</p>
                <p className="mt-1 text-xs text-slate-300">Tap a node in the scene to edit local parameters.</p>
                <div className="mt-4 space-y-4">
                  <Slider
                    label="Traffic Flow"
                    value={selectedState.trafficFlow}
                    min={50}
                    max={2500}
                    step={10}
                    unit=" veh/hr"
                    onChange={(v) => setSelectedNodeParam('trafficFlow', v)}
                    disabled={isSimulating}
                  />
                  <Slider
                    label="Avg Speed"
                    value={selectedState.avgSpeed}
                    min={5}
                    max={100}
                    step={1}
                    unit=" kmph"
                    onChange={(v) => setSelectedNodeParam('avgSpeed', v)}
                    disabled={isSimulating}
                  />
                </div>
              </div>

              <button
                type="button"
                onClick={resetAll}
                className="mt-6 w-full rounded-xl border border-cyan-300/30 bg-cyan-500/10 px-4 py-3 font-semibold text-cyan-100 transition hover:bg-cyan-500/20"
              >
                Reset Scenario
              </button>

              <button
                type="button"
                onClick={() => setCurrentView('reports')}
                className="mt-3 w-full rounded-xl border border-emerald-300/30 bg-emerald-500/10 px-4 py-3 font-semibold text-emerald-100 transition hover:bg-emerald-500/20"
              >
                View Generated Graphs
              </button>
            </motion.aside>

            <motion.section
              initial={{ y: 24, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.7, delay: 0.15 }}
              className="glass-panel relative h-full overflow-hidden rounded-3xl border border-slate-200/15"
              style={{ display: 'flex', flexDirection: 'column' }}
            >
              <div style={{ position: 'relative', flex: '0 0 58%', overflow: 'hidden' }}>
                <div className="h-full min-h-[320px] w-full">
                  <Canvas shadows gl={{ antialias: true }} dpr={[1, 1.8]}>
                    <PerspectiveCamera makeDefault position={[14, 14, 20]} fov={52} />
                    <NetworkScene
                      params={params}
                      selectedNode={selectedNode}
                      onNodeSelect={setSelectedNode}
                      network={network}
                      nodeState={nodeState}
                      networkLoad={networkLoad}
                    />
                  </Canvas>
                </div>
              </div>

              <div style={{
                height: 1,
                background: 'linear-gradient(90deg, transparent, rgba(103,232,249,0.25), transparent)',
                flexShrink: 0,
              }} />

              <div style={{
                flex: '1 1 0',
                overflow: 'hidden',
                background: 'rgba(4, 16, 34, 0.75)',
                backdropFilter: 'blur(12px)',
              }}>
                <NodeForecastTable
                  selectedNode={selectedNode}
                  onNodeSelect={setSelectedNode}
                  nodeOverrides={nodeOverrides}
                  onForecastData={setForecastData}
                  params={params}
                  nodes={nodes}
                  isSimulating={isSimulating}
                  simStep={simStep}
                />
              </div>
            </motion.section>
          </main>
        </>
      )}
    </div>
  )
}

export default App
