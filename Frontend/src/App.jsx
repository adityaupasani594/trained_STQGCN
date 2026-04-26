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
        const seed = ((idx * 37) % 100) / 100
        const congestion = nodeState[idx].congestion
        const heat = clamp(seed * 0.2 + congestion * 0.75 + networkLoad * 0.18, 0, 1)
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
            <sphereGeometry args={[0.28 + heat * 0.2 + (isSelected ? 0.08 : 0), 20, 20]} />
            <meshStandardMaterial
              color={isSelected ? '#d6f6ff' : heatColor(heat)}
              emissive={heatColor(heat)}
              emissiveIntensity={0.25 + heat * 0.68 + (isSelected ? 0.2 : 0)}
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

  const network = useMemo(() => buildComplexNetwork(), [])
  const nodes = network.junctionNodes
  const edges = network.junctionEdges
  const roadSegments = network.roadSegments

  const nodeState = useMemo(() => {
    const n = nodes.length
    const baseFlow = 420
    const baseSpeed = 42
    const depthWeights = [1, 0.55, 0.25]

    const neighbors = Array.from({ length: n }, () => [])
    edges.forEach(({ a, b }) => {
      neighbors[a].push(b)
      neighbors[b].push(a)
    })

    const flowDelta = Array.from({ length: n }, () => 0)
    const speedDelta = Array.from({ length: n }, () => 0)

    Object.entries(nodeOverrides).forEach(([k, override]) => {
      const source = Number(k)
      if (!Number.isInteger(source) || source < 0 || source >= n) return

      const sourceFlowDelta = (override.trafficFlow ?? baseFlow) - baseFlow
      const sourceSpeedDelta = (override.avgSpeed ?? baseSpeed) - baseSpeed

      const visitedDepth = Array.from({ length: n }, () => Infinity)
      const queue = [[source, 0]]
      visitedDepth[source] = 0

      while (queue.length > 0) {
        const [nodeIdx, depth] = queue.shift()
        if (depth > 2) continue

        const w = depthWeights[depth]
        flowDelta[nodeIdx] += sourceFlowDelta * w
        speedDelta[nodeIdx] += sourceSpeedDelta * w

        if (depth === 2) continue
        neighbors[nodeIdx].forEach((nb) => {
          if (visitedDepth[nb] > depth + 1) {
            visitedDepth[nb] = depth + 1
            queue.push([nb, depth + 1])
          }
        })
      }
    })

    return Array.from({ length: n }, (_, idx) => {
      const effectiveFlow = clamp(baseFlow + flowDelta[idx], 50, 1200)
      const effectiveSpeed = clamp(baseSpeed + speedDelta[idx], 5, 100)
      const flowNorm = clamp(effectiveFlow / 900, 0, 1.5)
      const speedNorm = clamp(effectiveSpeed / 80, 0, 1.2)
      const congestion = clamp(flowNorm * 0.55 + (1 - speedNorm) * 0.45, 0, 1.5)
      return {
        flow: effectiveFlow,
        speed: effectiveSpeed,
        congestion,
      }
    })
  }, [edges, nodeOverrides, nodes.length])

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
      const current = prev[selectedNode] ?? { trafficFlow: 420, avgSpeed: 42 }
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
    const timer = setInterval(loadRunPayload, 15000)
    return () => clearInterval(timer)
  }, [activeRun])

  const selectedState = nodeOverrides[selectedNode] ?? {
    trafficFlow: selectedNode !== null && selectedNode !== undefined && nodeState ? Math.round(nodeState[selectedNode]?.flow ?? 420) : 420,
    avgSpeed: selectedNode !== null && selectedNode !== undefined && nodeState ? Math.round(nodeState[selectedNode]?.speed ?? 42) : 42,
  }
  const selectedNodeName = selectedNode !== null && selectedNode !== undefined
    ? `JUNC-${String(selectedNode + 1).padStart(3, '0')}`
    : 'None'

  const liveHeatPressure = Math.round(
    clamp(((selectedState.trafficFlow / 1200) * 100 + (1 - selectedState.avgSpeed / 100) * 100) / 2, 0, 100),
  )
  const liveWeatherSeverity = Math.round((params.rain + params.wind) / 2)
  const bestVal = Number(metrics?.best_val_mse)
  const testMse = Number(metrics?.test_mse)
  const testMae = Number(metrics?.test_mae)
  const bestEpoch = Number(metrics?.best_epoch)
  const dayFactor = toDaylightFactor(params.timeOfDay)


  return (
    <div className="h-screen overflow-hidden bg-night-grid text-slate-100">
      <motion.header
        initial={{ y: -28, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.55, ease: 'easeOut' }}
        className="mx-auto flex w-full max-w-[1400px] items-center justify-between px-5 pb-4 pt-6 md:px-8"
      >
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-cyan-300/90">ST-QGCN Urban Digital Twin</p>
          <h1 className="mt-2 font-display text-3xl font-semibold leading-tight text-white md:text-4xl">
            Chembur Network Command Deck
          </h1>
        </div>
        <div className="hidden items-center gap-3 rounded-full border border-cyan-300/30 bg-cyan-900/30 px-4 py-2 text-xs font-semibold text-cyan-100 md:flex">
          <span className="h-2 w-2 rounded-full bg-emerald-300 shadow-[0_0_16px_#34d399]" />
          LIVE NETWORK FEED
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
            Click any node in the network to tune local traffic behavior.
          </p>

          <div className="mt-4 rounded-xl border border-cyan-300/20 bg-slate-900/50 p-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-cyan-200">Connected Run</p>
            <select
              className="mt-2 w-full rounded-lg border border-slate-500/40 bg-slate-900 px-3 py-2 text-sm text-slate-100"
              value={activeRun}
              onChange={(e) => setActiveRun(e.target.value)}
              disabled={runs.length === 0}
            >
              {runs.map((runName) => (
                <option key={runName} value={runName}>
                  {runName}
                </option>
              ))}
            </select>
            {isLoadingApi ? <p className="mt-2 text-xs text-slate-400">Loading backend runs...</p> : null}
            {apiError ? <p className="mt-2 text-xs text-rose-300">{apiError}</p> : null}
            <TrainingSparkline history={history} />
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
                label="Predicted Traffic Density"
                value={selectedState.trafficFlow}
                min={50}
                max={1200}
                step={1}
                unit=" veh/km"
                onChange={(v) => setSelectedNodeParam('trafficFlow', v)}
              />
              <Slider
                label="Predicted Avg Speed"
                value={selectedState.avgSpeed}
                min={5}
                max={100}
                step={1}
                unit=" kmph"
                onChange={(v) => setSelectedNodeParam('avgSpeed', v)}
              />
            </div>
          </div>

          <button
            type="button"
            onClick={() => {
              setParams(DEFAULTS)
              setNodeOverrides({})
            }}
            className="mt-6 w-full rounded-xl border border-cyan-300/30 bg-cyan-500/10 px-4 py-3 font-semibold text-cyan-100 transition hover:bg-cyan-500/20"
          >
            Reset Scenario
          </button>
        </motion.aside>

        <motion.section
          initial={{ y: 24, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          className="glass-panel relative h-full overflow-hidden rounded-3xl border border-slate-200/15"
        >
          <div className="absolute left-4 top-4 z-10 rounded-xl border border-amber-300/20 bg-slate-900/70 px-4 py-3 text-xs text-slate-200 backdrop-blur">
            <p className="font-semibold tracking-wide text-amber-200">Live Metrics</p>
            <p className="mt-1">Heat Pressure: {liveHeatPressure}%</p>
            <p>Weather Severity: {liveWeatherSeverity}%</p>
            <p>Daylight: {Math.round(dayFactor * 100)}%</p>
            <p className="mt-2 text-cyan-200">Backend Best Val MSE: {Number.isFinite(bestVal) ? bestVal.toFixed(4) : 'n/a'}</p>
            <p>Backend Test MSE: {Number.isFinite(testMse) ? testMse.toFixed(4) : 'n/a'}</p>
            <p>Backend Test MAE: {Number.isFinite(testMae) ? testMae.toFixed(4) : 'n/a'}</p>
            <p>Best Epoch: {Number.isFinite(bestEpoch) ? bestEpoch : 'n/a'}</p>
          </div>

          <div className="h-full min-h-[500px] w-full">
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
        </motion.section>
      </main>
    </div>
  )
}

export default App
