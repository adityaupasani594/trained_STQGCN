import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Line, OrbitControls, PerspectiveCamera, Html } from '@react-three/drei'
import { motion } from 'framer-motion'
import { useEffect, useMemo, useRef, useState } from 'react'

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULTS = {
  rain: 36,
  wind: 24,
  temperature: 28,
  quantumDepth: 5,
  speedFactor: 1.2,
  timeOfDay: 14,
}
const LIVE_REFRESH_MS = 5_000

// ─────────────────────────────────────────────────────────────────────────────
//  Hierarchical STQGCN — Layer 0 Network Topology Definitions
//
//  Three road network sub-graphs with distinct topologies.
//  Each node owns exactly 1 qubit (per QuantumSTLayer N_QUBITS=1 in train_stqgcn.py).
//
//  Layer 0 (Micro)    — 3 sub-networks below
//  Layer 1 (Regional) — pooled from L0, Y = 5.5
//  Layer 2 (National) — pooled from L1, Y = 11.0
// ─────────────────────────────────────────────────────────────────────────────

const NETWORK_DEFS = {
  tree: {
    key: 'tree',
    label: 'Tree Network',
    shortLabel: 'Tree',
    icon: '🌳',
    // Hierarchical branching: root → 2 children → 4 leaves
    color:       '#00e5a0',
    emissive:    '#00b87a',
    edgeColor:   '#00cc8a',
    glowColor:   'rgba(0,229,160,0.35)',
    borderColor: 'rgba(0,200,140,0.55)',
    bgColor:     'rgba(0,229,160,0.07)',
    textColor:   '#007a52',
    nodes: [
      [-13,   0.2, 0   ],  // 0  root
      [-15.5, 0.2, 4.5 ],  // 1  left child
      [-10.5, 0.2, 4.5 ],  // 2  right child
      [-17,   0.2, 9   ],  // 3  leaf LL
      [-14,   0.2, 9   ],  // 4  leaf LR
      [-12,   0.2, 9   ],  // 5  leaf RL
      [-9,    0.2, 9   ],  // 6  leaf RR
    ],
    edges: [[0,1],[0,2],[1,3],[1,4],[2,5],[2,6]],
    center:     [-13, 0, 4.5],
    labelPos:   [-13, 2.8, -2.5],
    cameraPos:  [-8, 15, 22],
    cameraLook: [-13, 1, 4.5],
    description: 'Hierarchical branching topology',
    baseCongestion: [0.15, 0.22, 0.28, 0.18, 0.35, 0.42, 0.52],
  },
  line: {
    key: 'line',
    label: 'Line Network',
    shortLabel: 'Line',
    icon: '〰️',
    // Sequential linear chain: n0 — n1 — n2 — n3 — n4
    color:       '#60a5fa',
    emissive:    '#2563eb',
    edgeColor:   '#3b82f6',
    glowColor:   'rgba(96,165,250,0.35)',
    borderColor: 'rgba(59,130,246,0.55)',
    bgColor:     'rgba(96,165,250,0.07)',
    textColor:   '#1d4ed8',
    nodes: [
      [0, 0.2, -9  ],  // 0
      [0, 0.2, -4.5],  // 1
      [0, 0.2,  0  ],  // 2  center
      [0, 0.2,  4.5],  // 3
      [0, 0.2,  9  ],  // 4
    ],
    edges: [[0,1],[1,2],[2,3],[3,4]],
    center:     [0, 0, 0],
    labelPos:   [0, 2.8, -11.5],
    cameraPos:  [14, 14, 2],
    cameraLook: [0, 1, 0],
    description: 'Sequential linear chain',
    baseCongestion: [0.20, 0.40, 0.62, 0.48, 0.30],
  },
  ring: {
    key: 'ring',
    label: 'Ring Network',
    shortLabel: 'Ring',
    icon: '⭕',
    // Circular ring: n0 — n1 — n2 — n3 — n4 — n0
    color:       '#fbbf24',
    emissive:    '#d97706',
    edgeColor:   '#f59e0b',
    glowColor:   'rgba(251,191,36,0.35)',
    borderColor: 'rgba(217,119,6,0.55)',
    bgColor:     'rgba(251,191,36,0.07)',
    textColor:   '#92400e',
    nodes: [
      [12,   0.2, -4  ],  // 0  top
      [16.5, 0.2,  1  ],  // 1  right
      [14,   0.2,  7.5],  // 2  bottom-right
      [9.5,  0.2,  7.5],  // 3  bottom-left
      [7,    0.2,  1  ],  // 4  left
    ],
    edges: [[0,1],[1,2],[2,3],[3,4],[4,0]],
    center:     [11.5, 0, 3],
    labelPos:   [11.5, 2.8, -6.5],
    cameraPos:  [11.5, 15, 21],
    cameraLook: [11.5, 1, 3],
    description: 'Circular ring topology',
    baseCongestion: [0.35, 0.58, 0.72, 0.55, 0.30],
  },
}

const NETWORK_KEYS   = ['tree', 'line', 'ring']
const NETWORK_OFFSETS = { tree: 0, line: 7, ring: 12 }
const NETWORK_COUNTS  = { tree: 7, line: 5, ring: 5 }
const TOTAL_L0        = 17   // 7 + 5 + 5

// ─────────────────────────────────────────────────────────────────────────────
//  Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

function clamp(v, min, max) { return Math.min(max, Math.max(min, v)) }

function heatColor(t) {
  const v = clamp(t, 0, 1)
  if (v < 0.2) return '#00e5a0'
  if (v < 0.4) return '#ffe040'
  if (v < 0.6) return '#ff8c00'
  if (v < 0.8) return '#ff3333'
  return '#cc00ff'
}

function lerp(a, b, t) { return a + (b - a) * t }

function lerpHexColor(hexA, hexB, t) {
  const a = hexA.replace('#', ''), b = hexB.replace('#', '')
  const ar = parseInt(a.slice(0,2),16), ag = parseInt(a.slice(2,4),16), ab = parseInt(a.slice(4,6),16)
  const br = parseInt(b.slice(0,2),16), bg = parseInt(b.slice(2,4),16), bb = parseInt(b.slice(4,6),16)
  return `#${Math.round(lerp(ar,br,t)).toString(16).padStart(2,'0')}` +
         `${Math.round(lerp(ag,bg,t)).toString(16).padStart(2,'0')}` +
         `${Math.round(lerp(ab,bb,t)).toString(16).padStart(2,'0')}`
}

function toDaylightFactor(hour) {
  const w = ((hour % 24) + 24) % 24
  return clamp((Math.cos(((w - 12) / 12) * Math.PI) + 0.12) / 1.12, 0, 1)
}

// Return which network key owns global node index idx
function nodeNetworkKey(idx) {
  for (const k of NETWORK_KEYS) {
    if (idx >= NETWORK_OFFSETS[k] && idx < NETWORK_OFFSETS[k] + NETWORK_COUNTS[k]) return k
  }
  return null
}

function apiUrl(path) {
  const base = import.meta.env.VITE_API_BASE?.trim()
  return base ? `${base.replace(/\/$/, '')}${path}` : path
}

// ─────────────────────────────────────────────────────────────────────────────
//  Hierarchical Network Builder
//  Produces L0 (three topology sub-graphs) → L1 (regional) → L2 (national)
// ─────────────────────────────────────────────────────────────────────────────

function poolNodes(sourceNodes, poolRatio, targetY, maxDist) {
  const unassigned = new Set(sourceNodes.map((_, i) => i))
  const pooled = [], assignments = []

  while (unassigned.size > 0) {
    const seed = Array.from(unassigned)[0]
    unassigned.delete(seed)
    const group = [seed]

    while (group.length < poolRatio && unassigned.size > 0) {
      const cx = group.reduce((s, i) => s + sourceNodes[i][0], 0) / group.length
      const cz = group.reduce((s, i) => s + sourceNodes[i][2], 0) / group.length
      let best = -1, bestD = Infinity
      for (const idx of unassigned) {
        const dx = sourceNodes[idx][0] - cx, dz = sourceNodes[idx][2] - cz
        const d = dx * dx + dz * dz
        if (d < bestD) { bestD = d; best = idx }
      }
      if (best !== -1) { group.push(best); unassigned.delete(best) }
    }

    const cx = group.reduce((s, i) => s + sourceNodes[i][0], 0) / group.length
    const cz = group.reduce((s, i) => s + sourceNodes[i][2], 0) / group.length
    const ni = pooled.length
    pooled.push([cx, targetY, cz])
    group.forEach(c => assignments.push({ parent: ni, child: c }))
  }

  const edges = []
  for (let i = 0; i < pooled.length; i++)
    for (let j = i + 1; j < pooled.length; j++) {
      const dx = pooled[i][0] - pooled[j][0], dz = pooled[i][2] - pooled[j][2]
      if (Math.sqrt(dx * dx + dz * dz) < maxDist) edges.push({ a: i, b: j })
    }

  return { nodes: pooled, edges, assignments }
}

function buildHierarchicalNetwork() {
  const l0Nodes = [
    ...NETWORK_DEFS.tree.nodes,
    ...NETWORK_DEFS.line.nodes,
    ...NETWORK_DEFS.ring.nodes,
  ]

  const l0Edges = []
  for (const key of NETWORK_KEYS) {
    const def = NETWORK_DEFS[key], off = NETWORK_OFFSETS[key]
    for (const [a, b] of def.edges)
      l0Edges.push({ a: off + a, b: off + b, network: key })
  }

  // Layer 1 — Regional pooling: group ~3 L0 nodes → 1 L1 node (Y=5.5)
  const l1 = poolNodes(l0Nodes, 3, 5.5, 18.0)
  // Layer 2 — National pooling: group ~3 L1 nodes → 1 L2 node (Y=11.0)
  const l2 = poolNodes(l1.nodes, 3, 11.0, 30.0)

  return {
    l0Nodes, l0Edges,
    l1Nodes: l1.nodes, l1Edges: l1.edges, l1Assignments: l1.assignments,
    l2Nodes: l2.nodes, l2Edges: l2.edges, l2Assignments: l2.assignments,
    totalQubits: l0Nodes.length + l1.nodes.length + l2.nodes.length,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mock Forecast Nodes (used when API is unavailable)
// ─────────────────────────────────────────────────────────────────────────────

function generateMockForecastNodes() {
  const out = []
  for (const key of NETWORK_KEYS) {
    const def = NETWORK_DEFS[key]
    const off = NETWORK_OFFSETS[key]
    for (let i = 0; i < def.nodes.length; i++) {
      const idx = off + i
      const cong = def.baseCongestion[i] ?? 0.3
      const cap  = 2000
      const flow = Math.round(cong * cap)
      out.push({
        node_index: idx,
        node_id:    `N${String(idx + 1).padStart(3, '0')}`,
        zone:       def.label,
        capacity_veh_per_hr: cap,
        flow_t:     flow,
        predictions: [
          Math.round(flow * 1.025 + Math.sin(idx * 1.3) * 28),
          Math.round(flow * 1.055 + Math.sin(idx * 1.8) * 45),
          Math.round(flow * 1.090 + Math.sin(idx * 2.2) * 62),
        ],
      })
    }
  }
  return out
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rain Field
// ─────────────────────────────────────────────────────────────────────────────

function RainField({ rain, wind }) {
  const ref = useRef()
  const count = Math.floor((rain / 100) * 1500)
  const particles = useMemo(() => {
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      arr[i*3]   = -28 + Math.random() * 56
      arr[i*3+1] = Math.random() * 24
      arr[i*3+2] = -16 + Math.random() * 32
    }
    return arr
  }, [count])

  useFrame((_, dt) => {
    if (!ref.current) return
    const pos = ref.current.geometry.attributes.position.array
    const fs = 9 + (rain / 100) * 9, drift = (wind / 100) * 14
    for (let i = 0; i < count; i++) {
      pos[i*3+1] -= fs * dt
      pos[i*3]   += drift * dt
      if (pos[i*3+1] < 0) { pos[i*3+1] = 24; pos[i*3] = -32 + Math.random() * 64 }
    }
    ref.current.geometry.attributes.position.needsUpdate = true
  })

  if (count === 0) return null
  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={particles} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial color="#90c8ff" size={0.055} transparent opacity={0.32} />
    </points>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
//  Scene Controller — camera smooth-fly + OrbitControls
// ─────────────────────────────────────────────────────────────────────────────

function SceneController({ focus }) {
  const { camera } = useThree()
  const ctrlRef = useRef()

  useFrame(() => {
    if (!focus || !ctrlRef.current) return
    const [px, py, pz] = focus.pos
    const [lx, ly, lz] = focus.look
    const SPEED = 0.045
    camera.position.x += (px - camera.position.x) * SPEED
    camera.position.y += (py - camera.position.y) * SPEED
    camera.position.z += (pz - camera.position.z) * SPEED
    ctrlRef.current.target.x += (lx - ctrlRef.current.target.x) * SPEED
    ctrlRef.current.target.y += (ly - ctrlRef.current.target.y) * SPEED
    ctrlRef.current.target.z += (lz - ctrlRef.current.target.z) * SPEED
    ctrlRef.current.update()
  })

  return (
    <OrbitControls
      ref={ctrlRef}
      enableDamping
      dampingFactor={0.08}
      maxPolarAngle={Math.PI / 2.05}
      minDistance={10}
      maxDistance={52}
    />
  )
}

// ─────────────────────────────────────────────────────────────────────────────
//  Network Scene — 3-D rendering of all three hierarchy layers
// ─────────────────────────────────────────────────────────────────────────────

function NetworkScene({ params, selectedNode, onNodeSelect, selectedNetwork, network, nodeState, networkLoad, cameraFocus }) {
  const l0Refs   = useRef([])
  const qRefs    = useRef([])   // qubit rings for L0 nodes
  const l1Refs   = useRef([])
  const l2Refs   = useRef([])
  const l1qRefs  = useRef([])   // qubit rings for L1 nodes
  const l2qRefs  = useRef([])   // qubit rings for L2 nodes
  const day = toDaylightFactor(params.timeOfDay)

  const bgColor = lerpHexColor('#04090f', '#0b1a2e', day * 0.55)
  const ambient = 0.35 + day * 0.4

  useFrame((state) => {
    const t = state.clock.getElapsedTime() * params.speedFactor

    // L0 node pulse — stronger pulse with higher congestion
    l0Refs.current.forEach((mesh, idx) => {
      if (!mesh) return
      const cong = nodeState[idx]?.congestion ?? 0
      const w = 1 + Math.sin(t * 2 + idx * 0.22) * (0.07 + cong * 0.20)
      mesh.scale.set(w, w, w)
    })

    // Qubit rings — orbital spin
    qRefs.current.forEach((mesh, idx) => {
      if (!mesh) return
      mesh.rotation.z = t * 1.8 + idx * 0.55
      mesh.rotation.x = Math.sin(t * 0.85 + idx * 0.38) * 0.38
    })

    // L1 node rotation
    l1Refs.current.forEach((mesh, idx) => {
      if (!mesh) return
      mesh.rotation.y = t * 0.55
      mesh.rotation.x = Math.sin(t * 0.9 + idx) * 0.12
    })

    // L1 qubit rings
    l1qRefs.current.forEach((mesh, idx) => {
      if (!mesh) return
      mesh.rotation.y = t * 1.2 + idx * 0.7
    })

    // L2 node + rings
    l2Refs.current.forEach((mesh, idx) => {
      if (!mesh) return
      mesh.rotation.y = -t * 0.85
      const w = 1 + Math.sin(t * 2.8 + idx) * 0.09
      mesh.scale.set(w, w, w)
    })
    l2qRefs.current.forEach((mesh, idx) => {
      if (!mesh) return
      mesh.rotation.z = t * 0.8 + idx
      mesh.rotation.x = Math.cos(t * 0.6 + idx) * 0.25
    })
  })

  return (
    <>
      <color attach="background" args={[bgColor]} />
      <fog attach="fog" args={[bgColor, 22, 60]} />

      {/* Scene lighting */}
      <ambientLight intensity={ambient} color="#7090b8" />
      <directionalLight position={[8 + day*10, 14 + day*16, 6]} intensity={0.3 + day * 1.0} color="#c8e8ff" castShadow />
      <pointLight position={[0, 20, 0]} intensity={1.2 + networkLoad * 0.8} color="#a0c8ff" />
      {/* Per-network colour fill lights */}
      <pointLight position={[-13, 7, 4.5]} intensity={0.7} color="#00e5a0" />
      <pointLight position={[0,   7, 0  ]} intensity={0.7} color="#60a5fa" />
      <pointLight position={[11.5,7, 3  ]} intensity={0.7} color="#fbbf24" />
      {/* L1/L2 accent lights */}
      <pointLight position={[0, 5.5, 0]} intensity={0.5} color="#8b5cf6" />
      <pointLight position={[0, 11,  0]} intensity={0.6} color="#ec4899" />

      {/* Ground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.08, 0]} receiveShadow>
        <planeGeometry args={[70, 42]} />
        <meshStandardMaterial color={lerpHexColor('#060c1a', '#101e30', day * 0.35)} roughness={0.96} metalness={0.04} />
      </mesh>

      {/* Network zone halos (subtle circles on ground) */}
      {NETWORK_KEYS.map(k => {
        const d = NETWORK_DEFS[k]
        const active = selectedNetwork === k
        return (
          <mesh key={`zone-${k}`} rotation={[-Math.PI / 2, 0, 0]} position={[d.center[0], 0.01, d.center[2]]}>
            <ringGeometry args={[active ? 10.2 : 9.5, active ? 11.0 : 10.2, 64]} />
            <meshBasicMaterial color={d.color} transparent opacity={active ? 0.28 : 0.10} />
          </mesh>
        )
      })}

      {/* ── Layer 0: Edges ─────────────────────────────────────────────── */}
      {network.l0Edges.map((edge, i) => {
        const d    = NETWORK_DEFS[edge.network]
        const aN   = network.l0Nodes[edge.a]
        const bN   = network.l0Nodes[edge.b]
        const cong = ((nodeState[edge.a]?.congestion ?? 0) + (nodeState[edge.b]?.congestion ?? 0)) / 2
        const col  = cong > 0.15 ? heatColor(cong) : d.edgeColor
        const act  = selectedNetwork === edge.network
        return (
          <Line key={`l0e-${i}`} points={[aN, bN]}
            color={col} lineWidth={act ? 4.2 : 2.4}
            transparent opacity={act ? 1.0 : 0.68}
          />
        )
      })}

      {/* ── Layer 0: Nodes with 1-qubit orbital rings ──────────────────── */}
      {network.l0Nodes.map((n, idx) => {
        const cong   = nodeState[idx]?.congestion ?? 0
        const isSel  = selectedNode === idx
        const netKey = nodeNetworkKey(idx)
        const d      = NETWORK_DEFS[netKey]
        const actNet = selectedNetwork === netKey
        const col    = cong > 0.15 ? heatColor(cong) : d.color
        const emi    = cong > 0.15 ? heatColor(cong) : d.emissive
        const r      = isSel ? 0.55 : 0.42

        return (
          <group key={`l0n-${idx}`}>
            {/* Node sphere */}
            <mesh
              ref={el => { l0Refs.current[idx] = el }}
              position={n}
              castShadow
              onPointerDown={e => { e.stopPropagation(); onNodeSelect(idx) }}
            >
              <sphereGeometry args={[r, 22, 22]} />
              <meshStandardMaterial
                color={isSel ? '#ffffff' : col}
                emissive={isSel ? '#60e0ff' : emi}
                emissiveIntensity={0.75 + cong * 1.3 + (isSel ? 1.1 : 0) + (actNet ? 0.4 : 0)}
                roughness={0.10} metalness={0.65}
              />
            </mesh>

            {/* 1-qubit orbital ring */}
            <mesh ref={el => { qRefs.current[idx] = el }} position={n}>
              <torusGeometry args={[0.72 + cong * 0.14, 0.035, 8, 38]} />
              <meshStandardMaterial
                color={d.color} emissive={d.emissive}
                emissiveIntensity={0.95 + cong * 0.65}
                transparent opacity={actNet ? 0.96 : isSel ? 0.85 : 0.58}
                roughness={0.08} metalness={0.82}
              />
            </mesh>
          </group>
        )
      })}

      {/* Network floating labels */}
      {NETWORK_KEYS.map(k => {
        const d = NETWORK_DEFS[k]
        return (
          <Html key={`lbl-${k}`} position={d.labelPos} center style={{ pointerEvents: 'none' }}>
            <div style={{
              background: 'rgba(4,9,15,0.90)',
              backdropFilter: 'blur(12px)',
              border: `1.5px solid ${d.color}`,
              borderRadius: 24,
              padding: '5px 16px',
              color: d.color,
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '0.72rem', fontWeight: 700,
              letterSpacing: '0.10em', textTransform: 'uppercase',
              whiteSpace: 'nowrap',
              boxShadow: `0 0 18px ${d.glowColor}, 0 2px 10px rgba(0,0,0,0.6)`,
            }}>
              {d.icon} {d.label} · {d.nodes.length} Qubits
            </div>
          </Html>
        )
      })}

      {/* Hierarchy level labels on the left */}
      {[
        { pos: [-22.5, 0.5,  7], color: '#4a6080', label: 'L0 · MICRO'    },
        { pos: [-22.5, 5.5,  0], color: '#8b5cf6', label: 'L1 · REGIONAL' },
        { pos: [-22.5, 11.0, 0], color: '#ec4899', label: 'L2 · NATIONAL' },
      ].map(({ pos, color, label }) => (
        <Html key={label} position={pos} style={{ pointerEvents: 'none' }}>
          <div style={{ color, fontFamily: 'JetBrains Mono, monospace', fontSize: '0.66rem', fontWeight: 700, letterSpacing: '0.09em', whiteSpace: 'nowrap' }}>
            {label}
          </div>
        </Html>
      ))}

      {/* ── Layer 1: Regional pooled nodes ─────────────────────────────── */}
      {network.l1Edges.map((e, i) => (
        <Line key={`l1e-${i}`}
          points={[network.l1Nodes[e.a], network.l1Nodes[e.b]]}
          color="#8b5cf6" lineWidth={2.4} transparent opacity={0.48}
          dashed dashScale={10} dashSize={1} gapSize={0.5}
        />
      ))}
      {network.l1Assignments.map((a, i) => (
        <Line key={`l1a-${i}`}
          points={[network.l1Nodes[a.parent], network.l0Nodes[a.child]]}
          color="#a78bfa" lineWidth={1.0} transparent opacity={0.22}
        />
      ))}
      {network.l1Nodes.map((n, idx) => (
        <group key={`l1n-${idx}`}>
          <mesh ref={el => { l1Refs.current[idx] = el }} position={n}>
            <octahedronGeometry args={[0.58]} />
            <meshStandardMaterial color="#c4b5fd" emissive="#7c3aed" emissiveIntensity={1.05} wireframe />
          </mesh>
          {/* L1 qubit ring */}
          <mesh ref={el => { l1qRefs.current[idx] = el }} position={n}>
            <torusGeometry args={[0.94, 0.048, 8, 38]} />
            <meshStandardMaterial color="#a78bfa" emissive="#6d28d9" emissiveIntensity={0.75}
              transparent opacity={0.68} roughness={0.08} metalness={0.82} />
          </mesh>
        </group>
      ))}

      {/* ── Layer 2: National pooled nodes ─────────────────────────────── */}
      {network.l2Edges.map((e, i) => (
        <Line key={`l2e-${i}`}
          points={[network.l2Nodes[e.a], network.l2Nodes[e.b]]}
          color="#ec4899" lineWidth={4.2} transparent opacity={0.68}
        />
      ))}
      {network.l2Assignments.map((a, i) => (
        <Line key={`l2a-${i}`}
          points={[network.l2Nodes[a.parent], network.l1Nodes[a.child]]}
          color="#f472b6" lineWidth={1.6} transparent opacity={0.30}
        />
      ))}
      {network.l2Nodes.map((n, idx) => (
        <group key={`l2n-${idx}`}>
          <mesh ref={el => { l2Refs.current[idx] = el }} position={n}>
            <icosahedronGeometry args={[0.88, 0]} />
            <meshStandardMaterial color="#fbcfe8" emissive="#ec4899" emissiveIntensity={1.55} wireframe />
          </mesh>
          {/* L2 qubit ring */}
          <mesh ref={el => { l2qRefs.current[idx] = el }} position={n}>
            <torusGeometry args={[1.18, 0.058, 8, 38]} />
            <meshStandardMaterial color="#f9a8d4" emissive="#be185d" emissiveIntensity={0.95}
              transparent opacity={0.78} roughness={0.08} metalness={0.82} />
          </mesh>
        </group>
      ))}

      <RainField rain={params.rain} wind={params.wind} />
      <SceneController focus={cameraFocus} />
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
//  Slider
// ─────────────────────────────────────────────────────────────────────────────

function Slider({ label, value, min, max, step, onChange, unit, disabled, accentColor }) {
  const accent = accentColor ?? 'rgba(45,125,210,0.25)'
  return (
    <label style={{ display: 'block', opacity: disabled ? 0.48 : 1 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: '0.95rem', fontWeight: 600, color: '#374151', letterSpacing: '0.01em' }}>
          {label}
        </span>
        <span style={{
          fontSize: '0.88rem', fontWeight: 700, fontFamily: 'JetBrains Mono, monospace',
          background: accent, color: '#1d5fa8', border: '1px solid rgba(45,125,210,0.22)',
          borderRadius: 99, padding: '1px 9px',
        }}>
          {value}{unit}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={e => onChange(Number(e.target.value))}
        style={{ height: 5, width: '100%', cursor: disabled ? 'not-allowed' : 'pointer',
          appearance: 'none', borderRadius: 99,
          background: `linear-gradient(90deg, ${accent}, rgba(45,125,210,0.10))`,
        }}
      />
    </label>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
//  Training Sparkline
// ─────────────────────────────────────────────────────────────────────────────

function TrainingSparkline({ history }) {
  if (!history || history.length < 2)
    return <p style={{ marginTop: 8, fontSize: '0.9rem', color: '#9a8c78' }}>No training history.</p>

  const values = history.map(r => Number(r?.val_mse ?? r?.val_loss ?? NaN)).filter(isFinite)
  if (values.length < 2)
    return <p style={{ marginTop: 8, fontSize: '0.9rem', color: '#9a8c78' }}>No validation curve.</p>

  const W = 280, H = 72
  const mn = Math.min(...values), mx = Math.max(...values), span = Math.max(mx - mn, 1e-9)
  const pts = values.map((v, i) =>
    `${((i / (values.length - 1)) * W).toFixed(1)},${(H - ((v - mn) / span) * (H - 4) - 2).toFixed(1)}`
  ).join(' ')

  return (
    <div style={{ marginTop: 10, borderRadius: 10, border: '1px solid rgba(45,125,210,0.18)', background: 'rgba(245,248,255,0.8)', padding: 8 }}>
      <p style={{ marginBottom: 4, fontSize: '0.78rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#2d7dd2' }}>
        Validation MSE Trend
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ height: 68, width: '100%' }}>
        <polyline fill="none" stroke="#2d7dd2" strokeWidth="2.2" points={pts} />
      </svg>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
//  Congestion badge helpers
// ─────────────────────────────────────────────────────────────────────────────

const CONG_STYLE = {
  'Free Flow':  { bg: 'rgba(0,200,140,0.12)',  text: '#007a52', border: 'rgba(0,180,120,0.35)' },
  'Moderate':   { bg: 'rgba(220,160,0,0.12)',  text: '#8a6000', border: 'rgba(200,140,0,0.35)' },
  'Heavy':      { bg: 'rgba(220,80,0,0.12)',   text: '#aa3800', border: 'rgba(220,80,0,0.35)'  },
  'Standstill': { bg: 'rgba(200,0,40,0.12)',   text: '#880020', border: 'rgba(200,0,40,0.35)'  },
}

function CongestionBadge({ level }) {
  const s = CONG_STYLE[level] ?? CONG_STYLE['Moderate']
  return (
    <span style={{ background: s.bg, color: s.text, border: `1px solid ${s.border}`,
      borderRadius: 9999, padding: '2px 9px',
      fontSize: '0.80rem', fontWeight: 700, letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>
      {level}
    </span>
  )
}

function MiniUtilBar({ flow, capacity }) {
  const util = Math.min(100, Math.round((flow / (capacity || 1)) * 100))
  const col = util > 74 ? '#dc2626' : util > 54 ? '#ea580c' : util > 29 ? '#ca8a04' : '#16a34a'
  return (
    <div style={{ marginTop: 3, width: '100%', height: 3, background: 'rgba(0,0,0,0.08)', borderRadius: 4, overflow: 'hidden' }}>
      <div style={{ width: `${util}%`, height: '100%', background: col, transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)' }} />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
//  Node Forecast Table
//  Fetches from the STQGCN inference API. Falls back to mock data when unavailable.
// ─────────────────────────────────────────────────────────────────────────────

function NodeForecastTable({ selectedNode, onNodeSelect, nodeOverrides, onForecastData, params, isSimulating, simStep }) {
  const [forecast, setForecast]   = useState(null)
  const [loading, setLoading]     = useState(true)
  const [inferring, setInferring] = useState(false)
  const [isMock, setIsMock]       = useState(false)
  const [lastTs, setLastTs]       = useState('')
  const rowRefs   = useRef({})
  const debounce  = useRef(null)

  const fetchForecast = async (overridesPayload, globalParams, stepOffset = 0) => {
    setInferring(true)
    try {
      const apiOverrides = {}
      Object.entries(overridesPayload).forEach(([idx, vals]) => {
        const e = {}
        if (vals.trafficFlow !== undefined) e.traffic_flow = vals.trafficFlow
        if (vals.avgSpeed    !== undefined) e.avg_speed    = vals.avgSpeed
        if (Object.keys(e).length > 0) apiOverrides[String(idx)] = e
      })
      const apiGlobal = { rain: globalParams.rain, temp: globalParams.temperature, hour: globalParams.timeOfDay, wind: globalParams.wind }

      const resp = await fetch(apiUrl('/api/nodes/forecast'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ overrides: apiOverrides, global_params: apiGlobal, step_offset: stepOffset, n_steps: 3 }),
      })
      if (!resp.ok) throw new Error(`API ${resp.status}`)
      const data = await resp.json()
      const nodes = data.nodes ?? []
      if (nodes.length === 0) throw new Error('Empty response')

      // Ensure predictions exist on every node (guard against malformed data)
      const safe = nodes.map(n => ({
        ...n,
        predictions: Array.isArray(n.predictions) && n.predictions.length >= 3
          ? n.predictions
          : [n.flow_t * 1.02, n.flow_t * 1.05, n.flow_t * 1.08],
      }))

      setForecast(safe)
      setIsMock(false)
      setLastTs(data.last_updated ?? '')
      if (onForecastData) onForecastData(safe)
    } catch (_) {
      // Fall back to deterministic mock data so predictions are always visible
      const mock = generateMockForecastNodes()
      setForecast(mock)
      setIsMock(true)
      setLastTs(new Date().toISOString())
      if (onForecastData) onForecastData(mock)
    } finally {
      setLoading(false)
      setInferring(false)
    }
  }

  // Initial load
  useEffect(() => { fetchForecast({}, params, 0) }, []) // eslint-disable-line

  // Re-inference on override/param/sim changes (debounced)
  useEffect(() => {
    if (debounce.current) clearTimeout(debounce.current)
    debounce.current = setTimeout(() => fetchForecast(nodeOverrides, params, simStep), 350)
    return () => { if (debounce.current) clearTimeout(debounce.current) }
  }, [nodeOverrides, params, simStep]) // eslint-disable-line

  // Scroll selected row into view
  useEffect(() => {
    const key = `N${String(selectedNode + 1).padStart(3, '0')}`
    rowRefs.current[key]?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [selectedNode])

  const selectedNodeId = `N${String(selectedNode + 1).padStart(3, '0')}`

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Table header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '9px 16px 8px',
        borderBottom: '1px solid rgba(160,120,60,0.15)',
        flexShrink: 0, background: 'rgba(245,242,235,0.85)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: isMock ? '#ca8a04' : '#16a34a',
            boxShadow: `0 0 8px ${isMock ? '#ca8a04' : '#16a34a'}`,
            display: 'inline-block', animation: 'pulse 2s infinite',
          }} />
          <p style={{ fontSize: '0.84rem', fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', color: '#1d5fa8', margin: 0, fontFamily: 'JetBrains Mono, monospace' }}>
            5-Second Ahead Forecast · Hierarchical ST-QGCN
          </p>
          {isMock && (
            <span style={{ background: 'rgba(202,138,4,0.15)', color: '#92400e', border: '1px solid rgba(202,138,4,0.35)', borderRadius: 9999, padding: '1px 8px', fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.06em' }}>
              DEMO
            </span>
          )}
        </div>
        <p style={{ fontSize: '0.80rem', color: inferring ? '#ca8a04' : '#9a8c78', margin: 0 }}>
          {loading ? 'Running model…' : inferring ? '⟳ Re-running inference…' : isMock ? 'Demo mode — API offline' : `Updated ${lastTs.slice(11, 16)} UTC`}
        </p>
      </div>

      {/* Table */}
      <div style={{ overflowY: 'auto', flex: 1, padding: '0 4px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.92rem' }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 2 }}>
            <tr style={{ background: '#f0ead8' }}>
              {['Node', 'Network', 'Current (t)', '+5s', '+10s', '+15s'].map(h => (
                <th key={h} style={{
                  padding: '9px 12px', textAlign: 'left', fontSize: '0.80rem', fontWeight: 700,
                  letterSpacing: '0.07em', textTransform: 'uppercase', color: '#7c5c2e',
                  borderBottom: '2px solid rgba(160,120,60,0.2)', whiteSpace: 'nowrap',
                  fontFamily: 'JetBrains Mono, monospace',
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(forecast ?? []).slice(0, TOTAL_L0).map((node, idx) => {
              const isSel = node.node_id === selectedNodeId
              const netKey = nodeNetworkKey(node.node_index ?? idx)
              const netDef = netKey ? NETWORK_DEFS[netKey] : null
              return (
                <tr
                  key={node.node_id}
                  ref={el => { rowRefs.current[node.node_id] = el }}
                  onClick={() => onNodeSelect(node.node_index ?? idx)}
                  style={{
                    cursor: 'pointer',
                    background: isSel
                      ? 'rgba(45,125,210,0.08)'
                      : idx % 2 === 0 ? 'rgba(250,246,238,0.6)' : 'rgba(245,240,230,0.4)',
                    boxShadow: isSel ? 'inset 0 0 0 1.5px rgba(45,125,210,0.5)' : 'none',
                    transition: 'background 0.2s',
                  }}
                >
                  <td style={{ padding: '7px 12px', fontFamily: 'JetBrains Mono, monospace', fontWeight: isSel ? 800 : 500, color: isSel ? '#1d5fa8' : '#374151', borderBottom: '1px solid rgba(160,120,60,0.07)', fontSize: '0.92rem' }}>
                    {isSel && <span style={{ color: '#2d7dd2', marginRight: 4 }}>▶</span>}
                    {node.node_id}
                  </td>
                  <td style={{ padding: '7px 12px', borderBottom: '1px solid rgba(160,120,60,0.07)' }}>
                    {netDef ? (
                      <span style={{ color: netDef.color, fontWeight: 700, fontSize: '0.80rem', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.05em' }}>
                        {netDef.icon} {netDef.shortLabel}
                      </span>
                    ) : (
                      <span style={{ color: '#6b7280', fontSize: '0.88rem' }}>{node.zone}</span>
                    )}
                  </td>
                  <td style={{ padding: '7px 12px', borderBottom: '1px solid rgba(160,120,60,0.07)', minWidth: 80 }}>
                    <div style={{ color: '#1f2937', fontWeight: 600, fontSize: '0.94rem' }}>{Math.round(node.flow_t).toLocaleString()}</div>
                    <MiniUtilBar flow={node.flow_t} capacity={node.capacity_veh_per_hr} />
                  </td>
                  <td style={{ padding: '7px 12px', borderBottom: '1px solid rgba(160,120,60,0.07)', minWidth: 80 }}>
                    <div style={{ color: '#374151', fontSize: '0.92rem' }}>{Math.round(node.predictions?.[0] ?? node.flow_t * 1.02).toLocaleString()}</div>
                    <MiniUtilBar flow={node.predictions?.[0] ?? node.flow_t * 1.02} capacity={node.capacity_veh_per_hr} />
                  </td>
                  <td style={{ padding: '7px 12px', borderBottom: '1px solid rgba(160,120,60,0.07)', minWidth: 80 }}>
                    <div style={{ color: '#4b5563', fontSize: '0.92rem' }}>{Math.round(node.predictions?.[1] ?? node.flow_t * 1.05).toLocaleString()}</div>
                    <MiniUtilBar flow={node.predictions?.[1] ?? node.flow_t * 1.05} capacity={node.capacity_veh_per_hr} />
                  </td>
                  <td style={{ padding: '7px 12px', borderBottom: '1px solid rgba(160,120,60,0.07)', minWidth: 80 }}>
                    <div style={{ color: '#6b7280', fontSize: '0.92rem' }}>{Math.round(node.predictions?.[2] ?? node.flow_t * 1.09).toLocaleString()}</div>
                    <MiniUtilBar flow={node.predictions?.[2] ?? node.flow_t * 1.09} capacity={node.capacity_veh_per_hr} />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {loading && (
          <p style={{ textAlign: 'center', color: '#9a8c78', fontSize: '0.92rem', padding: '14px 0' }}>
            Running ST-QGCN inference…
          </p>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
//  Reports View
// ─────────────────────────────────────────────────────────────────────────────

function ReportsView({ activeRun, runs, history, onBack }) {
  const [plots, setPlots]   = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState('')

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        const resp = await fetch(apiUrl(`/api/runs/${activeRun}/plots`))
        if (!resp.ok) throw new Error('Failed to fetch plots')
        const data = await resp.json()
        setPlots(data.plots || [])
      } catch (e) { setError(e.message) } finally { setLoading(false) }
    }
    if (activeRun) load()
  }, [activeRun])

  const trainingPlots = plots.filter(p => p.group === 'training' || !p.group)
  const blochPlots    = plots.filter(p => p.group === 'bloch_sphere').sort((a, b) => (a.qubit_idx ?? 99) - (b.qubit_idx ?? 99))
  const blochVideos   = plots.filter(p => p.group === 'bloch_video')
  const lossCurvePlot = trainingPlots.find(p => /training|loss|curve/i.test(p.name))
  const otherPlots    = trainingPlots.filter(p => p !== lossCurvePlot)

  const PlotCard = ({ plot }) => (
    <div style={{ display: 'flex', flexDirection: 'column', borderRadius: 14, border: '1px solid rgba(160,120,60,0.2)', background: 'rgba(250,246,238,0.6)', padding: 14, overflow: 'hidden' }}>
      <h3 style={{ marginBottom: 10, fontSize: '0.98rem', fontWeight: 600, color: '#374151', textTransform: 'capitalize', fontFamily: 'Inter, sans-serif' }}>
        {plot.name.replace(/_/g, ' ').replace(/\.[^/.]+$/, '')}
      </h3>
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 10, background: '#f5f0e8', padding: 6 }}>
        {plot.type === 'video' ? (
          <video src={apiUrl(plot.url)} controls autoPlay loop muted style={{ maxHeight: 360, width: '100%', objectFit: 'contain', borderRadius: 8 }} />
        ) : (
          <img src={apiUrl(plot.url)} alt={plot.name} style={{ maxHeight: 360, width: '100%', objectFit: 'contain' }} />
        )}
      </div>
    </div>
  )

  return (
    <div style={{ margin: '0 auto', display: 'flex', flexDirection: 'column', height: '100%', width: '100%', maxWidth: 1400, padding: '24px 32px', overflowY: 'auto' }}>
      <div style={{ marginBottom: 24, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.86rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: '#1d5fa8', margin: 0 }}>Run Reports</p>
          <h2 style={{ marginTop: 4, fontFamily: 'Merriweather Sans, sans-serif', fontSize: '2.1rem', fontWeight: 700, color: '#1a2e4a' }}>Simulation Graphs</h2>
        </div>
        <button onClick={onBack} style={{ borderRadius: 12, border: '1px solid rgba(45,125,210,0.3)', background: 'rgba(45,125,210,0.08)', padding: '10px 20px', fontWeight: 600, color: '#1d5fa8', cursor: 'pointer', fontSize: '1rem', fontFamily: 'Inter, sans-serif' }}>
          ← Back to Simulation
        </button>
      </div>

      {loading ? <p style={{ color: '#9a8c78' }}>Loading graphs…</p>
        : error ? <p style={{ color: '#cc2200' }}>{error}</p>
        : plots.length === 0 ? <p style={{ color: '#9a8c78' }}>No graphs found for this run.</p>
        : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 36 }}>
            {(lossCurvePlot || blochPlots.length > 0) && (
              <section>
                <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.80rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#7c5c2e', marginBottom: 14 }}>
                  Training Loss {blochPlots.length > 0 ? '· Qubit State Evolution' : ''}
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: blochPlots.length > 0 ? '1fr 1fr' : '1fr', gap: 24 }}>
                  {lossCurvePlot && (
                    <div style={{ display: 'flex', flexDirection: 'column', borderRadius: 16, border: '1px solid rgba(45,125,210,0.2)', background: 'rgba(245,248,255,0.7)', padding: 18, minHeight: 400 }}>
                      <h3 style={{ marginBottom: 12, fontSize: '1.08rem', fontWeight: 700, color: '#1a2e4a', fontFamily: 'Merriweather Sans, sans-serif' }}>Loss Curve (Training &amp; Validation)</h3>
                      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 12, background: '#f5f0e8', padding: 8 }}>
                        <img src={apiUrl(lossCurvePlot.url)} alt={lossCurvePlot.name} style={{ width: '100%', maxHeight: 360, objectFit: 'contain' }} />
                      </div>
                    </div>
                  )}
                  {blochPlots.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', borderRadius: 16, border: '2px solid rgba(139,92,246,0.35)', background: 'rgba(245,240,255,0.7)', padding: 18 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                        <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: '#8b5cf6', boxShadow: '0 0 8px #8b5cf6', animation: 'pulse 2s infinite' }} />
                        <h3 style={{ margin: 0, fontSize: '1.08rem', fontWeight: 700, color: '#3b0764', fontFamily: 'Merriweather Sans, sans-serif' }}>
                          Bloch Sphere — Layer 1 Qubits (Regional)
                        </h3>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, flex: 1 }}>
                        {blochPlots.map(plot => (
                          <div key={plot.name} style={{ borderRadius: 12, overflow: 'hidden', border: '1px solid rgba(139,92,246,0.25)', background: 'rgba(255,255,255,0.6)', display: 'flex', flexDirection: 'column' }}>
                            <p style={{ margin: 0, padding: '5px 10px', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.78rem', fontWeight: 700, color: '#5b21b6', background: 'rgba(139,92,246,0.08)', borderBottom: '1px solid rgba(139,92,246,0.15)' }}>
                              Node {plot.qubit_idx ?? '?'} · Qubit {plot.qubit_idx ?? '?'}
                            </p>
                            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 4 }}>
                              <img src={apiUrl(plot.url)} alt={plot.name} style={{ width: '100%', maxHeight: 175, objectFit: 'contain' }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </section>
            )}

            {blochVideos.length > 0 && (
              <section>
                <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.80rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#5b21b6', marginBottom: 14 }}>
                  Qubit State Animation — Bloch Sphere Evolution
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 24 }}>
                  {blochVideos.map(plot => (
                    <div key={plot.name} style={{ borderRadius: 16, border: '2px solid rgba(139,92,246,0.4)', background: 'rgba(245,240,255,0.8)', padding: 16, boxShadow: '0 0 24px rgba(139,92,246,0.12)' }}>
                      <h3 style={{ marginBottom: 10, fontSize: '0.98rem', fontWeight: 700, color: '#3b0764', fontFamily: 'Merriweather Sans, sans-serif' }}>
                        {plot.name.replace(/_/g, ' ').replace(/\.[^/.]+$/, '')}
                      </h3>
                      <video src={apiUrl(plot.url)} controls autoPlay loop muted style={{ width: '100%', maxHeight: 380, objectFit: 'contain', borderRadius: 10, background: '#0a0a14' }} />
                    </div>
                  ))}
                </div>
              </section>
            )}

            {otherPlots.length > 0 && (
              <section>
                <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.80rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#7c5c2e', marginBottom: 14 }}>
                  Additional Training Analysis
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 28 }}>
                  {otherPlots.map(plot => <PlotCard key={plot.name} plot={plot} />)}
                </div>
              </section>
            )}
          </div>
        )
      }
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section Label
// ─────────────────────────────────────────────────────────────────────────────

function SectionLabel({ children, color }) {
  return (
    <p style={{
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: '0.78rem', fontWeight: 700,
      textTransform: 'uppercase', letterSpacing: '0.12em',
      color: color ?? '#7c5c2e', margin: '0 0 6px 0',
    }}>
      {children}
    </p>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
//  App — Main Component
// ─────────────────────────────────────────────────────────────────────────────

function App() {
  const [params, setParams]   = useState(DEFAULTS)
  const [runs, setRuns]       = useState([])
  const [activeRun, setActiveRun] = useState('')
  const [metrics, setMetrics] = useState(null)
  const [history, setHistory] = useState([])
  const [apiError, setApiError] = useState('')
  const [isLoadingApi, setIsLoadingApi] = useState(true)

  const [selectedNode, setSelectedNode] = useState(0)
  const [nodeOverrides, setNodeOverrides] = useState({})
  const [currentView, setCurrentView] = useState('simulation')

  // ── Network-level state ──────────────────────────────────────────────────
  const [selectedNetwork, setSelectedNetwork] = useState(null)   // 'tree' | 'line' | 'ring' | null
  const [networkCongestion, setNetworkCongestion] = useState({ tree: 0, line: 0, ring: 0 })
  const [cameraFocus, setCameraFocus]   = useState(null)  // { pos, look }

  // ── Simulation ───────────────────────────────────────────────────────────
  const [isSimulating, setIsSimulating] = useState(false)
  const [simStep, setSimStep]     = useState(0)

  // ── Forecast data ────────────────────────────────────────────────────────
  const [forecastData, setForecastData] = useState([])

  // Ticker
  useEffect(() => {
    if (!isSimulating) return
    const iv = setInterval(() => setSimStep(s => s + 1), LIVE_REFRESH_MS)
    return () => clearInterval(iv)
  }, [isSimulating])

  // Build network once
  const network = useMemo(() => buildHierarchicalNetwork(), [])

  // ── Derive node state from forecast + network congestion overrides ────────
  const baseNodeState = useMemo(() => {
    if (forecastData.length === 0)
      return Array.from({ length: TOTAL_L0 }, () => ({ flow: 0, congestion: 0 }))
    const byIdx = {}
    forecastData.forEach(r => { byIdx[r.node_index] = r })
    return Array.from({ length: TOTAL_L0 }, (_, i) => {
      const r = byIdx[i]
      if (!r) return { flow: 0, congestion: 0 }
      const ratio = r.capacity_veh_per_hr > 0 ? clamp(r.flow_t / r.capacity_veh_per_hr, 0, 1) : 0
      return { flow: r.flow_t, congestion: ratio }
    })
  }, [forecastData])

  const nodeState = useMemo(() => {
    return baseNodeState.map((s, idx) => {
      const k = nodeNetworkKey(idx)
      if (!k) return s
      const nc = networkCongestion[k]
      return nc > s.congestion ? { ...s, congestion: nc } : s
    })
  }, [baseNodeState, networkCongestion])

  const networkLoad = useMemo(() =>
    nodeState.length === 0 ? 0 : nodeState.reduce((sum, n) => sum + n.congestion, 0) / nodeState.length
  , [nodeState])

  // ── API bootstrap ────────────────────────────────────────────────────────
  useEffect(() => {
    const load = async () => {
      try {
        setApiError('')
        setIsLoadingApi(true)
        const resp = await fetch(apiUrl('/api/runs'))
        if (!resp.ok) throw new Error(`Runs API failed (${resp.status})`)
        const data = await resp.json()
        const names = (data.runs ?? []).map(r => r.name)
        setRuns(names)
        if (data.default_run) setActiveRun(data.default_run)
        else if (names.length > 0) setActiveRun(names[0])
        else setApiError('No runs found.')
      } catch (e) {
        setApiError(e.message || 'Failed to connect to backend API.')
      } finally { setIsLoadingApi(false) }
    }
    load()
  }, [])

  useEffect(() => {
    if (!activeRun) return
    const load = async () => {
      try {
        setApiError('')
        const [mr, hr] = await Promise.all([
          fetch(apiUrl(`/api/runs/${activeRun}/metrics`)),
          fetch(apiUrl(`/api/runs/${activeRun}/history`)),
        ])
        if (!mr.ok) throw new Error(`Metrics API failed (${mr.status})`)
        if (!hr.ok) throw new Error(`History API failed (${hr.status})`)
        const md = await mr.json(), hd = await hr.json()
        setMetrics(md.metrics ?? null)
        setHistory(hd.history ?? [])
      } catch (e) { setApiError(e.message || 'Failed to load run data.') }
    }
    load()
    const iv = setInterval(load, LIVE_REFRESH_MS)
    return () => clearInterval(iv)
  }, [activeRun])

  // ── Handlers ─────────────────────────────────────────────────────────────
  const setParam = key => val => setParams(prev => ({ ...prev, [key]: val }))

  const handleEnhanceNetwork = key => {
    const def = NETWORK_DEFS[key]
    setCameraFocus({ pos: def.cameraPos, look: def.cameraLook })
    setSelectedNetwork(key)
  }

  const handleSelectNetwork = key => {
    setSelectedNetwork(prev => prev === key ? null : key)
    if (selectedNetwork === key) setCameraFocus(null)
  }

  const handleNodeSelect = idx => {
    setSelectedNode(idx)
    const k = nodeNetworkKey(idx)
    if (k) setSelectedNetwork(k)
  }

  const setSelectedNodeParam = (key, val) => {
    if (selectedNode === null) return
    setNodeOverrides(prev => {
      const fr = forecastData.find(r => r.node_index === selectedNode)
      const cur = prev[selectedNode] ?? { trafficFlow: fr ? Math.round(fr.flow_t) : 500, avgSpeed: 42 }
      return { ...prev, [selectedNode]: { ...cur, [key]: val } }
    })
  }

  // ── Derived display values ────────────────────────────────────────────────
  const selectedFR = forecastData.find(r => r.node_index === selectedNode)
  const selectedState = nodeOverrides[selectedNode] ?? {
    trafficFlow: selectedFR ? Math.round(selectedFR.flow_t) : 500,
    avgSpeed: 42,
  }
  const selectedNetKey = nodeNetworkKey(selectedNode)
  const selectedNetDef = selectedNetKey ? NETWORK_DEFS[selectedNetKey] : null
  const selectedNodeName = `JUNC-${String(selectedNode + 1).padStart(3, '0')}`

  const bestVal   = Number(metrics?.best_val_mse)
  const testMse   = Number(metrics?.test_mse)
  const testMae   = Number(metrics?.test_mae)
  const bestEpoch = Number(metrics?.best_epoch)

  const resetAll = () => {
    setParams(DEFAULTS)
    setNodeOverrides({})
    setNetworkCongestion({ tree: 0, line: 0, ring: 0 })
    setIsSimulating(false)
    setSimStep(0)
    setCameraFocus(null)
    setSelectedNetwork(null)
  }

  const btnStyle = (variant = 'primary') => {
    const base = { width: '100%', borderRadius: 10, padding: '11px 14px', fontWeight: 600, fontSize: '0.98rem', cursor: 'pointer', border: '1.5px solid', transition: 'background 0.2s, box-shadow 0.2s', fontFamily: 'Inter, sans-serif' }
    if (variant === 'danger')    return { ...base, background: 'rgba(220,30,30,0.08)',   borderColor: 'rgba(220,30,30,0.3)',   color: '#991b1b' }
    if (variant === 'success')   return { ...base, background: 'rgba(22,163,74,0.08)',   borderColor: 'rgba(22,163,74,0.3)',   color: '#166534' }
    if (variant === 'amber')     return { ...base, background: 'rgba(202,138,4,0.08)',   borderColor: 'rgba(202,138,4,0.3)',   color: '#854d0e' }
    if (variant === 'purple')    return { ...base, background: 'rgba(139,92,246,0.10)',  borderColor: 'rgba(139,92,246,0.35)', color: '#5b21b6' }
    return { ...base, background: 'rgba(45,125,210,0.08)', borderColor: 'rgba(45,125,210,0.28)', color: '#1d5fa8' }
  }

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="bg-research-paper" style={{ height: '100vh', overflow: 'hidden', color: '#1a2e4a' }}>
      {currentView === 'reports' ? (
        <ReportsView
          activeRun={activeRun} runs={runs} history={history}
          onBack={() => setCurrentView('simulation')}
        />
      ) : (
        <>
          {/* ── Header ─────────────────────────────────────────────────── */}
          <motion.header
            initial={{ y: -28, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.55, ease: 'easeOut' }}
            style={{
              margin: '0 auto', display: 'flex', width: '100%', maxWidth: 1440,
              alignItems: 'center', justifyContent: 'space-between',
              padding: '16px 32px 12px',
              borderBottom: '1px solid rgba(160,120,60,0.2)',
            }}
          >
            <div>
              <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.84rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.16em', color: '#1d5fa8', margin: 0 }}>
                Hierarchical ST-QGCN · 3-Layer Quantum Graph Network
              </p>
              <h1 style={{ marginTop: 4, fontFamily: 'Merriweather Sans, sans-serif', fontSize: '2.0rem', fontWeight: 800, color: '#1a2e4a', lineHeight: 1.2 }}>
                Chembur Network Command Deck
              </h1>
            </div>

            {/* Stats pills */}
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              {[
                { label: 'L0 Nodes',  value: TOTAL_L0,                     color: '#1d5fa8' },
                { label: 'L1 Nodes',  value: network.l1Nodes.length,       color: '#7c3aed' },
                { label: 'L2 Nodes',  value: network.l2Nodes.length,       color: '#db2777' },
                { label: 'Qubits',    value: network.totalQubits,          color: '#059669' },
                { label: 'Networks',  value: 3,                            color: '#d97706' },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ background: 'rgba(45,125,210,0.07)', border: '1px solid rgba(45,125,210,0.20)', borderRadius: 99, padding: '3px 12px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '1.15rem', fontWeight: 700, color, lineHeight: 1.2 }}>{value}</span>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#7c5c2e' }}>{label}</span>
                </div>
              ))}
            </div>
          </motion.header>

          {/* ── Main layout ─────────────────────────────────────────────── */}
          <main style={{
            margin: '0 auto', display: 'grid', height: 'calc(100vh - 94px)',
            width: '100%', maxWidth: 1440,
            gap: 18, padding: '14px 32px 18px',
            gridTemplateColumns: '360px 1fr',
          }}>
            {/* ── Sidebar ─────────────────────────────────────────────── */}
            <motion.aside
              initial={{ x: -24, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="glass-panel"
              style={{ height: '100%', overflowY: 'auto', borderRadius: 20, padding: '18px 18px 18px 18px' }}
            >
              <h2 style={{ fontFamily: 'Merriweather Sans, sans-serif', fontSize: '1.28rem', fontWeight: 700, color: '#1a2e4a', margin: '0 0 14px 0' }}>
                Simulation Controls
              </h2>

              {/* Sim start/stop */}
              <button type="button" onClick={() => setIsSimulating(s => !s)} style={isSimulating ? btnStyle('danger') : btnStyle('success')}>
                {isSimulating ? `⏹ Stop Simulation (Step ${simStep})` : '▶ Start Real-Time Simulation'}
              </button>

              {/* ── Layer 0 Network Selector ─────────────────────────── */}
              <div style={{ marginTop: 18, borderTop: '1px solid rgba(160,120,60,0.15)', paddingTop: 14 }}>
                <SectionLabel>Layer 0 Networks · 1 Qubit / Node</SectionLabel>

                {NETWORK_KEYS.map(key => {
                  const def     = NETWORK_DEFS[key]
                  const isActive = selectedNetwork === key
                  const netCong  = networkCongestion[key]

                  return (
                    <div
                      key={key}
                      style={{
                        marginTop: 8, borderRadius: 12, overflow: 'hidden',
                        border: `1.5px solid ${isActive ? def.borderColor : 'rgba(160,120,60,0.15)'}`,
                        background: isActive ? def.bgColor : 'rgba(250,246,238,0.5)',
                        transition: 'border-color 0.2s, background 0.2s',
                      }}
                    >
                      {/* Network header row */}
                      <div style={{ display: 'flex', alignItems: 'center', padding: '10px 12px', gap: 8 }}>
                        {/* Select toggle */}
                        <button
                          type="button"
                          onClick={() => handleSelectNetwork(key)}
                          style={{
                            flex: 1, display: 'flex', alignItems: 'center', gap: 8,
                            background: 'none', border: 'none', cursor: 'pointer',
                            textAlign: 'left', padding: 0,
                          }}
                        >
                          <span style={{ fontSize: '1.1rem' }}>{def.icon}</span>
                          <div>
                            <div style={{ fontWeight: 700, fontSize: '0.94rem', color: isActive ? def.textColor : '#1a2e4a', fontFamily: 'Inter, sans-serif' }}>
                              {def.label}
                            </div>
                            <div style={{ fontSize: '0.72rem', color: '#6b7280', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.05em' }}>
                              {def.nodes.length} nodes · {def.description}
                            </div>
                          </div>
                        </button>

                        {/* Enhance button */}
                        <button
                          type="button"
                          onClick={() => handleEnhanceNetwork(key)}
                          title={`Zoom to ${def.label}`}
                          style={{
                            flexShrink: 0, padding: '5px 10px', borderRadius: 8,
                            border: `1px solid ${def.borderColor}`, background: def.bgColor,
                            color: def.textColor, fontWeight: 700, fontSize: '0.75rem',
                            cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace',
                            letterSpacing: '0.05em', transition: 'opacity 0.2s',
                          }}
                        >
                          🔍 Enhance
                        </button>
                      </div>

                      {/* Congestion slider — only shown when network is selected */}
                      {isActive && (
                        <div style={{ padding: '2px 12px 12px 12px', borderTop: `1px solid ${def.borderColor}` }}>
                          <div style={{ marginTop: 10 }}>
                            <Slider
                              label="Congestion Override"
                              value={Math.round(netCong * 100)}
                              min={0} max={100} step={1} unit="%"
                              accentColor={def.bgColor}
                              onChange={v => setNetworkCongestion(prev => ({ ...prev, [key]: v / 100 }))}
                              disabled={isSimulating}
                            />
                          </div>
                          <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
                            {[0, 25, 50, 75, 100].map(pct => (
                              <button
                                key={pct}
                                type="button"
                                onClick={() => setNetworkCongestion(prev => ({ ...prev, [key]: pct / 100 }))}
                                disabled={isSimulating}
                                style={{
                                  padding: '2px 10px', borderRadius: 99, fontSize: '0.72rem', fontWeight: 700,
                                  border: `1px solid ${def.borderColor}`, background: Math.round(netCong * 100) === pct ? def.bgColor : 'transparent',
                                  color: def.textColor, cursor: isSimulating ? 'not-allowed' : 'pointer',
                                  fontFamily: 'JetBrains Mono, monospace',
                                }}
                              >
                                {pct}%
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>

              {/* ── Selected Node Override ─────────────────────────────── */}
              <div style={{ marginTop: 18, borderRadius: 12, border: '1px solid rgba(202,138,4,0.28)', background: 'rgba(254,249,238,0.7)', padding: 14 }}>
                <SectionLabel>Selected Node — Override</SectionLabel>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <p style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, color: '#1a2e4a', margin: 0, fontSize: '1.08rem' }}>
                    {selectedNodeName}
                  </p>
                  {selectedNetDef && (
                    <span style={{ color: selectedNetDef.color, fontWeight: 700, fontSize: '0.78rem', fontFamily: 'JetBrains Mono, monospace', background: selectedNetDef.bgColor, border: `1px solid ${selectedNetDef.borderColor}`, borderRadius: 99, padding: '1px 8px' }}>
                      {selectedNetDef.icon} {selectedNetDef.shortLabel}
                    </span>
                  )}
                </div>
                <p style={{ fontSize: '0.88rem', color: '#7c5c2e', margin: '0 0 10px 0' }}>Click a node in the 3-D scene to select.</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <Slider label="Traffic Flow" value={selectedState.trafficFlow} min={50} max={2500} step={10} unit=" veh/hr"
                    onChange={v => setSelectedNodeParam('trafficFlow', v)} disabled={isSimulating} />
                  <Slider label="Avg Speed" value={selectedState.avgSpeed} min={5} max={100} step={1} unit=" km/h"
                    onChange={v => setSelectedNodeParam('avgSpeed', v)} disabled={isSimulating} />
                </div>
              </div>

              {/* ── Environmental Parameters ────────────────────────────── */}
              <div style={{ marginTop: 18, borderTop: '1px solid rgba(160,120,60,0.15)', paddingTop: 14 }}>
                <SectionLabel>Environmental Parameters</SectionLabel>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 8 }}>
                  <Slider label="Rain Intensity" value={params.rain}        min={0}  max={100} step={1} unit="%" onChange={setParam('rain')} />
                  <Slider label="Wind Speed"     value={params.wind}        min={0}  max={60}  step={1} unit=" km/h" onChange={setParam('wind')} />
                  <Slider label="Time of Day"    value={params.timeOfDay}   min={0}  max={23}  step={1} unit=":00" onChange={setParam('timeOfDay')} />
                  <Slider label="Temperature"    value={params.temperature} min={-5} max={45}  step={1} unit=" °C" onChange={setParam('temperature')} />
                </div>
              </div>

              {/* ── Model Metrics ──────────────────────────────────────── */}
              {metrics && (
                <div style={{ marginTop: 16, borderRadius: 12, border: '1px solid rgba(45,125,210,0.2)', background: 'rgba(245,248,255,0.7)', padding: 14 }}>
                  <SectionLabel color="#1d5fa8">Model Performance</SectionLabel>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
                    {[
                      { k: 'Test MSE',   v: isFinite(testMse)   ? testMse.toFixed(4)   : '—' },
                      { k: 'Test MAE',   v: isFinite(testMae)   ? testMae.toFixed(4)   : '—' },
                      { k: 'Best Val',   v: isFinite(bestVal)   ? bestVal.toFixed(4)   : '—' },
                      { k: 'Best Epoch', v: isFinite(bestEpoch) ? bestEpoch            : '—' },
                    ].map(({ k, v }) => (
                      <div key={k} style={{ background: 'rgba(255,255,255,0.6)', borderRadius: 8, padding: '7px 10px', border: '1px solid rgba(45,125,210,0.12)' }}>
                        <p style={{ fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#7c5c2e', margin: 0 }}>{k}</p>
                        <p style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, color: '#1d5fa8', margin: 0, fontSize: '1.05rem' }}>{v}</p>
                      </div>
                    ))}
                  </div>
                  <TrainingSparkline history={history} />
                </div>
              )}

              {/* ── Action buttons ─────────────────────────────────────── */}
              <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <button type="button" onClick={resetAll} style={btnStyle('amber')}>
                  ↺ Reset Scenario
                </button>
                <button type="button" onClick={() => setCurrentView('reports')} style={btnStyle('purple')}>
                  📊 View Generated Graphs
                </button>
              </div>

              {apiError && (
                <p style={{ marginTop: 12, fontSize: '0.86rem', color: '#991b1b', background: 'rgba(220,30,30,0.06)', borderRadius: 8, padding: '8px 12px', border: '1px solid rgba(220,30,30,0.18)' }}>
                  ⚠ {apiError}
                </p>
              )}
            </motion.aside>

            {/* ── 3-D Scene + Forecast Table ──────────────────────────── */}
            <motion.section
              initial={{ y: 24, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.7, delay: 0.15 }}
              className="glass-panel"
              style={{ position: 'relative', height: '100%', overflow: 'hidden', borderRadius: 20, display: 'flex', flexDirection: 'column' }}
            >
              {/* 3-D Canvas */}
              <div style={{ position: 'relative', flex: '0 0 58%', overflow: 'hidden' }}>
                <div style={{ height: '100%', minHeight: 300, width: '100%' }}>
                  <Canvas shadows gl={{ antialias: true }} dpr={[1, 1.8]}>
                    <color attach="background" args={['#ffffff']} />
                    <PerspectiveCamera makeDefault position={[2, 16, 26]} fov={52} />
                    <NetworkScene
                      params={params}
                      selectedNode={selectedNode}
                      onNodeSelect={handleNodeSelect}
                      selectedNetwork={selectedNetwork}
                      network={network}
                      nodeState={nodeState}
                      networkLoad={networkLoad}
                      cameraFocus={cameraFocus}
                    />
                  </Canvas>
                </div>

                {/* Hierarchy legend overlay */}
                <div style={{
                  position: 'absolute', top: 12, left: 14,
                  display: 'flex', flexDirection: 'column', gap: 6,
                  pointerEvents: 'none',
                }}>
                  {[
                    { label: 'L2 · National',  color: '#ec4899', dot: '#fbcfe8', desc: `${network.l2Nodes.length} nodes` },
                    { label: 'L1 · Regional',  color: '#8b5cf6', dot: '#c4b5fd', desc: `${network.l1Nodes.length} nodes` },
                    { label: 'L0 · Micro',     color: '#60a5fa', dot: '#93c5fd', desc: `${TOTAL_L0} nodes / 3 sub-nets` },
                  ].map(({ label, color, dot, desc }) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 7, background: 'rgba(5,10,20,0.75)', backdropFilter: 'blur(8px)', borderRadius: 8, padding: '4px 10px', border: `1px solid ${color}44` }}>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: dot, boxShadow: `0 0 6px ${color}` }} />
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.70rem', fontWeight: 700, color, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                        {label}
                      </span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.65rem', color: '#6b7280' }}>
                        {desc}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Congestion colour legend */}
                <div style={{
                  position: 'absolute', bottom: 12, left: 14,
                  display: 'flex', gap: 9, alignItems: 'center',
                  background: 'rgba(5,10,20,0.78)', backdropFilter: 'blur(8px)',
                  borderRadius: 99, padding: '5px 14px',
                  border: '1px solid rgba(255,255,255,0.08)',
                }}>
                  {[
                    { color: '#00e5a0', label: 'Free Flow' },
                    { color: '#ffe040', label: 'Light' },
                    { color: '#ff8c00', label: 'Moderate' },
                    { color: '#ff3333', label: 'Heavy' },
                    { color: '#cc00ff', label: 'Standstill' },
                  ].map(({ color, label }) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, boxShadow: `0 0 5px ${color}` }} />
                      <span style={{ fontSize: '0.76rem', color: '#a0b0c0', fontWeight: 600 }}>{label}</span>
                    </div>
                  ))}
                </div>

                {/* Active network badge */}
                {selectedNetwork && (
                  <div style={{
                    position: 'absolute', top: 12, right: 12,
                    background: NETWORK_DEFS[selectedNetwork].bgColor,
                    backdropFilter: 'blur(8px)',
                    border: `1px solid ${NETWORK_DEFS[selectedNetwork].borderColor}`,
                    borderRadius: 99, padding: '5px 16px',
                    fontSize: '0.82rem', fontWeight: 700,
                    fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.08em',
                    color: NETWORK_DEFS[selectedNetwork].textColor,
                    textTransform: 'uppercase',
                  }}>
                    {NETWORK_DEFS[selectedNetwork].icon} {NETWORK_DEFS[selectedNetwork].label} — Active
                  </div>
                )}
              </div>

              {/* Divider */}
              <div style={{ height: 1, background: 'linear-gradient(90deg, transparent, rgba(160,120,60,0.25), transparent)', flexShrink: 0 }} />

              {/* Forecast Table */}
              <div style={{ flex: '1 1 0', overflow: 'hidden', background: 'rgba(250,246,238,0.72)', backdropFilter: 'blur(12px)' }}>
                <NodeForecastTable
                  selectedNode={selectedNode}
                  onNodeSelect={handleNodeSelect}
                  nodeOverrides={nodeOverrides}
                  onForecastData={setForecastData}
                  params={params}
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
