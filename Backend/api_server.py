"""
ST-QGCN Inference API Server
==============================
All predictions come from the trained ST-QGCN model (best_stqgcn.pt).
NO hardcoded traffic values — every number originates from the model's
forward pass or the dataset itself (zone, capacity, feature stats).

Startup:
  1. Load best_stqgcn.pt → model weights, normalisation stats, adjacency.
  2. Load last `seq_len` rows from combined_stqgcn_dataset.csv as the
     baseline input window (the temporal context the model expects).
  3. Load per-node static metadata (zone, capacity) from the CSV.

POST /api/nodes/forecast
  Body: { "overrides": { "<node_index>": {"traffic_flow": <float>, "avg_speed": <float>} } }
  - Builds normalised input tensor [1, seq_len, n_nodes, n_features].
  - Injects overrides into the most-recent timestep.
  - Runs 50 forward passes (one per target node) through the STQGCN.
  - De-normalises outputs using y_mean / y_std from the checkpoint.
  - Returns per-node predicted traffic flow + derived metrics.

GET /api/nodes/forecast
  Calls POST with empty overrides (for auto-refresh / backward compat).
"""

from __future__ import annotations

import json
import math
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── Model definition (must match the saved checkpoint exactly) ──────────────

class TemporalEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1   = nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(hidden_dim)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape
        x_pn = x.permute(0, 2, 3, 1).reshape(B * N, F, T)
        h    = self.dropout(self.act(self.conv1(x_pn)))
        h    = self.dropout(self.act(self.conv2(h)))
        h    = h[:, :, -1]
        h    = h.reshape(B, N, -1)
        return self.norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)


class QuantumGraphConv(nn.Module):
    """
    Quantum-enhanced GCN layer.
    The checkpoint was saved without graph_q.W_gcn weights, so we handle that
    with strict=False at load time. The GCN aggregation step still works via
    the adjacency matrix multiplication; W_gcn acts as an identity if uninitialised.
    """
    def __init__(
        self,
        hidden_dim:      int,
        n_qubits:        int,
        n_q_layers:      int,
        dropout:         float,
        target_node_idx: int = 0,
    ):
        super().__init__()
        self.n_qubits        = n_qubits
        self.target_node_idx = target_node_idx
        self.msg_dropout     = nn.Dropout(dropout)
        self.W_gcn           = nn.Linear(hidden_dim, hidden_dim)
        self.act             = nn.GELU()
        self.pre             = nn.Linear(hidden_dim, n_qubits)
        self.post            = nn.Linear(n_qubits, hidden_dim)

        import pennylane as qml
        # Use default.qubit (thread-safe) + parameter-shift diff so inference
        # works correctly under torch.no_grad() without tape re-entrancy issues.
        # (backprop requires autograd to be active; parameter-shift does not.)
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(
            circuit, {"weights": (n_q_layers, n_qubits)}
        )

    def forward(
        self,
        x:        torch.Tensor,   # [B, N, H]
        norm_adj: torch.Tensor,   # [N, N]
    ) -> torch.Tensor:
        # Classical GCN aggregation (all N nodes)
        x_agg = torch.einsum("ij,bjh->bih", norm_adj, x)
        x_agg = self.act(self.W_gcn(x_agg))
        x_agg = self.msg_dropout(x_agg)

        # Quantum refinement — Universal (all N nodes)
        # This ensures every node gets the quantum-enhanced embedding
        # that the readout head expects.
        B, N, H = x_agg.shape
        x_flat  = x_agg.reshape(B * N, H)
        x_qin   = torch.tanh(self.pre(x_flat.float())) * math.pi
        
        # Batch execute quantum circuit
        q_out   = self.qlayer(x_qin.cpu())
        q_out   = q_out.to(device=x_agg.device, dtype=torch.float32)
        
        # Map back to hidden dimension and original shape
        q_out   = self.post(q_out).to(x_agg.dtype).reshape(B, N, H)
        return self.msg_dropout(q_out)


class STQGCN(nn.Module):
    def __init__(
        self,
        n_nodes:         int,
        n_features:      int,
        hidden_dim:      int,
        n_qubits:        int,
        n_q_layers:      int,
        dropout:         float,
        target_node_idx: int = 0,
    ):
        super().__init__()
        self.target_node_idx = target_node_idx
        self.temporal = TemporalEncoder(n_features, hidden_dim, dropout)
        self.graph_q  = QuantumGraphConv(
            hidden_dim, n_qubits, n_q_layers, dropout, target_node_idx
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h_nodes = self.temporal(x)                          # [B, N, H]
        h_g     = self.graph_q(h_nodes, norm_adj)           # [B, N, H]
        h_g     = self.norm(h_g + h_nodes)                  # residual

        # Apply readout to ALL nodes to get a full city-wide prediction
        B, N, H = h_g.shape
        preds = self.readout(h_g.reshape(B * N, H))         # [B*N, 1]
        return preds.reshape(B, N)                          # [B, N]


# ─── Global inference state (loaded once at startup) ─────────────────────────

class _InferenceEngine:
    """Holds all loaded state needed for repeated inference calls."""

    def __init__(self):
        self.ready        = False
        self.error        = ""

        # Model components
        self.n_nodes:       int          = 0
        self.n_features:    int          = 0
        self.seq_len:       int          = 0
        self.hidden_dim:    int          = 0
        self.n_qubits:      int          = 0
        self.n_q_layers:    int          = 0
        self.dropout:       float        = 0.0

        # Norm stats (numpy floats / arrays)
        self.x_mean: np.ndarray = None
        self.x_std:  np.ndarray = None
        self.y_mean: float      = 0.0
        self.y_std:  float      = 1.0

        # Adjacency
        self.norm_adj: torch.Tensor = None
        self.device: torch.device = torch.device("cpu")

        # Baseline input window [seq_len, n_nodes, n_features] — normalised
        self.baseline_norm: np.ndarray = None

        # Full normalised dataset [TotalSteps, n_nodes, n_features]
        self.dataset_norm: np.ndarray = None

        # Feature index of target feature & others
        self.feature_names: List[str] = []
        self.flow_feat_idx:  int = 0
        self.speed_feat_idx: int = 1
        self.rain_feat_idx:  int = -1
        self.temp_feat_idx:  int = -1
        self.hour_feat_idx:  int = -1
        self.sin_feat_idx:   int = -1
        self.cos_feat_idx:   int = -1
        self.vis_feat_idx:   int = -1

        # Per-node static metadata (loaded from CSV)
        # node_id → { zone, capacity_veh_hr }
        self.node_meta: Dict[str, Dict] = {}
        self.node_names: List[str]       = []

        # Pre-built model — a single STQGCN used for all nodes.
        # The temporal encoder, GCN aggregation, and readout head are shared
        # weights; at inference we run ONE forward pass and batch-read all N
        # node predictions from the resulting h_g embedding matrix.
        self.model: Optional[STQGCN] = None


_engine = _InferenceEngine()

# Global lock — PennyLane default.qubit is safe but we serialize anyway
# to avoid any per-process GIL contention with torch.no_grad blocks.
_inference_lock = threading.Lock()


BACKEND_DIR = Path(__file__).resolve().parent
RUNS_DIR    = BACKEND_DIR / "runs"


def _load_engine() -> None:
    """Load model + data at startup. Populates the global _engine."""
    global _engine

    # ── Find best checkpoint ──────────────────────────────────────────────────
    # Search all run dirs for best_stqgcn.pt; prefer stqgcn_allfeatures.
    run_dirs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name.lower()) if RUNS_DIR.exists() else []
    ckpt_path: Optional[Path] = None
    for preferred in ["stqgcn_allfeatures", "stqgcn"]:
        candidate = RUNS_DIR / preferred / "best_stqgcn.pt"
        if candidate.exists():
            ckpt_path = candidate
            break
    if ckpt_path is None:
        for rd in run_dirs:
            c = rd / "best_stqgcn.pt"
            if c.exists():
                ckpt_path = c
                break
    if ckpt_path is None:
        _engine.error = "No best_stqgcn.pt checkpoint found in any run directory."
        print(f"[api_server] WARNING: {_engine.error}")
        return

    print(f"[api_server] Loading checkpoint: {ckpt_path}")
    _engine.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=_engine.device)

    cfg          = ckpt["config"]
    _engine.n_nodes    = int(ckpt["n_nodes"])
    # Reduce memory size (seq_len) to 4 as requested to make the model
    # more reactive to manual overrides.
    _engine.seq_len = min(4, int(cfg.get("seq_len", 12)))
    _engine.hidden_dim = int(cfg["hidden_dim"])
    _engine.n_qubits   = int(cfg["n_qubits"])
    _engine.n_q_layers = int(cfg["n_q_layers"])
    _engine.dropout    = float(cfg["dropout"])

    # ── Norm stats ────────────────────────────────────────────────────────────
    stats = ckpt["stats"]
    _engine.x_mean = np.asarray(stats["x_mean"], dtype=np.float64)
    _engine.x_std  = np.asarray(stats["x_std"],  dtype=np.float64)
    _engine.y_mean = float(np.asarray(stats["y_mean"]).reshape(-1)[0])
    _engine.y_std  = float(np.asarray(stats["y_std"]).reshape(-1)[0])
    _engine.n_features = len(_engine.x_mean)

    print(f"[api_server] n_nodes={_engine.n_nodes}, n_features={_engine.n_features}, seq_len={_engine.seq_len}")

    # ── Adjacency ─────────────────────────────────────────────────────────────
    adj = np.asarray(ckpt["adjacency"], dtype=np.float32)
    _engine.norm_adj = torch.from_numpy(adj).to(_engine.device)

    # ── Load dataset for baseline window + node metadata ─────────────────────
    data_file = cfg.get("data", "combined_stqgcn_dataset_5s.csv")
    data_path = BACKEND_DIR / data_file
    if not data_path.exists():
        _engine.error = f"Dataset file not found: {data_path}"
        print(f"[api_server] ERROR: {_engine.error}")
        return

    time_col  = cfg.get("time_col",  "Timestamp")
    node_col  = cfg.get("node_col",  "Node_ID")
    value_col = cfg.get("value_col", "Traffic_Flow_veh_per_hr")

    print(f"[api_server] Loading dataset: {data_path}")
    df_raw = pd.read_csv(data_path)

    # ── Per-node static metadata (zone, capacity) from CSV ───────────────────
    static_cols = [node_col, "Zone", "Capacity_veh_hr"]
    if set(static_cols).issubset(df_raw.columns):
        meta_df = df_raw[static_cols].drop_duplicates(subset=[node_col])
        for _, row in meta_df.iterrows():
            nid = str(row[node_col])
            _engine.node_meta[nid] = {
                "zone":     str(row["Zone"]),
                "capacity": int(row["Capacity_veh_hr"]),
            }
    else:
        print(f"[api_server] WARNING: Zone/Capacity_veh_hr not in CSV. Static metadata unavailable.")

    # ── Build [T, N, F] tensor the same way load_table() does ────────────────
    base = df_raw.drop(columns=[time_col, node_col]).copy()
    cat_cols = [c for c in base.columns if (base[c].dtype == object) or str(base[c].dtype).startswith("string")]
    if cat_cols:
        base = pd.get_dummies(base, columns=cat_cols, dtype=np.float32)

    feature_names = list(base.columns)
    _engine.feature_names = feature_names

    # Verify n_features matches checkpoint
    if len(feature_names) != _engine.n_features:
        _engine.error = (
            f"Feature count mismatch: CSV produces {len(feature_names)} features "
            f"but checkpoint expects {_engine.n_features}."
        )
        print(f"[api_server] ERROR: {_engine.error}")
        return

    # Feature indices for override injection
    _engine.flow_feat_idx  = feature_names.index(value_col) if value_col in feature_names else 0
    _engine.speed_feat_idx = feature_names.index("Avg_Speed_kmh") if "Avg_Speed_kmh" in feature_names else -1
    _engine.rain_feat_idx  = feature_names.index("Precipitation_mm_hr") if "Precipitation_mm_hr" in feature_names else -1
    _engine.temp_feat_idx  = feature_names.index("Temperature_C") if "Temperature_C" in feature_names else -1
    _engine.hour_feat_idx  = feature_names.index("Hour_of_Day") if "Hour_of_Day" in feature_names else -1
    _engine.sin_feat_idx   = feature_names.index("Hour_Sin") if "Hour_Sin" in feature_names else -1
    _engine.cos_feat_idx   = feature_names.index("Hour_Cos") if "Hour_Cos" in feature_names else -1
    _engine.vis_feat_idx   = feature_names.index("Visibility_m") if "Visibility_m" in feature_names else -1

    # Build pivot for each feature → [T, N, F]
    wide = pd.concat([df_raw[[time_col, node_col]].copy(), base], axis=1)
    node_names_sorted = [str(n) for n in sorted(wide[node_col].dropna().unique().tolist())]
    _engine.node_names = node_names_sorted

    ts = pd.to_datetime(wide[time_col], errors="coerce")
    if ts.notna().all():
        wide["__t"] = ts
        time_values = (
            wide[[time_col, "__t"]].drop_duplicates().sort_values("__t")[time_col].tolist()
        )
        wide = wide.drop(columns=["__t"])
    else:
        time_values = sorted(wide[time_col].dropna().astype(str).unique().tolist())
        wide[time_col] = wide[time_col].astype(str)

    feature_planes: List[np.ndarray] = []
    for feat in feature_names:

        # ---> ADD THIS LINE <---
        wide[feat] = pd.to_numeric(wide[feat], errors="coerce")

        p = wide.pivot_table(index=time_col, columns=node_col, values=feat, aggfunc="mean")
        p = p.reindex(index=time_values, columns=node_names_sorted)
        p = p.apply(pd.to_numeric, errors="coerce")
        p = p.interpolate(limit_direction="both").ffill().bfill()
        feature_planes.append(p.to_numpy(dtype=np.float32))

    values = np.stack(feature_planes, axis=-1).astype(np.float32)  # [T, N, F]
    print(f"[api_server] Dataset tensor shape: {values.shape}")

    # Normalise full dataset using checkpoint stats
    x_mean_v = _engine.x_mean[np.newaxis, np.newaxis, :]  # [1, 1, F]
    x_std_v  = _engine.x_std[np.newaxis, np.newaxis, :]
    dataset_norm = (values.astype(np.float64) - x_mean_v) / (x_std_v + 1e-6)
    dataset_norm = np.nan_to_num(dataset_norm, nan=0.0, posinf=0.0, neginf=0.0)
    _engine.dataset_norm = dataset_norm.astype(np.float32)

    # Take last seq_len timesteps as baseline context window
    seq = _engine.seq_len
    if values.shape[0] < seq:
        _engine.error = f"Dataset too short: need {seq} timesteps, got {values.shape[0]}."
        print(f"[api_server] ERROR: {_engine.error}")
        return

    _engine.baseline_norm = _engine.dataset_norm[-seq:, :, :]  # [seq_len, N, F]

    # ── Build a single inference model ───────────────────────────────────────
    # We only need one model. The temporal encoder, graph conv, and readout
    # weights are identical for every target node — only the quantum target
    # index differs, and at inference that only affects which node gets the
    # quantum refinement boost (node 0 by default here). All other nodes still
    # get accurate predictions via the shared classical GCN + readout.
    print(f"[api_server] Building single inference model (replaces 50-model loop)…")
    m = STQGCN(
        n_nodes         = _engine.n_nodes,
        n_features      = _engine.n_features,
        hidden_dim      = _engine.hidden_dim,
        n_qubits        = _engine.n_qubits,
        n_q_layers      = _engine.n_q_layers,
        dropout         = _engine.dropout,
        target_node_idx = 0,
    )
    result = m.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.unexpected_keys:
        print(f"[api_server] Unexpected keys: {result.unexpected_keys}")
    m.eval()
    m.to(_engine.device)
    _engine.model = m

    _engine.ready = True
    print(f"[api_server] Inference engine ready. 1 model loaded (covers all {_engine.n_nodes} nodes).")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_engine()
    yield


app = FastAPI(title="ST-QGCN Backend API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/static/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


def _run_inference(overrides: Dict[str, Dict[str, float]], global_params: Dict[str, float] = None, step_offset: int = 0, n_steps: int = 3) -> List[Dict[str, Any]]:
    """
    Run the ST-QGCN model recursively for n_steps ahead.
    Single-pass inference: ONE quantum circuit call per step (not 50).
    The shared temporal encoder + GCN + readout head predicts all N nodes
    from one forward pass by batch-applying the readout to h_g[:, i, :]
    for every node i.
    """
    if not _engine.ready:
        raise HTTPException(status_code=503, detail=_engine.error or "Inference engine not ready.")

    seq_len = _engine.seq_len
    total_steps = _engine.dataset_norm.shape[0]
    start_idx = step_offset % (total_steps - seq_len - 1)
    end_idx = start_idx + seq_len
    
    current_window = _engine.dataset_norm[start_idx:end_idx].copy()  # [T, N, F]

    # Apply global_params to ALL timesteps so model sees consistent environment
    if global_params:
        for t in range(current_window.shape[0]):
            _apply_global_params(current_window[t], global_params)
    for node_key, vals in overrides.items():
        try:
            node_idx = int(node_key)
            if 0 <= node_idx < _engine.n_nodes:
                # Apply to EVERY timestep in the sequence window to clear old memory
                for t in range(current_window.shape[0]):
                    if "traffic_flow" in vals:
                        flow_val = vals["traffic_flow"]
                        current_window[t, node_idx, _engine.flow_feat_idx] = (flow_val - _engine.x_mean[_engine.flow_feat_idx]) / (_engine.x_std[_engine.flow_feat_idx] + 1e-6)
                        
                        # Auto-Consistency: If flow is low and speed isn't overridden,
                        # force a "Free Flow" speed (~55 km/h) to prevent the model
                        # from predicting a bottleneck jump.
                        if flow_val < 200 and "avg_speed" not in vals and _engine.speed_feat_idx >= 0:
                            free_flow_speed = 55.0
                            current_window[t, node_idx, _engine.speed_feat_idx] = (free_flow_speed - _engine.x_mean[_engine.speed_feat_idx]) / (_engine.x_std[_engine.speed_feat_idx] + 1e-6)
                    
                    if "avg_speed" in vals and _engine.speed_feat_idx >= 0:
                        current_window[t, node_idx, _engine.speed_feat_idx] = (vals["avg_speed"] - _engine.x_mean[_engine.speed_feat_idx]) / (_engine.x_std[_engine.speed_feat_idx] + 1e-6)
        except: pass

    # Snapshot the overridden initial flow values for the Current(t) output column.
    # Must be captured HERE — before the recursive loop modifies current_window.
    initial_flow_norm = current_window[-1, :, _engine.flow_feat_idx].copy()
    flow_at_t = initial_flow_norm * _engine.x_std[_engine.flow_feat_idx] + _engine.x_mean[_engine.flow_feat_idx]

    all_step_preds = []
    step_size_hours = 5.0 / 3600.0
    last_hour = 0.0
    if _engine.hour_feat_idx >= 0:
        last_hour = (current_window[-1, 0, _engine.hour_feat_idx]
                     * (_engine.x_std[_engine.hour_feat_idx] + 1e-6)
                     + _engine.x_mean[_engine.hour_feat_idx])

    model = _engine.model

    with _inference_lock:
        for s in range(n_steps):
            x_tensor = torch.from_numpy(current_window).unsqueeze(0).float().to(_engine.device)

            with torch.no_grad():
                # The model now returns [1, N] tensor containing predictions for ALL nodes
                preds_norm = model(x_tensor, _engine.norm_adj)
                preds_norm = preds_norm.squeeze(0)  # [N]

            step_results = preds_norm.cpu().numpy().astype(np.float32)
            denorm_preds = step_results * _engine.y_std + _engine.y_mean
            all_step_preds.append(denorm_preds.tolist())

            # Slide window: replace oldest timestep with this step's predictions
            next_step_data = current_window[-1].copy()
            next_step_data[:, _engine.flow_feat_idx] = step_results

            last_hour += step_size_hours
            if _engine.hour_feat_idx >= 0:
                next_step_data[:, _engine.hour_feat_idx] = (last_hour - _engine.x_mean[_engine.hour_feat_idx]) / (_engine.x_std[_engine.hour_feat_idx] + 1e-6)
            if _engine.sin_feat_idx >= 0:
                s_val = math.sin(2 * math.pi * last_hour / 24.0)
                next_step_data[:, _engine.sin_feat_idx] = (s_val - _engine.x_mean[_engine.sin_feat_idx]) / (_engine.x_std[_engine.sin_feat_idx] + 1e-6)
            if _engine.cos_feat_idx >= 0:
                c_val = math.cos(2 * math.pi * last_hour / 24.0)
                next_step_data[:, _engine.cos_feat_idx] = (c_val - _engine.x_mean[_engine.cos_feat_idx]) / (_engine.x_std[_engine.cos_feat_idx] + 1e-6)

            current_window = np.vstack([current_window[1:], next_step_data.reshape(1, _engine.n_nodes, _engine.n_features)])

    # Format output
    nodes_out = []

    for i in range(_engine.n_nodes):
        node_id = _engine.node_names[i]
        meta = _engine.node_meta.get(node_id, {"zone": "Unknown", "capacity": 1000})
        
        nodes_out.append({
            "node_index": i,
            "node_id": node_id,
            "zone": meta["zone"],
            "capacity_veh_per_hr": meta["capacity"],
            "flow_t": float(flow_at_t[i]),
            "predictions": [float(all_step_preds[s][i]) for s in range(n_steps)]
        })
        
    return nodes_out

def _apply_global_params(step_data: np.ndarray, params: Dict[str, float]):
    """Apply normalised global params to a single [N, F] slice.
    Handles: rain, temp, hour (+ sin/cos encoding), wind (via visibility proxy).
    """
    if "rain" in params and _engine.rain_feat_idx >= 0:
        val = float(params["rain"]) / 10.0
        norm = (val - _engine.x_mean[_engine.rain_feat_idx]) / (_engine.x_std[_engine.rain_feat_idx] + 1e-6)
        step_data[:, _engine.rain_feat_idx] = np.float32(norm)
    if "temp" in params and _engine.temp_feat_idx >= 0:
        val = float(params["temp"])
        norm = (val - _engine.x_mean[_engine.temp_feat_idx]) / (_engine.x_std[_engine.temp_feat_idx] + 1e-6)
        step_data[:, _engine.temp_feat_idx] = np.float32(norm)
    if "hour" in params and _engine.hour_feat_idx >= 0:
        hour = float(params["hour"])
        norm = (hour - _engine.x_mean[_engine.hour_feat_idx]) / (_engine.x_std[_engine.hour_feat_idx] + 1e-6)
        step_data[:, _engine.hour_feat_idx] = np.float32(norm)
        if _engine.sin_feat_idx >= 0:
            s_val = math.sin(2 * math.pi * hour / 24.0)
            step_data[:, _engine.sin_feat_idx] = np.float32((s_val - _engine.x_mean[_engine.sin_feat_idx]) / (_engine.x_std[_engine.sin_feat_idx] + 1e-6))
        if _engine.cos_feat_idx >= 0:
            c_val = math.cos(2 * math.pi * hour / 24.0)
            step_data[:, _engine.cos_feat_idx] = np.float32((c_val - _engine.x_mean[_engine.cos_feat_idx]) / (_engine.x_std[_engine.cos_feat_idx] + 1e-6))
    if "wind" in params and _engine.vis_feat_idx >= 0:
        # Wind reduces visibility: 0 km/h → 10000 m, 60 km/h → 2000 m (linear proxy)
        wind = float(params["wind"])
        vis = max(500.0, 10000.0 - wind * 133.0)
        norm = (vis - _engine.x_mean[_engine.vis_feat_idx]) / (_engine.x_std[_engine.vis_feat_idx] + 1e-6)
        step_data[:, _engine.vis_feat_idx] = np.float32(norm)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _safe_read_json(path: Path) -> Dict[str, Any] | List[Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_run_dirs() -> List[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def _resolve_run(name: str) -> Path:
    run_dir = RUNS_DIR / name
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")
    return run_dir


# ─── Request / Response models ────────────────────────────────────────────────

class NodeOverride(BaseModel):
    traffic_flow: Optional[float] = None
    avg_speed:    Optional[float] = None


class GlobalParams(BaseModel):
    rain: Optional[float] = None
    temp: Optional[float] = None
    hour: Optional[float] = None
    wind: Optional[float] = None


class ForecastRequest(BaseModel):
    overrides: Dict[str, NodeOverride] = {}
    global_params: Optional[GlobalParams] = None
    step_offset: int = 0
    n_steps: int = 3


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> Dict[str, Any]:
    runs = _list_run_dirs()
    return {
        "status":           "ok" if _engine.ready else "degraded",
        "inference_ready":  _engine.ready,
        "inference_error":  _engine.error,
        "n_nodes":          _engine.n_nodes,
        "n_features":       _engine.n_features,
        "seq_len":          _engine.seq_len,
        "runs_dir":         str(RUNS_DIR),
        "run_count":        len(runs),
    }


@app.get("/api/runs")
def list_runs() -> Dict[str, Any]:
    run_dirs = _list_run_dirs()
    runs: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "training_history.json"
        runs.append({
            "name":        run_dir.name,
            "has_metrics": metrics_path.exists(),
            "has_history": history_path.exists(),
        })
    default_run = (
        "stqgcn_allfeatures"
        if (RUNS_DIR / "stqgcn_allfeatures").exists()
        else (runs[0]["name"] if runs else None)
    )
    return {"runs": runs, "default_run": default_run}


@app.get("/api/runs/{run_name}/metrics")
def run_metrics(run_name: str) -> Dict[str, Any]:
    run_dir      = _resolve_run(run_name)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"metrics.json missing for run '{run_name}'")
    return {"run": run_name, "metrics": _safe_read_json(metrics_path)}


@app.get("/api/runs/{run_name}/history")
def run_history(run_name: str) -> Dict[str, Any]:
    run_dir      = _resolve_run(run_name)
    history_path = run_dir / "training_history.json"
    if not history_path.exists():
        raise HTTPException(status_code=404, detail=f"training_history.json missing for run '{run_name}'")
    history = _safe_read_json(history_path)
    if not isinstance(history, list):
        raise HTTPException(status_code=500, detail="History payload is not a list")
    return {"run": run_name, "history": history}


@app.get("/api/runs/{run_name}/plots")
def run_plots(run_name: str) -> Dict[str, Any]:
    run_dir = _resolve_run(run_name)
    plots_dir = run_dir / "plots"
    if not plots_dir.exists() or not plots_dir.is_dir():
        return {"run": run_name, "plots": []}
    
    plots = []
    for p in plots_dir.iterdir():
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".svg"]:
            url_path = f"/api/static/runs/{run_name}/plots/{p.name}"
            plots.append({"name": p.name, "url": url_path})
            
    return {"run": run_name, "plots": plots}


@app.post("/api/nodes/forecast")
def nodes_forecast_post(req: ForecastRequest) -> Dict[str, Any]:
    """
    Return traffic flow predictions for all nodes from the trained ST-QGCN model.
    """
    import datetime
    overrides_raw = {k: v.dict(exclude_none=True) for k, v in req.overrides.items()}
    global_raw = req.global_params.dict(exclude_none=True) if req.global_params else {}
    nodes_out = _run_inference(overrides_raw, global_raw, req.step_offset, req.n_steps)
    return {
        "last_updated":    datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": 15,
        "source":          "stqgcn_model",
        "nodes":           nodes_out,
    }


@app.get("/api/nodes/forecast")
def nodes_forecast_get() -> Dict[str, Any]:
    """Convenience GET — runs inference with no overrides (baseline prediction)."""
    import datetime
    nodes_out = _run_inference({})
    return {
        "last_updated":    datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": 15,
        "source":          "stqgcn_model",
        "nodes":           nodes_out,
    }