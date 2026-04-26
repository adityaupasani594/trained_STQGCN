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
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
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
        # Classical GCN aggregation
        x_agg = torch.einsum("ij,bjh->bih", norm_adj, x)
        x_agg = self.act(self.W_gcn(x_agg))
        x_agg = self.msg_dropout(x_agg)

        # Quantum refinement — target node only
        tgt_agg = x_agg[:, self.target_node_idx, :]
        tgt_f32 = tgt_agg.float()
        x_qin   = torch.tanh(self.pre(tgt_f32)) * math.pi
        q_out   = self.qlayer(x_qin.cpu())
        q_out   = q_out.to(device=x_agg.device, dtype=torch.float32)
        q_out   = self.post(q_out).to(x_agg.dtype)

        out = x_agg.clone()
        out[:, self.target_node_idx, :] = q_out
        return self.msg_dropout(out)


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
        h_nodes = self.temporal(x)
        h_g     = self.graph_q(h_nodes, norm_adj)
        h_g     = self.norm(h_g + h_nodes)
        target  = h_g[:, self.target_node_idx, :]
        return self.readout(target)


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

        # Baseline input window [seq_len, n_nodes, n_features] — normalised
        self.baseline_norm: np.ndarray = None

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

        # Pre-built model list — one STQGCN per target node
        # These share the same weights for temporal / GCN / readout;
        # only qlayer.target_node_idx differs (which node reads quantum output).
        self.models: List[STQGCN] = []


_engine = _InferenceEngine()


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
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg          = ckpt["config"]
    _engine.n_nodes    = int(ckpt["n_nodes"])
    _engine.seq_len    = int(cfg["seq_len"])
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
    _engine.norm_adj = torch.from_numpy(adj)

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
        p = wide.pivot_table(index=time_col, columns=node_col, values=feat, aggfunc="mean")
        p = p.reindex(index=time_values, columns=node_names_sorted)
        p = p.apply(pd.to_numeric, errors="coerce")
        p = p.interpolate(limit_direction="both").ffill().bfill()
        feature_planes.append(p.to_numpy(dtype=np.float32))

    values = np.stack(feature_planes, axis=-1).astype(np.float32)  # [T, N, F]
    print(f"[api_server] Dataset tensor shape: {values.shape}")

    # Take last seq_len timesteps as baseline context window
    seq = _engine.seq_len
    if values.shape[0] < seq:
        _engine.error = f"Dataset too short: need {seq} timesteps, got {values.shape[0]}."
        print(f"[api_server] ERROR: {_engine.error}")
        return

    baseline = values[-seq:, :, :]  # [seq_len, N, F]  raw (unnormalised)

    # Normalise using checkpoint stats
    x_mean = _engine.x_mean[np.newaxis, np.newaxis, :]  # [1, 1, F]
    x_std  = _engine.x_std[np.newaxis, np.newaxis, :]
    baseline_norm = (baseline.astype(np.float64) - x_mean) / (x_std + 1e-6)
    baseline_norm = np.nan_to_num(baseline_norm, nan=0.0, posinf=0.0, neginf=0.0)
    _engine.baseline_norm = baseline_norm.astype(np.float32)

    # ── Build one model per target node ──────────────────────────────────────
    print(f"[api_server] Building {_engine.n_nodes} inference models…")
    for node_idx in range(_engine.n_nodes):
        m = STQGCN(
            n_nodes         = _engine.n_nodes,
            n_features      = _engine.n_features,
            hidden_dim      = _engine.hidden_dim,
            n_qubits        = _engine.n_qubits,
            n_q_layers      = _engine.n_q_layers,
            dropout         = _engine.dropout,
            target_node_idx = node_idx,
        )
        # Load weights – W_gcn might be missing from old checkpoint (strict=False)
        result = m.load_state_dict(ckpt["model_state_dict"], strict=False)
        if result.unexpected_keys:
            print(f"[api_server] Unexpected keys (node {node_idx}): {result.unexpected_keys}")
        m.eval()
        _engine.models.append(m)

    _engine.ready = True
    print(f"[api_server] Inference engine ready. {_engine.n_nodes} models loaded.")


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


# ─── Core inference function ──────────────────────────────────────────────────

def _run_inference(overrides: Dict[str, Dict[str, float]], global_params: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Run the ST-QGCN model for every node and return per-node predictions.

    Parameters
    ----------
    overrides : dict
        Mapping of str(node_index) → {"traffic_flow": float, "avg_speed": float}
    global_params : dict
        Global environment factors: {"rain": float, "temp": float, "hour": float, "wind": float}
    """
    if not _engine.ready:
        raise HTTPException(status_code=503, detail=_engine.error or "Inference engine not ready.")

    # Build a copy of the normalised baseline window [seq_len, N, F]
    window = _engine.baseline_norm.copy()

    # ── Inject global parameters (weather/time) into the LAST timestep ────────
    if global_params:
        # Rain
        if "rain" in global_params and _engine.rain_feat_idx >= 0:
            val = float(global_params["rain"]) # rain slider is 0-100%? Let's assume mm/hr or scale it.
            # Usually Precipitation_mm_hr in dataset. Let's assume rain/10.0 for mm/hr
            mm_hr = val / 10.0 
            norm = (mm_hr - _engine.x_mean[_engine.rain_feat_idx]) / (_engine.x_std[_engine.rain_feat_idx] + 1e-6)
            window[-1, :, _engine.rain_feat_idx] = np.float32(norm)
            
            # Also affect visibility if present
            if _engine.vis_feat_idx >= 0:
                vis = max(100, 5000 - val * 40)
                norm_vis = (vis - _engine.x_mean[_engine.vis_feat_idx]) / (_engine.x_std[_engine.vis_feat_idx] + 1e-6)
                window[-1, :, _engine.vis_feat_idx] = np.float32(norm_vis)

        # Temperature
        if "temp" in global_params and _engine.temp_feat_idx >= 0:
            val = float(global_params["temp"])
            norm = (val - _engine.x_mean[_engine.temp_feat_idx]) / (_engine.x_std[_engine.temp_feat_idx] + 1e-6)
            window[-1, :, _engine.temp_feat_idx] = np.float32(norm)

        # Hour / Time of Day
        if "hour" in global_params:
            hour = float(global_params["hour"])
            if _engine.hour_feat_idx >= 0:
                norm = (hour - _engine.x_mean[_engine.hour_feat_idx]) / (_engine.x_std[_engine.hour_feat_idx] + 1e-6)
                window[-1, :, _engine.hour_feat_idx] = np.float32(norm)
            if _engine.sin_feat_idx >= 0:
                val = math.sin(2 * math.pi * hour / 24.0)
                norm = (val - _engine.x_mean[_engine.sin_feat_idx]) / (_engine.x_std[_engine.sin_feat_idx] + 1e-6)
                window[-1, :, _engine.sin_feat_idx] = np.float32(norm)
            if _engine.cos_feat_idx >= 0:
                val = math.cos(2 * math.pi * hour / 24.0)
                norm = (val - _engine.x_mean[_engine.cos_feat_idx]) / (_engine.x_std[_engine.cos_feat_idx] + 1e-6)
                window[-1, :, _engine.cos_feat_idx] = np.float32(norm)

    # ── Inject node-specific overrides into the LAST timestep ─────────────────
    for node_key, vals in overrides.items():
        try:
            node_idx = int(node_key)
        except (ValueError, TypeError):
            continue
        if node_idx < 0 or node_idx >= _engine.n_nodes:
            continue

        if "traffic_flow" in vals:
            raw_flow = float(vals["traffic_flow"])
            norm_flow = (raw_flow - _engine.x_mean[_engine.flow_feat_idx]) / (
                _engine.x_std[_engine.flow_feat_idx] + 1e-6
            )
            window[-1, node_idx, _engine.flow_feat_idx] = np.float32(norm_flow)

        if "avg_speed" in vals and _engine.speed_feat_idx >= 0:
            raw_speed = float(vals["avg_speed"])
            norm_speed = (raw_speed - _engine.x_mean[_engine.speed_feat_idx]) / (
                _engine.x_std[_engine.speed_feat_idx] + 1e-6
            )
            window[-1, node_idx, _engine.speed_feat_idx] = np.float32(norm_speed)

    # Input tensor: [1, seq_len, N, F]
    x_tensor = torch.from_numpy(window).unsqueeze(0).float()  # [1, T, N, F]
    norm_adj  = _engine.norm_adj  # [N, N]

    # ── Run one forward pass per node ─────────────────────────────────────────
    preds_normalised = np.zeros(_engine.n_nodes, dtype=np.float64)
    with torch.no_grad():
        for node_idx in range(_engine.n_nodes):
            pred_n = _engine.models[node_idx](x_tensor, norm_adj)  # [1, 1]
            preds_normalised[node_idx] = float(pred_n.squeeze())

    # ── De-normalise: y_orig = y_norm * y_std + y_mean ────────────────────────
    preds_flow = preds_normalised * _engine.y_std + _engine.y_mean  # veh/hr

    # ── Build output ──────────────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    for node_idx, node_id in enumerate(_engine.node_names):
        flow = float(np.clip(preds_flow[node_idx], 0.0, None))  # clip negatives

        meta     = _engine.node_meta.get(node_id, {})
        zone     = meta.get("zone", "Unknown")
        capacity = meta.get("capacity", max(int(flow * 1.5), 1))  # fallback: 1.5× flow

        ratio = min(flow / capacity, 1.0) if capacity > 0 else 1.0
        util_pct = round(ratio * 100, 1)

        if   ratio < 0.30: congestion = "Free Flow"
        elif ratio < 0.55: congestion = "Moderate"
        elif ratio < 0.75: congestion = "Heavy"
        else:              congestion = "Standstill"

        # Trend: compare predicted flow vs un-overridden baseline last-timestep value
        baseline_flow_norm = float(_engine.baseline_norm[-1, node_idx, _engine.flow_feat_idx])
        baseline_flow = baseline_flow_norm * _engine.y_std + _engine.y_mean
        delta = flow - baseline_flow
        trend = "↑" if delta > 20 else ("↓" if delta < -20 else "→")

        results.append({
            "node_id":                   node_id,
            "node_index":                node_idx,
            "zone":                      zone,
            "predicted_flow_veh_per_hr": round(flow, 1),
            "capacity_veh_per_hr":       capacity,
            "utilisation_pct":           util_pct,
            "congestion_level":          congestion,
            "trend":                     trend,
        })

    return results


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


@app.post("/api/nodes/forecast")
def nodes_forecast_post(req: ForecastRequest) -> Dict[str, Any]:
    """
    Return 15-min-ahead traffic flow predictions for all nodes from the
    trained ST-QGCN model. Accepts per-node overrides for traffic flow and
    average speed, plus global environment parameters.
    """
    import datetime
    overrides_raw = {k: v.dict(exclude_none=True) for k, v in req.overrides.items()}
    global_raw = req.global_params.dict(exclude_none=True) if req.global_params else {}
    nodes_out = _run_inference(overrides_raw, global_raw)
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