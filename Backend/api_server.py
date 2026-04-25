from pathlib import Path
from typing import Any, Dict, List

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


BACKEND_DIR = Path(__file__).resolve().parent
RUNS_DIR = BACKEND_DIR / "runs"


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


app = FastAPI(title="ST-QGCN Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    runs = _list_run_dirs()
    return {
        "status": "ok",
        "runs_dir": str(RUNS_DIR),
        "run_count": len(runs),
    }


@app.get("/api/runs")
def list_runs() -> Dict[str, Any]:
    run_dirs = _list_run_dirs()
    runs: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "training_history.json"
        runs.append(
            {
                "name": run_dir.name,
                "has_metrics": metrics_path.exists(),
                "has_history": history_path.exists(),
            }
        )

    # Prefer all-features run when available.
    default_run = "stqgcn_allfeatures" if (RUNS_DIR / "stqgcn_allfeatures").exists() else (runs[0]["name"] if runs else None)
    return {
        "runs": runs,
        "default_run": default_run,
    }


@app.get("/api/runs/{run_name}/metrics")
def run_metrics(run_name: str) -> Dict[str, Any]:
    run_dir = _resolve_run(run_name)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"metrics.json missing for run '{run_name}'")
    metrics = _safe_read_json(metrics_path)
    return {
        "run": run_name,
        "metrics": metrics,
    }


@app.get("/api/runs/{run_name}/history")
def run_history(run_name: str) -> Dict[str, Any]:
    run_dir = _resolve_run(run_name)
    history_path = run_dir / "training_history.json"
    if not history_path.exists():
        raise HTTPException(status_code=404, detail=f"training_history.json missing for run '{run_name}'")
    history = _safe_read_json(history_path)
    if not isinstance(history, list):
        raise HTTPException(status_code=500, detail="History payload is not a list")
    return {
        "run": run_name,
        "history": history,
    }
