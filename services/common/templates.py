import os
from pathlib import Path
from typing import Optional

from services.config.loader import load_config, get_template_path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    return ROOT_DIR / p


def load_slurm_template(path: Optional[str] = None, cluster_name: Optional[str] = None) -> str:
    """Load a SLURM template from an explicit path or config; no built-in fallback."""
    cfg = load_config()
    cfg_path = get_template_path(cfg, "slurm", cluster_name)

    candidates = []
    if path:
        candidates.append(Path(path))
    if cfg_path:
        candidates.append(Path(cfg_path))

    for p in candidates:
        if not p:
            continue
        resolved = _resolve_path(p)
        if resolved.exists():
            with open(resolved, "r", encoding="utf-8") as f:
                return f.read()

    raise FileNotFoundError("SLURM template not found; set templates.slurm in config or pass path explicitly")
