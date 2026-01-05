from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "services" / "config" / "default_config.json"


def load_config(path: Optional[str] = None) -> Dict:
    """Load platform config JSON. Falls back to empty dict if missing."""
    cfg_path = Path(path) if path else CONFIG_PATH
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def get_active_cluster(cfg: Dict) -> str:
    return cfg.get("active_cluster") or "default"


def get_cluster(cfg: Dict, name: Optional[str] = None) -> Dict:
    clusters = cfg.get("clusters", {}) or {}
    return clusters.get(name or get_active_cluster(cfg), {})


def get_remote_server(cfg: Dict, cluster_name: Optional[str] = None) -> Dict:
    cluster = get_cluster(cfg, cluster_name)
    return cluster.get("remote_server") or cfg.get("remote_server") or {}


def get_remote_paths(cfg: Dict, cluster_name: Optional[str] = None) -> Dict:
    cluster = get_cluster(cfg, cluster_name)
    return cluster.get("remote_paths") or cfg.get("remote_paths") or {}


def get_queue_defaults(cfg: Dict, cluster_name: Optional[str] = None) -> Dict:
    cluster = get_cluster(cfg, cluster_name)
    return cluster.get("queue") or cfg.get("queue_system") or {}


def get_template_path(cfg: Dict, kind: str = "slurm", cluster_name: Optional[str] = None) -> Optional[str]:
    cluster = get_cluster(cfg, cluster_name)
    cluster_tpl = (cluster.get("templates") or {}).get(kind)
    if cluster_tpl:
        return cluster_tpl
    global_tpl = (cfg.get("templates") or {}).get(kind)
    if global_tpl:
        return global_tpl
    return None
