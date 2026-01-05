import logging
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from services.config.loader import (
    get_cluster,
    get_queue_defaults,
    get_remote_paths,
    load_config,
)
from services.common.templates import load_slurm_template
from services.remote_server.ssh_manager import SSHManager

logger = logging.getLogger(__name__)


def _build_remote_dir(cfg: Dict, module: str, remote_subdir: str = "", cluster_name: str = None) -> str:
    cluster = get_cluster(cfg, cluster_name)
    paths = get_remote_paths(cfg, cluster_name)
    work_base = paths.get("work_base") or cluster.get("remote_paths", {}).get("work_base", ".")
    subdir = remote_subdir.strip("/") if remote_subdir else "jobs"
    job_uid = uuid.uuid4().hex[:12]
    return f"{work_base}/{module}/{subdir}/{job_uid}"


def _ensure_connection(cfg: Dict, cluster_name: str = None) -> SSHManager:
    cluster = get_cluster(cfg, cluster_name)
    remote = cluster.get("remote_server") or cfg.get("remote_server") or {}
    mgr = SSHManager(
        hostname=remote.get("hostname", ""),
        port=int(remote.get("port", 22)),
        username=remote.get("username", ""),
        password=remote.get("password", ""),
    )
    ok, msg = mgr.connect()
    if not ok:
        raise RuntimeError(f"SSH connect failed: {msg}")
    mgr.open_sftp()
    return mgr


def _render_slurm(template: str, queue: Dict, job_name: str, command: str) -> str:
    safe = queue or {}
    return template.format(
        job_name=job_name,
        partition=safe.get("partition", ""),
        time_limit=safe.get("time_limit", "24:00:00"),
        nodes=safe.get("nodes", 1),
        ntasks_per_node=safe.get("ntasks_per_node", 1),
        gpus=safe.get("gpus", 0),
        command=command,
    )


def submit_job(
    module: str,
    command: str,
    files: List[Dict[str, str]] | None = None,
    slurm_overrides: Dict | None = None,
    remote_subdir: str = "",
    cluster_name: str | None = None,
) -> Tuple[str, str]:
    """
    Submit a job to remote HPC via SSH + SLURM.

    Returns: (job_id, remote_dir)
    """
    cfg = load_config()
    queue_cfg = slurm_overrides or get_queue_defaults(cfg, cluster_name)

    try:
        template = load_slurm_template(cluster_name=cluster_name)
    except FileNotFoundError:
        template = """#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p {partition}
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH -t {time_limit}

module purge
{command}
"""

    mgr = _ensure_connection(cfg, cluster_name)
    remote_dir = _build_remote_dir(cfg, module, remote_subdir, cluster_name)
    ok, msg = mgr.mkdir_remote(remote_dir)
    if not ok:
        mgr.close()
        raise RuntimeError(msg)

    if files:
        for f in files:
            fname = f.get("name")
            content = f.get("content", "")
            if not fname:
                continue
            mgr.write_remote_file(f"{remote_dir}/{fname}", content.replace("\r\n", "\n"))

    has_slurm = any((f.get("name") or "").lower() == "slurm.sh" for f in (files or []))
    if not has_slurm:
        slurm_content = _render_slurm(template, queue_cfg, job_name=module, command=command)
        mgr.write_remote_file(f"{remote_dir}/slurm.sh", slurm_content)

    success, job_id = mgr.submit_job_slurm(remote_dir, script_name="slurm.sh")
    mgr.close()
    if not success:
        raise RuntimeError(job_id)
    return job_id, remote_dir


def fetch_status(job_id: str, cluster_name: str | None = None) -> str:
    cfg = load_config()
    mgr = _ensure_connection(cfg, cluster_name)
    exists, status = mgr.query_slurm_status(job_id)
    mgr.close()
    if exists:
        return "RUNNING"
    return "COMPLETED"
