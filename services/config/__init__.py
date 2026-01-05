"""
配置模块（已并入 services.config）
"""

from .manager import ConfigManager, get_config, init_config
from .loader import (
    load_config,
    get_active_cluster,
    get_cluster,
    get_remote_server,
    get_remote_paths,
    get_queue_defaults,
    get_template_path,
)

__all__ = [
    "ConfigManager",
    "get_config",
    "init_config",
    "load_config",
    "get_active_cluster",
    "get_cluster",
    "get_remote_server",
    "get_remote_paths",
    "get_queue_defaults",
    "get_template_path",
]
