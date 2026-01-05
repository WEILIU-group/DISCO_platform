"""
全局配置管理
处理服务器连接、路径、数据库等配置
"""

import os
from typing import Dict, Optional
import json


class ConfigManager:
    """全局配置管理器"""

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_file: 配置文件路径（JSON 格式）
        """
        self.config = self._get_default_config()

        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            # SSH 服务器配置（全局回退）
            "remote_server": {
                "hostname": "10.10.69.102",
                "port": 22,
                "username": "zyy",
                "password": "zyy249720"
            },

            # 本地路径配置
            "local_paths": {
                "data_dir": "./data",
                "results_dir": "./results",
                "temp_dir": "./temp",
                "elements_csv": "elements_properties_all.csv"
            },

            # 远程服务器工作路径（全局回退）
            "remote_paths": {
                "work_base": ".",
                "sisso_base": "SISSO_JOBS",
                "vasp_base": "VASP_JOBS"
            },

            # SISSO 默认参数
            "sisso_defaults": {
                "desc_dim": 2,
                "fcomplexity": 3,
                "ops": "(+)(-)(*)(/)",
                "nmodel": 100,
                "method_so": "L0"
            },

            # VASP 默认参数
            "vasp_defaults": {
                "encut": 500,
                "nsw": 100,
                "ibrion": 2,
                "potim": 0.5,
                "ediff": 1e-5,
                "prec": "High",
                "nelm": 100,
                "ismear": 0,
                "sigma": 0.05,
                "kmesh": [4, 4, 4]
            },

            # 队列系统配置（全局回退）
            "queue_system": {
                "type": "slurm",  # slurm, pbs, sge
                "poll_interval": 5,
                "max_wait_time": 1800
            },

            # 集群定义（便于多集群切换）
            "active_cluster": "default",
            "clusters": {
                "default": {
                    "remote_server": {
                        "hostname": "10.10.69.102",
                        "port": 22,
                        "username": "zyy",
                        "password": "zyy249720"
                    },
                    "queue": {
                        "type": "slurm",
                        "partition": "vasp",
                        "time_limit": "24:00:00",
                        "nodes": 1,
                        "ntasks_per_node": 30,
                        "poll_interval": 5,
                        "max_wait_time": 1800
                    },
                    "remote_paths": {
                        "work_base": ".",
                        "sisso_base": "SISSO_JOBS",
                        "vasp_base": "VASP_JOBS"
                    }
                },
                "aliyun_root": {
                    "remote_server": {
                        "hostname": "120.55.2.205",
                        "port": 22,
                        "username": "root",
                        "password": "zyy249720"
                    },
                    "queue": {
                        "type": "slurm",
                        "partition": "vasp",
                        "time_limit": "24:00:00",
                        "nodes": 1,
                        "ntasks_per_node": 30,
                        "poll_interval": 5,
                        "max_wait_time": 1800
                    },
                    "remote_paths": {
                        "work_base": ".",
                        "sisso_base": "SISSO_JOBS",
                        "vasp_base": "VASP_JOBS"
                    }
                }
            },

            # 应用配置
            "app": {
                "debug": False,
                "port": 8050,
                "host": "127.0.0.1"
            }
        }

    def get(self, key: str, default=None):
        """
        获取配置值（支持点号路径）

        Args:
            key: 配置键（如 'remote_server.hostname' 或 'sisso_defaults'）
            default: 默认值

        Returns:
            配置值
        """
        if "." not in key:
            return self.config.get(key, default)

        parts = key.split(".")
        value = self.config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def set(self, key: str, value):
        """
        设置配置值（支持点号路径）

        Args:
            key: 配置键
            value: 新值
        """
        if "." not in key:
            self.config[key] = value
            return

        parts = key.split(".")
        current = self.config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def load_from_file(self, config_file: str):
        """从 JSON 文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 深度更新
                self._deep_update(self.config, data)
        except Exception as e:
            print(f"警告: 加载配置文件失败 {config_file}: {e}")

    def save_to_file(self, config_file: str):
        """将配置保存到 JSON 文件"""
        try:
            os.makedirs(os.path.dirname(config_file) or ".", exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"警告: 保存配置文件失败 {config_file}: {e}")

    @staticmethod
    def _deep_update(target: Dict, source: Dict):
        """深度更新字典"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                ConfigManager._deep_update(target[key], value)
            else:
                target[key] = value

    def to_dict(self) -> Dict:
        """返回完整配置字典"""
        return self.config.copy()

    def __repr__(self) -> str:
        return json.dumps(self.config, indent=2, ensure_ascii=False)


# 全局配置实例
_global_config = None


def get_config() -> ConfigManager:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config


def init_config(config_file: Optional[str] = None) -> ConfigManager:
    """初始化全局配置"""
    global _global_config
    _global_config = ConfigManager(config_file)
    return _global_config
