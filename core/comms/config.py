from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_CONFIG = PROJECT_ROOT / "config" / "industrial_protocols.yaml"


def load_protocol_config(path: str | Path | None = None) -> DictConfig:
    """加载并解析工业协议配置。"""

    config_path = Path(path).resolve() if path else DEFAULT_PROTOCOL_CONFIG
    config = OmegaConf.load(config_path)
    config.runtime.project_root = str(PROJECT_ROOT)
    return config


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """使用 OmegaConf 加载普通 YAML 为原生字典。"""

    container = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"配置根节点必须为映射: {path}")
    return container


def to_mapping(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """将 OmegaConf 配置转换为可跨进程序列化的字典。"""

    if isinstance(config, DictConfig):
        container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("协议配置根节点必须为映射")
        return container
    return config

