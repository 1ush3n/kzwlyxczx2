from __future__ import annotations

import copy
import socket
import time
from pathlib import Path
from typing import Any

from core.comms.config import load_protocol_config, to_mapping


def find_free_port() -> int:
    """获取仅供当前测试使用的本机空闲端口。"""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_test_config(
    tmp_path: Path,
    secure: bool = False,
) -> dict[str, Any]:
    """构造使用动态端口的隔离协议配置。"""

    config = copy.deepcopy(to_mapping(load_protocol_config()))
    modbus_port = find_free_port()
    opcua_port = find_free_port()
    config["modbus"]["port"] = modbus_port
    config["opcua"]["endpoint"] = (
        f"opc.tcp://127.0.0.1:{opcua_port}/apal/cps/"
    )
    config["opcua"]["security_mode"] = "secure" if secure else "development"
    config["opcua"]["certificates"]["directory"] = str(tmp_path / "certs")
    config["runtime"]["startup_timeout_sec"] = 10.0
    return config


def wait_for_port(
    host: str,
    port: int,
    process: Any,
    timeout: float = 10.0,
) -> None:
    """等待子进程开始监听端口。"""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError(f"服务进程提前退出，exitcode={process.exitcode}")
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"等待端口超时: {host}:{port}")


def stop_process(process: Any) -> None:
    """终止测试创建的服务进程。"""

    if process.is_alive():
        process.terminate()
    process.join(timeout=5.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=2.0)

