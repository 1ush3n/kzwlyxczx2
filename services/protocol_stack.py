from __future__ import annotations

import argparse
import multiprocessing as mp
import socket
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from core.comms.config import load_protocol_config, to_mapping
from services.opcua_gateway import run_opcua_gateway
from services.mapek_service import run_mapek_service
from services.virtual_plc import run_virtual_plc


class ProtocolStackSupervisor:
    """跨平台管理虚拟 PLC 与 OPC UA 网关子进程。"""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self._context = mp.get_context("spawn")
        self._processes: list[mp.Process] = []

    def start(self) -> None:
        plc_process = self._context.Process(
            target=run_virtual_plc,
            args=(self.config,),
            name="apal-virtual-plc",
        )
        plc_process.start()
        self._processes.append(plc_process)

        modbus = self.config["modbus"]
        self._wait_for_port(
            host=str(modbus["host"]),
            port=int(modbus["port"]),
            process=plc_process,
        )

        opcua_process = self._context.Process(
            target=run_opcua_gateway,
            args=(self.config,),
            name="apal-opcua-gateway",
        )
        opcua_process.start()
        self._processes.append(opcua_process)

        endpoint = urlparse(str(self.config["opcua"]["endpoint"]))
        self._wait_for_port(
            host=endpoint.hostname or "127.0.0.1",
            port=endpoint.port or 4840,
            process=opcua_process,
        )

        if bool(self.config.get("mapek", {}).get("enabled", False)):
            mapek_process = self._context.Process(
                target=run_mapek_service,
                args=(self.config,),
                name="apal-mapek-service",
            )
            mapek_process.start()
            self._processes.append(mapek_process)

    def wait(self) -> None:
        try:
            while all(process.is_alive() for process in self._processes):
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        for process in reversed(self._processes):
            if process.is_alive():
                process.terminate()
        for process in reversed(self._processes):
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        self._processes.clear()

    def _wait_for_port(self, host: str, port: int, process: mp.Process) -> None:
        timeout = float(self.config["runtime"]["startup_timeout_sec"])
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not process.is_alive():
                self.stop()
                raise RuntimeError(f"子进程 {process.name} 启动失败")
            try:
                with socket.create_connection((host, port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        self.stop()
        raise TimeoutError(f"等待服务端口超时: {host}:{port}")


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 APAL 工业协议栈")
    parser.add_argument("--config", type=Path, default=None, help="工业协议配置文件")
    args = parser.parse_args()
    config = to_mapping(load_protocol_config(args.config))
    supervisor = ProtocolStackSupervisor(config)
    supervisor.start()
    print("工业协议栈已就绪，按 Ctrl+C 停止。")
    supervisor.wait()


if __name__ == "__main__":
    mp.freeze_support()
    main()
