from __future__ import annotations

import argparse
import asyncio
import copy
import time
from pathlib import Path
from typing import Any

from asyncua import Client, ua

from core.comms.config import load_protocol_config, to_mapping
from core.comms.models import ControlMode
from services.autonomy_report import generate_report
from services.generate_certificates import ensure_certificates
from services.protocol_stack import ProtocolStackSupervisor


class AutonomousCPSDemo:
    """通过 OPC UA 驱动三种故障场景并验证 MAPE-K 响应。"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.client = Client(str(config["opcua"]["endpoint"]))
        self.namespace = 0

    async def run(self) -> None:
        await self._configure_security()
        await self.client.connect()
        try:
            self.namespace = await self.client.get_namespace_index(
                str(self.config["opcua"]["namespace_uri"])
            )
            await self._wait_for_mapek()
            await self._reset()
            await self._drive(15, velocity=0.35)
            await self._scenario_ap_attenuation()
            await self._scenario_rtt_spike()
            await self._scenario_plc_disconnect()
        finally:
            await self.client.disconnect()

    async def _scenario_ap_attenuation(self) -> None:
        initial_ap = int(await self._read("Network.ActiveAP"))
        await self._write("FaultInjection.FaultAP", initial_ap, ua.VariantType.UInt16)
        await self._write(
            "FaultInjection.APAttenuationDb",
            60.0,
            ua.VariantType.Double,
        )
        await self._write(
            "FaultInjection.FaultDurationSec",
            2.0,
            ua.VariantType.Double,
        )
        await self._call("FaultInjection", "InjectAPFault")
        started = time.monotonic()
        await self._drive_until(
            lambda: self._read("Network.ActiveAP"),
            lambda value: int(value) != initial_ap,
            timeout=2.0,
            velocity=0.35,
        )
        switched = int(await self._read("Network.ActiveAP"))
        print(
            f"[AP 衰减] AP{initial_ap} -> AP{switched}，"
            f"恢复时间 {time.monotonic()-started:.3f} s"
        )

    async def _scenario_rtt_spike(self) -> None:
        before_bd = float(await self._read("Control.ActualBd"))
        before_kd = float(await self._read("Control.ActualKd"))
        await self._write(
            "FaultInjection.DelaySpikeSec",
            0.25,
            ua.VariantType.Double,
        )
        await self._write(
            "FaultInjection.FaultDurationSec",
            1.4,
            ua.VariantType.Double,
        )
        await self._call("FaultInjection", "InjectDelaySpike")
        await self._drive_until(
            lambda: self._read("Control.ActualKd"),
            lambda value: float(value) < before_kd - 1.0,
            timeout=2.5,
            velocity=0.35,
        )
        after_bd = float(await self._read("Control.ActualBd"))
        after_kd = float(await self._read("Control.ActualKd"))
        print(
            f"[RTT 突增] Bd {before_bd:.1f}->{after_bd:.1f}，"
            f"Kd {before_kd:.1f}->{after_kd:.1f}"
        )
        await self._drive(35, velocity=0.35)

    async def _scenario_plc_disconnect(self) -> None:
        before = (
            float(await self._read("Control.ActualMd")),
            float(await self._read("Control.ActualBd")),
            float(await self._read("Control.ActualKd")),
        )
        await self._write(
            "FaultInjection.CommunicationLossSec",
            0.8,
            ua.VariantType.Double,
        )
        await self._call("FaultInjection", "InjectPLCCommunicationLoss")
        deadline = time.monotonic() + 1.1
        while time.monotonic() < deadline:
            try:
                await self._call("Control", "StepSimulation")
            except Exception:
                pass
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.2)
        mode = int(await self._read("Diagnostics.ControlMode"))
        velocity = float(await self._read("Motion.MasterVelocity"))
        after = (
            float(await self._read("Control.ActualMd")),
            float(await self._read("Control.ActualBd")),
            float(await self._read("Control.ActualKd")),
        )
        assert mode == int(ControlMode.SAFETY_STOP)
        assert abs(velocity) < 1e-6
        assert max(abs(a - b) for a, b in zip(before, after, strict=True)) < 1e-3
        print("[PLC 断链] 本地看门狗已锁存停车，最后安全阻抗保持不变")

        await self._call("Control", "StepSimulation")
        assert int(await self._read("Diagnostics.ControlMode")) == int(
            ControlMode.SAFETY_STOP
        )
        await self._call("Control", "ResetSafety")
        await self._call("Control", "StepSimulation")
        assert int(await self._read("Diagnostics.ControlMode")) != int(
            ControlMode.SAFETY_STOP
        )
        print("[PLC 恢复] 未自动启动；显式 ResetSafety 后恢复运行")

    async def _reset(self) -> None:
        await self._write("Control.ResetSeed", 42, ua.VariantType.UInt16)
        await self._write(
            "Control.RequestedMasterVelocity",
            0.35,
            ua.VariantType.Double,
        )
        await self._call("Control", "ResetSimulation")

    async def _drive(self, cycles: int, velocity: float) -> None:
        await self._write(
            "Control.RequestedMasterVelocity",
            velocity,
            ua.VariantType.Double,
        )
        for _ in range(cycles):
            await self._call("Control", "StepSimulation")
            await asyncio.sleep(0.05)

    async def _drive_until(
        self,
        read_value: Any,
        condition: Any,
        timeout: float,
        velocity: float,
    ) -> Any:
        await self._write(
            "Control.RequestedMasterVelocity",
            velocity,
            ua.VariantType.Double,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await self._call("Control", "StepSimulation")
            value = await read_value()
            if condition(value):
                return value
            await asyncio.sleep(0.05)
        raise TimeoutError("等待 MAPE-K 自主响应超时")

    async def _wait_for_mapek(self) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if str(await self._read("Autonomy.MAPEKStatus")) == "running":
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("MAPE-K 服务未进入运行状态")

    async def _configure_security(self) -> None:
        if str(self.config["opcua"]["security_mode"]).lower() != "secure":
            return
        paths = ensure_certificates(self.config)
        await self.client.set_security_string(
            "Basic256Sha256,SignAndEncrypt,"
            f"{paths['client_certificate']},{paths['client_private_key']},"
            f"{paths['server_certificate']}"
        )

    async def _read(self, suffix: str) -> Any:
        return await self._node(suffix).read_value()

    async def _write(
        self,
        suffix: str,
        value: Any,
        variant_type: ua.VariantType,
    ) -> None:
        await self._node(suffix).write_value(ua.Variant(value, variant_type))

    async def _call(self, folder: str, method: str) -> Any:
        parent = self._node(folder)
        method_node = self._node(f"{folder}.{method}")
        return await parent.call_method(method_node)

    def _node(self, suffix: str) -> Any:
        return self.client.get_node(
            ua.NodeId(f"APALCPS.AGV1.{suffix}", self.namespace)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行数字孪生与 MAPE-K 故障演示")
    parser.add_argument("--config", type=Path, default=None, help="工业协议配置文件")
    args = parser.parse_args()
    config = copy.deepcopy(to_mapping(load_protocol_config(args.config)))
    config["mapek"]["enabled"] = True
    project_root = Path(str(config["runtime"]["project_root"]))
    database_path = project_root / str(config["mapek"]["database_path"])
    if database_path.exists():
        database_path.unlink()
    supervisor = ProtocolStackSupervisor(config)
    supervisor.start()
    try:
        asyncio.run(AutonomousCPSDemo(config).run())
    finally:
        supervisor.stop()
    report = generate_report(
        database_path,
        project_root / str(config["mapek"]["export_directory"]),
    )
    print(f"演示完成，报告: {report}")


if __name__ == "__main__":
    main()
