from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from asyncua import Client, ua

from core.autonomy.mapek import (
    AdaptationAction,
    AdaptationPlan,
    MAPEKAnalyzer,
    MAPEKPlanner,
)
from core.comms.config import load_protocol_config, load_yaml_mapping, to_mapping
from core.twin.aas import AASExporter
from core.twin.builder import TwinStateBuilder
from core.twin.models import FaultType
from core.twin.repository import SQLiteTwinRepository
from services.generate_certificates import ensure_certificates


MONITOR_NODE_IDS = {
    "MasterPosition": "Motion.MasterPosition",
    "MasterVelocity": "Motion.MasterVelocity",
    "SlavePosition": "Motion.SlavePosition",
    "SlaveVelocity": "Motion.SlaveVelocity",
    "TrackingError": "Motion.TrackingError",
    "ExternalForce": "Structure.ExternalForce",
    "RTT": "Network.RTT",
    "AP1RSSI": "Network.AP1RSSI",
    "AP2RSSI": "Network.AP2RSSI",
    "ActiveAP": "Network.ActiveAP",
    "ActiveRoute": "Network.ActiveRoute",
    "LinkOccupancy": "Network.LinkOccupancy",
    "ActualMd": "Control.ActualMd",
    "ActualBd": "Control.ActualBd",
    "ActualKd": "Control.ActualKd",
    "ModbusConnected": "Diagnostics.ModbusConnected",
    "AlarmBits": "Diagnostics.AlarmBits",
    "ControlMode": "Diagnostics.ControlMode",
    "LastUpdate": "Diagnostics.LastUpdate",
}


class MAPEKExecutor:
    """通过 OPC UA 方法执行 Plan，不直接依赖 Modbus 实现。"""

    def __init__(self, client: Client, namespace: int):
        self._client = client
        self._namespace = namespace
        self._control = self._node("Control")
        self._requested = {
            name: self._node(f"Control.{name}")
            for name in ("RequestedMd", "RequestedBd", "RequestedKd", "RequestedAP")
        }
        self._methods = {
            name: self._node(f"Control.{name}")
            for name in (
                "ApplyImpedance",
                "ApplyRoute",
                "SafeStop",
                "ResetSafety",
            )
        }

    async def execute(self, plan: AdaptationPlan) -> tuple[bool, str, str]:
        try:
            if plan.action in (
                AdaptationAction.ADAPT_IMPEDANCE,
                AdaptationAction.RESTORE_IMPEDANCE,
            ):
                assert plan.target_impedance is not None
                md, bd, kd = plan.target_impedance
                await self._requested["RequestedMd"].write_value(
                    ua.Variant(md, ua.VariantType.Double)
                )
                await self._requested["RequestedBd"].write_value(
                    ua.Variant(bd, ua.VariantType.Double)
                )
                await self._requested["RequestedKd"].write_value(
                    ua.Variant(kd, ua.VariantType.Double)
                )
                await self._control.call_method(self._methods["ApplyImpedance"])
            elif plan.action is AdaptationAction.REROUTE:
                assert plan.target_ap is not None
                await self._requested["RequestedAP"].write_value(
                    ua.Variant(plan.target_ap, ua.VariantType.UInt16)
                )
                await self._control.call_method(self._methods["ApplyRoute"])
            elif plan.action is AdaptationAction.SAFE_STOP:
                await self._control.call_method(self._methods["SafeStop"])
            elif plan.action in (
                AdaptationAction.NONE,
                AdaptationAction.MAINTENANCE_ALERT,
            ):
                return True, "Good", "无需调用执行器"
            return True, "Good", plan.reason
        except Exception as exc:
            return False, type(exc).__name__, str(exc)

    def _node(self, suffix: str) -> Any:
        return self._client.get_node(
            ua.NodeId(f"APALCPS.AGV1.{suffix}", self._namespace)
        )


class MAPEKService:
    """独立运行的 MAPE-K 自主闭环服务。"""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.project_root = Path(str(config["runtime"]["project_root"]))
        self.mapek_config = config["mapek"]
        agv_path = self.project_root / str(config["runtime"]["agv_config_path"])
        self.agv_config = load_yaml_mapping(agv_path)
        self.repository = SQLiteTwinRepository(
            self.project_root / str(self.mapek_config["database_path"])
        )
        self.builder = TwinStateBuilder(self.agv_config)
        self.analyzer = MAPEKAnalyzer(self.mapek_config["thresholds"])
        self.planner = MAPEKPlanner(self.mapek_config, self.agv_config["impedance"])
        self.aas = AASExporter(config["aas"])
        self._last_faults: set[FaultType] = set()
        self._last_execution: dict[AdaptationAction, float] = {}
        self._last_aas_update = 0.0

    async def run(self) -> None:
        retry_sec = 0.5
        try:
            while True:
                client = Client(str(self.config["opcua"]["endpoint"]))
                await self._configure_client_security(client)
                try:
                    await client.connect()
                    await self._run_connected(client)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.repository.record_fault(
                        FaultType.PLC_DISCONNECTED.value,
                        "service_reconnect",
                        str(exc),
                    )
                    await asyncio.sleep(retry_sec)
                finally:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
        finally:
            self.repository.close()

    async def _run_connected(self, client: Client) -> None:
        namespace = await client.get_namespace_index(
            str(self.config["opcua"]["namespace_uri"])
        )
        nodes = {
            name: client.get_node(
                ua.NodeId(f"APALCPS.AGV1.{suffix}", namespace)
            )
            for name, suffix in MONITOR_NODE_IDS.items()
        }
        autonomy = {
            name: client.get_node(
                ua.NodeId(f"APALCPS.AGV1.Autonomy.{name}", namespace)
            )
            for name in (
                "MAPEKStatus",
                "HealthState",
                "LastDecision",
                "LastDecisionReason",
                "AutonomyHeartbeat",
            )
        }
        executor = MAPEKExecutor(client, namespace)
        await autonomy["MAPEKStatus"].write_value("running")
        cycle = float(self.mapek_config["cycle_sec"])
        while True:
            started = time.monotonic()
            raw_values = await client.read_values(list(nodes.values()))
            values = dict(zip(nodes, raw_values, strict=True))
            state = self.builder.build(values)
            analysis = self.analyzer.analyze(state)
            plan = self.planner.plan(state, analysis)
            self.repository.save_state(state)
            self._record_fault_transitions(set(analysis.active_faults))

            if self._should_execute(plan):
                decision_id = self.repository.record_decision(
                    plan.action.value,
                    analysis.primary_fault.value,
                    plan.reason,
                )
                success, status, detail = await executor.execute(plan)
                self.repository.record_execution(
                    decision_id,
                    success,
                    status,
                    detail,
                )
                self._last_execution[plan.action] = time.monotonic()

            await autonomy["HealthState"].write_value(
                analysis.primary_fault.value
            )
            await autonomy["LastDecision"].write_value(plan.action.value)
            await autonomy["LastDecisionReason"].write_value(plan.reason)
            await autonomy["AutonomyHeartbeat"].write_value(
                ua.Variant(datetime.now(timezone.utc), ua.VariantType.DateTime)
            )
            if time.monotonic() - self._last_aas_update >= float(
                self.mapek_config["aas_update_sec"]
            ):
                self.aas.export(
                    state,
                    self.project_root / str(self.mapek_config["aas_path"]),
                )
                self._last_aas_update = time.monotonic()
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.0, cycle - elapsed))

    def _should_execute(self, plan: AdaptationPlan) -> bool:
        if plan.action in (
            AdaptationAction.NONE,
            AdaptationAction.MAINTENANCE_ALERT,
        ):
            return plan.action is AdaptationAction.MAINTENANCE_ALERT and (
                time.monotonic()
                - self._last_execution.get(plan.action, 0.0)
                >= 5.0
            )
        minimum_interval = {
            AdaptationAction.SAFE_STOP: 1.0,
            AdaptationAction.REROUTE: 1.0,
            AdaptationAction.ADAPT_IMPEDANCE: 0.5,
            AdaptationAction.RESTORE_IMPEDANCE: 0.5,
        }[plan.action]
        return (
            time.monotonic()
            - self._last_execution.get(plan.action, 0.0)
            >= minimum_interval
        )

    def _record_fault_transitions(self, current: set[FaultType]) -> None:
        for fault in current - self._last_faults:
            self.repository.record_fault(fault.value, "start", "故障条件成立")
        for fault in self._last_faults - current:
            self.repository.record_fault(fault.value, "recovered", "恢复条件成立")
        self._last_faults = current

    async def _configure_client_security(self, client: Client) -> None:
        if str(self.config["opcua"]["security_mode"]).lower() != "secure":
            return
        paths = ensure_certificates(self.config)
        await client.set_security_string(
            "Basic256Sha256,SignAndEncrypt,"
            f"{paths['client_certificate']},{paths['client_private_key']},"
            f"{paths['server_certificate']}"
        )


def run_mapek_service(config: Mapping[str, Any] | None = None) -> None:
    resolved = to_mapping(load_protocol_config()) if config is None else dict(config)
    asyncio.run(MAPEKService(resolved).run())


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 APAL CPS MAPE-K 自主闭环")
    parser.add_argument("--config", type=Path, default=None, help="工业协议配置文件")
    args = parser.parse_args()
    run_mapek_service(to_mapping(load_protocol_config(args.config)))


if __name__ == "__main__":
    main()
