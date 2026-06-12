from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from asyncua import Server, ua, uamethod

from core.comms.config import load_protocol_config, load_yaml_mapping, to_mapping
from core.comms.models import PLCCommunicationError, PLCSnapshot, PLCStatus
from core.comms.plc_interface import ModbusTCPPLC
from core.network.tsn_runtime import TSNRuntimeController
from services.generate_certificates import ensure_certificates


UNIT_NAMESPACE_URI = "http://www.opcfoundation.org/UA/units/un/cefact"


class OPCUAGateway:
    """将 Modbus 现场数据、TSN 运行态和自治执行接口统一映射到 OPC UA。"""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.project_root = Path(str(config["runtime"]["project_root"]))
        agv_config_path = self.project_root / str(config["runtime"]["agv_config_path"])
        self.agv_config = load_yaml_mapping(agv_config_path)
        self.plc = ModbusTCPPLC(self.agv_config, config)
        self.tsn = TSNRuntimeController(config["tsn_runtime"], self.project_root)
        self.server = Server()
        self.namespace_index = 0
        self.nodes: dict[str, tuple[Any, ua.VariantType]] = {}
        self._requested_nodes: dict[str, Any] = {}
        self._control_lock = asyncio.Lock()
        self._last_snapshot = PLCSnapshot(
            md=float(self.agv_config["impedance"]["M_base"]),
            bd=float(self.agv_config["impedance"]["B_base"]),
            kd=float(self.agv_config["impedance"]["K_base"]),
            connected=False,
        )
        self._delay_fault_until = 0.0
        self._delay_fault_sec = 0.0
        self._communication_loss_until = 0.0

    async def setup(self) -> None:
        opcua = self.config["opcua"]
        await self.server.init()
        self.server.set_endpoint(str(opcua["endpoint"]))
        self.server.set_server_name(str(opcua["server_name"]))
        await self.server.set_application_uri(str(opcua["application_uri"]))
        await self._configure_security()
        self.namespace_index = await self.server.register_namespace(
            str(opcua["namespace_uri"])
        )
        await self._build_address_space()

    async def run(self) -> None:
        await self.setup()
        print(f"OPC UA 网关已启动: {self.config['opcua']['endpoint']}")
        try:
            async with self.server:
                while True:
                    await self.poll_once()
                    await asyncio.sleep(float(self.config["opcua"]["poll_interval_sec"]))
        finally:
            self.plc.close()

    async def poll_once(self) -> None:
        if self._communication_blocked():
            await self._publish_disconnected()
            return
        try:
            snapshot = await asyncio.to_thread(self.plc.read_snapshot)
            self._last_snapshot = snapshot
            await self._publish_snapshot(snapshot)
        except PLCCommunicationError:
            await self._publish_disconnected()

    async def _configure_security(self) -> None:
        mode = str(self.config["opcua"]["security_mode"]).lower()
        if mode == "development":
            self.server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
            return
        if mode != "secure":
            raise ValueError(f"不支持的 OPC UA 安全模式: {mode}")
        paths = ensure_certificates(self.config)
        await self.server.load_certificate(paths["server_certificate"])
        await self.server.load_private_key(paths["server_private_key"])
        self.server.set_security_policy(
            [ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt]
        )

    async def _build_address_space(self) -> None:
        idx = self.namespace_index
        root = await self.server.nodes.objects.add_object(
            ua.NodeId("APALCPS", idx),
            ua.QualifiedName("APALCPS", idx),
        )
        agv = await root.add_object(
            ua.NodeId("APALCPS.AGV1", idx),
            ua.QualifiedName("AGV1", idx),
        )
        names = (
            "Identification",
            "Motion",
            "Structure",
            "Network",
            "Control",
            "Diagnostics",
            "Autonomy",
            "FaultInjection",
        )
        folders = {
            name: await agv.add_object(
                ua.NodeId(f"APALCPS.AGV1.{name}", idx),
                ua.QualifiedName(name, idx),
            )
            for name in names
        }
        await self._add_variable(
            folders["Identification"], "AssetId", "AGV1", ua.VariantType.String
        )
        await self._add_variable(
            folders["Identification"],
            "Manufacturer",
            "APAL CPS Course Project",
            ua.VariantType.String,
        )
        await self._add_variable(
            folders["Identification"],
            "Protocols",
            "Modbus TCP; OPC UA; TSN",
            ua.VariantType.String,
        )
        for name, unit in (
            ("MasterPosition", "m"),
            ("MasterVelocity", "m/s"),
            ("SlavePosition", "m"),
            ("SlaveVelocity", "m/s"),
            ("TrackingError", "m"),
            ("TrackingErrorRate", "m/s"),
            ("SimulationTime", "s"),
        ):
            await self._add_variable(
                folders["Motion"], name, 0.0, ua.VariantType.Double, unit
            )
        await self._add_variable(
            folders["Structure"], "ExternalForce", 0.0, ua.VariantType.Double, "N"
        )
        for name, initial, variant, unit in (
            ("RTT", 0.0, ua.VariantType.Double, "s"),
            ("DeltaXCommand", 0.0, ua.VariantType.Double, "m"),
            ("AP1RSSI", -50.0, ua.VariantType.Double, "dBm"),
            ("AP2RSSI", -50.0, ua.VariantType.Double, "dBm"),
            ("ActiveAP", 1, ua.VariantType.UInt16, None),
            ("ActiveRoute", "", ua.VariantType.String, None),
            ("RouteSwitchCount", 0, ua.VariantType.UInt32, None),
            ("LinkOccupancy", 0.0, ua.VariantType.Double, "1"),
        ):
            await self._add_variable(
                folders["Network"], name, initial, variant, unit
            )
        for name, initial, unit in (
            ("ActualMd", self._last_snapshot.md, "kg"),
            ("ActualBd", self._last_snapshot.bd, "N·s/m"),
            ("ActualKd", self._last_snapshot.kd, "N/m"),
        ):
            await self._add_variable(
                folders["Control"], name, initial, ua.VariantType.Double, unit
            )
        for name, initial, unit in (
            ("RequestedMd", self._last_snapshot.md, "kg"),
            ("RequestedBd", self._last_snapshot.bd, "N·s/m"),
            ("RequestedKd", self._last_snapshot.kd, "N/m"),
            ("RequestedMasterVelocity", 1.5, "m/s"),
        ):
            node = await self._add_variable(
                folders["Control"],
                name,
                initial,
                ua.VariantType.Double,
                unit,
                writable=True,
            )
            self._requested_nodes[name] = node
        requested_ap = await self._add_variable(
            folders["Control"],
            "RequestedAP",
            1,
            ua.VariantType.UInt16,
            writable=True,
        )
        self._requested_nodes["RequestedAP"] = requested_ap
        reset_seed = await self._add_variable(
            folders["Control"],
            "ResetSeed",
            42,
            ua.VariantType.UInt16,
            writable=True,
        )
        self._requested_nodes["ResetSeed"] = reset_seed
        for name, initial, variant in (
            ("ModbusConnected", False, ua.VariantType.Boolean),
            ("StatusCode", 0, ua.VariantType.UInt16),
            ("AlarmBits", 0, ua.VariantType.UInt16),
            ("ControlMode", 0, ua.VariantType.UInt16),
            ("StepCount", 0, ua.VariantType.UInt32),
            ("LastUpdate", datetime.now(timezone.utc), ua.VariantType.DateTime),
        ):
            await self._add_variable(folders["Diagnostics"], name, initial, variant)
        for name, initial, variant in (
            ("MAPEKStatus", "starting", ua.VariantType.String),
            ("HealthState", "unknown", ua.VariantType.String),
            ("LastDecision", "none", ua.VariantType.String),
            ("LastDecisionReason", "", ua.VariantType.String),
            ("AutonomyHeartbeat", datetime.now(timezone.utc), ua.VariantType.DateTime),
        ):
            await self._add_variable(
                folders["Autonomy"], name, initial, variant, writable=True
            )
        for name, initial, variant in (
            ("FaultAP", 1, ua.VariantType.UInt16),
            ("APAttenuationDb", 60.0, ua.VariantType.Double),
            ("FaultDurationSec", 2.0, ua.VariantType.Double),
            ("DelaySpikeSec", 0.25, ua.VariantType.Double),
            ("CommunicationLossSec", 0.8, ua.VariantType.Double),
        ):
            node = await self._add_variable(
                folders["FaultInjection"],
                name,
                initial,
                variant,
                writable=True,
            )
            self._requested_nodes[name] = node

        await self._add_methods(folders)

    async def _add_methods(self, folders: Mapping[str, Any]) -> None:
        idx = self.namespace_index

        def method(name: str, function: Any, folder: str) -> Any:
            return folders[folder].add_method(
                ua.NodeId(f"APALCPS.AGV1.{folder}.{name}", idx),
                ua.QualifiedName(name, idx),
                function,
            )

        @uamethod
        async def apply_impedance(parent: ua.NodeId) -> ua.StatusCode:
            del parent
            return await self._apply_impedance()

        @uamethod
        async def release_manual_control(parent: ua.NodeId) -> ua.StatusCode:
            del parent
            return await self._plc_method(self.plc.release_manual_control)

        @uamethod
        async def safe_stop(parent: ua.NodeId) -> ua.StatusCode:
            del parent
            return await self._plc_method(self.plc.safe_stop)

        @uamethod
        async def reset_safety(parent: ua.NodeId) -> ua.StatusCode:
            del parent
            return await self._plc_method(self.plc.reset_safety)

        @uamethod
        async def reset_simulation(parent: ua.NodeId) -> ua.StatusCode:
            del parent
            seed = int(await self._requested_nodes["ResetSeed"].read_value())
            return await self._plc_method(self.plc.reset, seed)

        @uamethod
        async def step_simulation(parent: ua.NodeId) -> ua.StatusCode:
            del parent
            return await self._step_simulation()

        @uamethod
        async def apply_route(parent: ua.NodeId) -> ua.StatusCode:
            del parent
            ap = int(await self._requested_nodes["RequestedAP"].read_value())
            if not self.tsn.select_ap(ap):
                return ua.StatusCode(ua.StatusCodes.BadInvalidArgument)
            await self._publish_network()
            return ua.StatusCode(ua.StatusCodes.Good)

        @uamethod
        async def inject_ap_fault(parent: ua.NodeId) -> ua.StatusCode:
            del parent
            ap = int(await self._requested_nodes["FaultAP"].read_value())
            loss = float(await self._requested_nodes["APAttenuationDb"].read_value())
            duration = float(await self._requested_nodes["FaultDurationSec"].read_value())
            try:
                self.tsn.inject_ap_attenuation(ap, loss, duration)
                return ua.StatusCode(ua.StatusCodes.Good)
            except ValueError:
                return ua.StatusCode(ua.StatusCodes.BadInvalidArgument)

        @uamethod
        async def inject_delay(parent: ua.NodeId) -> ua.StatusCode:
            del parent
            self._delay_fault_sec = float(
                await self._requested_nodes["DelaySpikeSec"].read_value()
            )
            duration = float(await self._requested_nodes["FaultDurationSec"].read_value())
            self._delay_fault_until = time.monotonic() + duration
            return ua.StatusCode(ua.StatusCodes.Good)

        @uamethod
        async def inject_communication_loss(parent: ua.NodeId) -> ua.StatusCode:
            del parent
            duration = float(
                await self._requested_nodes["CommunicationLossSec"].read_value()
            )
            self._communication_loss_until = time.monotonic() + duration
            await self._publish_disconnected()
            return ua.StatusCode(ua.StatusCodes.Good)

        await method("ApplyImpedance", apply_impedance, "Control")
        await method("ReleaseManualControl", release_manual_control, "Control")
        await method("SafeStop", safe_stop, "Control")
        await method("ResetSafety", reset_safety, "Control")
        await method("ResetSimulation", reset_simulation, "Control")
        await method("StepSimulation", step_simulation, "Control")
        await method("ApplyRoute", apply_route, "Control")
        await method("InjectAPFault", inject_ap_fault, "FaultInjection")
        await method("InjectDelaySpike", inject_delay, "FaultInjection")
        await method(
            "InjectPLCCommunicationLoss",
            inject_communication_loss,
            "FaultInjection",
        )

    async def _add_variable(
        self,
        parent: Any,
        name: str,
        initial_value: Any,
        variant_type: ua.VariantType,
        unit: str | None = None,
        writable: bool = False,
    ) -> Any:
        parent_name = (await parent.read_browse_name()).Name
        node_id = ua.NodeId(
            f"APALCPS.AGV1.{parent_name}.{name}",
            self.namespace_index,
        )
        node = await parent.add_variable(
            node_id,
            ua.QualifiedName(name, self.namespace_index),
            ua.Variant(initial_value, variant_type),
        )
        if writable:
            await node.set_writable()
        if unit is not None:
            engineering_unit = ua.EUInformation(
                NamespaceUri=UNIT_NAMESPACE_URI,
                UnitId=0,
                DisplayName=ua.LocalizedText(unit),
                Description=ua.LocalizedText(unit),
            )
            await node.add_property(
                ua.NodeId(f"{node_id.Identifier}.EngineeringUnits", self.namespace_index),
                ua.QualifiedName("EngineeringUnits", 0),
                ua.Variant(engineering_unit, ua.VariantType.ExtensionObject),
            )
        self.nodes[name] = (node, variant_type)
        return node

    async def _publish_snapshot(self, snapshot: PLCSnapshot) -> None:
        values = {
            "MasterPosition": snapshot.master_position,
            "MasterVelocity": snapshot.master_velocity,
            "SlavePosition": snapshot.slave_position,
            "SlaveVelocity": snapshot.slave_velocity,
            "TrackingError": snapshot.error,
            "TrackingErrorRate": snapshot.error_rate,
            "SimulationTime": snapshot.simulation_time,
            "ExternalForce": snapshot.external_force,
            "RTT": snapshot.rtt_sec,
            "DeltaXCommand": snapshot.delta_x_cmd,
            "ActualMd": snapshot.md,
            "ActualBd": snapshot.bd,
            "ActualKd": snapshot.kd,
            "ModbusConnected": snapshot.connected,
            "StatusCode": int(snapshot.status),
            "AlarmBits": int(snapshot.alarm),
            "ControlMode": int(snapshot.control_mode),
            "StepCount": snapshot.step_count,
            "LastUpdate": snapshot.updated_at,
        }
        for name, value in values.items():
            await self._write_node(name, value)
        await self._publish_network()

    async def _publish_network(self) -> None:
        network = self.tsn.observe(self._last_snapshot.master_position)
        for name, value in {
            "AP1RSSI": network.ap1_rssi,
            "AP2RSSI": network.ap2_rssi,
            "ActiveAP": network.active_ap,
            "ActiveRoute": network.active_route,
            "RouteSwitchCount": network.route_switch_count,
            "LinkOccupancy": network.link_occupancy,
        }.items():
            await self._write_node(name, value)

    async def _publish_disconnected(self) -> None:
        await self._write_node("ModbusConnected", False)
        await self._write_node("StatusCode", int(PLCStatus.COMMUNICATION_ERROR))
        await self._publish_network()

    async def _write_node(self, name: str, value: Any) -> None:
        node, variant_type = self.nodes[name]
        await node.write_value(ua.Variant(value, variant_type))

    async def _apply_impedance(self) -> ua.StatusCode:
        md = float(await self._requested_nodes["RequestedMd"].read_value())
        bd = float(await self._requested_nodes["RequestedBd"].read_value())
        kd = float(await self._requested_nodes["RequestedKd"].read_value())
        if not self._impedance_is_valid(md, bd, kd):
            return ua.StatusCode(ua.StatusCodes.BadOutOfRange)
        return await self._plc_method(self.plc.apply_manual_impedance, md, bd, kd)

    async def _step_simulation(self) -> ua.StatusCode:
        velocity = float(
            await self._requested_nodes["RequestedMasterVelocity"].read_value()
        )
        if time.monotonic() < self._delay_fault_until:
            self.plc.inject_tsn_delay(self._delay_fault_sec)
        return await self._plc_method(self.plc.step_simulation, velocity)

    async def _plc_method(self, function: Any, *args: Any) -> ua.StatusCode:
        async with self._control_lock:
            if self._communication_blocked():
                return ua.StatusCode(ua.StatusCodes.BadCommunicationError)
            try:
                snapshot = await asyncio.to_thread(function, *args)
                self._last_snapshot = snapshot
                await self._publish_snapshot(snapshot)
                return ua.StatusCode(ua.StatusCodes.Good)
            except PLCCommunicationError:
                await self._publish_disconnected()
                return ua.StatusCode(ua.StatusCodes.BadCommunicationError)

    def _communication_blocked(self) -> bool:
        return time.monotonic() < self._communication_loss_until

    def _impedance_is_valid(self, md: float, bd: float, kd: float) -> bool:
        impedance = self.agv_config["impedance"]
        return all(
            lower <= value <= upper
            for value, lower, upper in (
                (
                    md,
                    max(float(impedance["M_base"]) - float(impedance["M_delta_max"]), 1e-4),
                    float(impedance["M_base"]) + float(impedance["M_delta_max"]),
                ),
                (
                    bd,
                    max(float(impedance["B_base"]) - float(impedance["B_delta_max"]), 1e-4),
                    float(impedance["B_base"]) + float(impedance["B_delta_max"]),
                ),
                (
                    kd,
                    max(float(impedance["K_base"]) - float(impedance["K_delta_max"]), 1e-4),
                    float(impedance["K_base"]) + float(impedance["K_delta_max"]),
                ),
            )
        )


def run_opcua_gateway(config: Mapping[str, Any] | None = None) -> None:
    resolved = to_mapping(load_protocol_config()) if config is None else dict(config)
    asyncio.run(OPCUAGateway(resolved).run())


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 APAL CPS OPC UA 网关")
    parser.add_argument("--config", type=Path, default=None, help="工业协议配置文件")
    args = parser.parse_args()
    run_opcua_gateway(to_mapping(load_protocol_config(args.config)))


if __name__ == "__main__":
    main()
