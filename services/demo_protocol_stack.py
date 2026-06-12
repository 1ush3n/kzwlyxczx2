from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from asyncua import Client, ua

from core.comms.config import PROJECT_ROOT, load_protocol_config, load_yaml_mapping, to_mapping
from core.comms.plc_interface import ModbusTCPPLC
from services.generate_certificates import ensure_certificates
from services.protocol_stack import ProtocolStackSupervisor


class DemoSubscriptionHandler:
    """打印 OPC UA 实时数据变化。"""

    def datachange_notification(self, node, value, data) -> None:
        del data
        print(f"订阅更新: {node.nodeid} = {value}")


async def run_demo(config: dict) -> None:
    """执行复位、订阅、手动阻抗和自动恢复演示。"""

    client = Client(str(config["opcua"]["endpoint"]))
    if str(config["opcua"]["security_mode"]).lower() == "secure":
        paths = ensure_certificates(config)
        await client.set_security_string(
            "Basic256Sha256,SignAndEncrypt,"
            f"{paths['client_certificate']},{paths['client_private_key']},"
            f"{paths['server_certificate']}"
        )

    agv_config = load_yaml_mapping(PROJECT_ROOT / "config" / "agv_env_config.yaml")
    plc = ModbusTCPPLC(agv_config, config)
    await client.connect()
    try:
        namespace = await client.get_namespace_index(
            str(config["opcua"]["namespace_uri"])
        )
        control = client.get_node(ua.NodeId("APALCPS.AGV1.Control", namespace))
        requested_md = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.RequestedMd", namespace)
        )
        requested_bd = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.RequestedBd", namespace)
        )
        requested_kd = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.RequestedKd", namespace)
        )
        apply_method = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.ApplyImpedance", namespace)
        )
        release_method = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.ReleaseManualControl", namespace)
        )
        force = client.get_node(
            ua.NodeId("APALCPS.AGV1.Structure.ExternalForce", namespace)
        )
        actual_md = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.ActualMd", namespace)
        )

        subscription = await client.create_subscription(
            50,
            DemoSubscriptionHandler(),
        )
        handle = await subscription.subscribe_data_change(force)

        await asyncio.to_thread(plc.reset, 42)
        for _ in range(5):
            plc.inject_tsn_delay(0.02)
            await asyncio.to_thread(plc.step_simulation, 1.5)
            await asyncio.sleep(0.1)

        await requested_md.write_value(ua.Variant(55.0, ua.VariantType.Double))
        await requested_bd.write_value(ua.Variant(600.0, ua.VariantType.Double))
        await requested_kd.write_value(ua.Variant(2500.0, ua.VariantType.Double))
        await control.call_method(apply_method)
        await asyncio.sleep(0.2)
        print(f"人工阻抗已生效: Md={await actual_md.read_value():.2f}")

        await control.call_method(release_method)
        await asyncio.sleep(0.2)
        print("已释放人工控制，PLC 返回自动模式。")

        await subscription.unsubscribe(handle)
        await subscription.delete()
    finally:
        plc.close()
        await client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="运行工业协议闭环演示")
    parser.add_argument("--config", type=Path, default=None, help="工业协议配置文件")
    args = parser.parse_args()
    config = to_mapping(load_protocol_config(args.config))
    config["mapek"]["enabled"] = False
    supervisor = ProtocolStackSupervisor(config)
    supervisor.start()
    try:
        asyncio.run(run_demo(config))
    finally:
        supervisor.stop()


if __name__ == "__main__":
    main()
