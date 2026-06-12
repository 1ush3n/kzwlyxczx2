from __future__ import annotations

import asyncio
import multiprocessing as mp
from pathlib import Path
from urllib.parse import urlparse

import pytest
from asyncua import Client, ua

from core.comms.config import PROJECT_ROOT, load_yaml_mapping
from core.comms.plc_interface import ModbusTCPPLC
from services.generate_certificates import ensure_certificates
from services.opcua_gateway import run_opcua_gateway
from services.virtual_plc import run_virtual_plc
from tests.protocol_test_utils import (
    build_test_config,
    stop_process,
    wait_for_port,
)


class DataChangeHandler:
    """记录 OPC UA 订阅回调。"""

    def __init__(self):
        self.values: list[float] = []

    def datachange_notification(self, node, value, data) -> None:
        del node, data
        self.values.append(float(value))


@pytest.fixture
def opcua_services(tmp_path: Path):
    config = build_test_config(tmp_path)
    context = mp.get_context("spawn")
    plc_process = context.Process(target=run_virtual_plc, args=(config,))
    plc_process.start()
    wait_for_port(
        "127.0.0.1",
        int(config["modbus"]["port"]),
        plc_process,
    )

    opcua_process = context.Process(target=run_opcua_gateway, args=(config,))
    opcua_process.start()
    endpoint = urlparse(str(config["opcua"]["endpoint"]))
    wait_for_port(
        endpoint.hostname or "127.0.0.1",
        endpoint.port or 4840,
        opcua_process,
    )
    try:
        yield config
    finally:
        stop_process(opcua_process)
        stop_process(plc_process)


def test_opcua_browse_subscribe_and_control(opcua_services: dict) -> None:
    asyncio.run(_exercise_opcua_gateway(opcua_services))


async def _exercise_opcua_gateway(config: dict) -> None:
    endpoint = str(config["opcua"]["endpoint"])
    client = Client(endpoint)
    await client.connect()
    modbus = ModbusTCPPLC(
        load_yaml_mapping(PROJECT_ROOT / "config" / "agv_env_config.yaml"),
        config,
    )
    try:
        namespace = await client.get_namespace_index(
            str(config["opcua"]["namespace_uri"])
        )
        actual_md = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.ActualMd", namespace)
        )
        requested_md = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.RequestedMd", namespace)
        )
        requested_bd = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.RequestedBd", namespace)
        )
        requested_kd = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.RequestedKd", namespace)
        )
        control = client.get_node(ua.NodeId("APALCPS.AGV1.Control", namespace))
        apply_method = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.ApplyImpedance", namespace)
        )
        release_method = client.get_node(
            ua.NodeId("APALCPS.AGV1.Control.ReleaseManualControl", namespace)
        )
        force = client.get_node(
            ua.NodeId("APALCPS.AGV1.Structure.ExternalForce", namespace)
        )
        control_mode = client.get_node(
            ua.NodeId("APALCPS.AGV1.Diagnostics.ControlMode", namespace)
        )

        engineering_units = await actual_md.get_child(
            [ua.QualifiedName("EngineeringUnits", 0)]
        )
        unit_info = await engineering_units.read_value()
        assert unit_info.DisplayName.Text == "kg"

        await requested_md.write_value(ua.Variant(55.0, ua.VariantType.Double))
        await requested_bd.write_value(ua.Variant(600.0, ua.VariantType.Double))
        await requested_kd.write_value(ua.Variant(2500.0, ua.VariantType.Double))
        assert await requested_md.read_value() == pytest.approx(55.0)
        assert await requested_bd.read_value() == pytest.approx(600.0)
        assert await requested_kd.read_value() == pytest.approx(2500.0)
        await control.call_method(apply_method)
        await asyncio.sleep(0.15)
        assert await control_mode.read_value() == 1
        assert await actual_md.read_value() == pytest.approx(55.0)

        handler = DataChangeHandler()
        subscription = await client.create_subscription(25, handler)
        handle = await subscription.subscribe_data_change(force)
        modbus.reset(seed=42)
        modbus.inject_tsn_delay(0.02)
        modbus.step_simulation(1.5)
        await asyncio.sleep(0.2)
        assert handler.values
        await subscription.unsubscribe(handle)
        await subscription.delete()

        await requested_md.write_value(ua.Variant(1000.0, ua.VariantType.Double))
        with pytest.raises(ua.UaStatusCodeError):
            await control.call_method(apply_method)

        await requested_md.write_value(ua.Variant(50.0, ua.VariantType.Double))
        await control.call_method(release_method)
    finally:
        modbus.close()
        await client.disconnect()


def test_secure_endpoint_requires_certificate(tmp_path: Path) -> None:
    config = build_test_config(tmp_path, secure=True)
    ensure_certificates(config, force=True)
    context = mp.get_context("spawn")
    plc_process = context.Process(target=run_virtual_plc, args=(config,))
    opcua_process = context.Process(target=run_opcua_gateway, args=(config,))
    plc_process.start()
    wait_for_port(
        "127.0.0.1",
        int(config["modbus"]["port"]),
        plc_process,
    )
    opcua_process.start()
    endpoint = urlparse(str(config["opcua"]["endpoint"]))
    wait_for_port(
        endpoint.hostname or "127.0.0.1",
        endpoint.port or 4840,
        opcua_process,
    )
    try:
        asyncio.run(_verify_secure_endpoint(config))
    finally:
        stop_process(opcua_process)
        stop_process(plc_process)


async def _verify_secure_endpoint(config: dict) -> None:
    endpoint = str(config["opcua"]["endpoint"])
    insecure_client = Client(endpoint)
    with pytest.raises(Exception):
        await insecure_client.connect()

    paths = ensure_certificates(config)
    secure_client = Client(endpoint)
    await secure_client.set_security_string(
        "Basic256Sha256,SignAndEncrypt,"
        f"{paths['client_certificate']},{paths['client_private_key']},"
        f"{paths['server_certificate']}"
    )
    await secure_client.connect()
    try:
        namespace = await secure_client.get_namespace_index(
            str(config["opcua"]["namespace_uri"])
        )
        assert namespace > 0
    finally:
        await secure_client.disconnect()
