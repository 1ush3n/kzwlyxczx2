from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.autonomy.mapek import AdaptationAction, MAPEKAnalyzer, MAPEKPlanner
from core.comms.config import load_protocol_config, load_yaml_mapping, to_mapping
from core.network.tsn_runtime import TSNRuntimeController
from core.twin.models import FaultType
from tests.test_twin_and_aas import make_state


def test_ap_fault_requires_consecutive_samples_and_has_priority() -> None:
    config = to_mapping(load_protocol_config())
    analyzer = MAPEKAnalyzer(config["mapek"]["thresholds"])
    state = make_state()
    network = replace(state.network, ap1_rssi=-90.0, active_ap=1)
    fault_state = replace(state, network=network)
    assert analyzer.analyze(fault_state).primary_fault is FaultType.NONE
    assert analyzer.analyze(fault_state).primary_fault is FaultType.NONE
    result = analyzer.analyze(fault_state)
    assert result.primary_fault is FaultType.AP_SIGNAL

    agv = load_yaml_mapping(Path(config["runtime"]["project_root"]) / "config" / "agv_env_config.yaml")
    planner = MAPEKPlanner(config["mapek"], agv["impedance"])
    plan = planner.plan(fault_state, result)
    assert plan.action is AdaptationAction.REROUTE
    assert plan.target_ap == 2


def test_rtt_adaptation_and_plc_priority() -> None:
    config = to_mapping(load_protocol_config())
    agv = load_yaml_mapping(Path(config["runtime"]["project_root"]) / "config" / "agv_env_config.yaml")
    analyzer = MAPEKAnalyzer(config["mapek"]["thresholds"])
    planner = MAPEKPlanner(config["mapek"], agv["impedance"])
    state = make_state()
    rtt_state = replace(state, network=replace(state.network, rtt_sec=0.25))
    analyzer.analyze(rtt_state)
    analyzer.analyze(rtt_state)
    analysis = analyzer.analyze(rtt_state)
    plan = planner.plan(rtt_state, analysis)
    assert plan.action is AdaptationAction.ADAPT_IMPEDANCE
    assert plan.target_impedance == (50.0, 600.0, 1800.0)

    disconnected = replace(
        rtt_state,
        health=replace(rtt_state.health, data_quality="bad"),
        synchronization=replace(rtt_state.synchronization, data_age_sec=0.5),
    )
    stop_plan = planner.plan(disconnected, analyzer.analyze(disconnected))
    assert stop_plan.action is AdaptationAction.SAFE_STOP


def test_tsn_runtime_attenuation_and_switch_cooldown(tmp_path: Path) -> None:
    config = to_mapping(load_protocol_config())["tsn_runtime"]
    controller = TSNRuntimeController(config, tmp_path)
    controller.inject_ap_attenuation(1, 60.0, 1.0)
    state = controller.observe(0.0)
    assert state.ap1_rssi < state.ap2_rssi
    assert controller.select_ap(2)
    assert controller.observe(0.0).route_switch_count == 1
    assert not controller.select_ap(1)
