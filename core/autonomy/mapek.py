from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from core.twin.models import FaultType, TwinState


class AdaptationAction(StrEnum):
    NONE = "none"
    SAFE_STOP = "safe_stop"
    REROUTE = "reroute"
    ADAPT_IMPEDANCE = "adapt_impedance"
    MAINTENANCE_ALERT = "maintenance_alert"
    RESTORE_IMPEDANCE = "restore_impedance"


@dataclass(frozen=True)
class AnalysisResult:
    primary_fault: FaultType
    active_faults: tuple[FaultType, ...]
    healthy_streak: int
    reason: str


@dataclass(frozen=True)
class AdaptationPlan:
    action: AdaptationAction
    reason: str
    target_ap: int | None = None
    target_impedance: tuple[float, float, float] | None = None


class MAPEKAnalyzer:
    """带连续样本、恢复迟滞和状态记忆的 Analyze 阶段。"""

    def __init__(self, thresholds: Mapping[str, Any]):
        self._t = thresholds
        self._counts: dict[str, int] = {}
        self._latched: set[FaultType] = set()
        self._healthy_streak = 0

    def analyze(self, state: TwinState) -> AnalysisResult:
        active_rssi = (
            state.network.ap1_rssi
            if state.network.active_ap == 1
            else state.network.ap2_rssi
        )
        self._update_latched(
            FaultType.AP_SIGNAL,
            active_rssi <= float(self._t["ap_fail_dbm"]),
            int(self._t["ap_fail_samples"]),
            active_rssi > float(self._t["ap_recover_dbm"]),
            int(self._t["ap_recover_samples"]),
        )
        self._update_latched(
            FaultType.RTT_SPIKE,
            state.network.rtt_sec >= float(self._t["rtt_spike_sec"]),
            int(self._t["rtt_spike_samples"]),
            state.network.rtt_sec < float(self._t["rtt_recover_sec"]),
            int(self._t["rtt_recover_samples"]),
        )
        self._update_latched(
            FaultType.STRESS_WARNING,
            abs(state.physical.stress) >= float(self._t["stress_warning_n"]),
            int(self._t["stress_warning_samples"]),
            abs(state.physical.stress) < float(self._t["stress_warning_n"]) * 0.8,
            int(self._t["healthy_samples"]),
        )
        self._update_latched(
            FaultType.MODEL_MISMATCH,
            state.prediction.normalized_residual
            > float(self._t["model_residual_ratio"]),
            int(self._t["model_residual_samples"]),
            state.prediction.normalized_residual
            <= float(self._t["model_residual_ratio"]) * 0.8,
            int(self._t["healthy_samples"]),
        )

        if (
            state.synchronization.data_age_sec
            > float(self._t["plc_stale_sec"])
            or state.health.data_quality != "good"
        ):
            self._latched.add(FaultType.PLC_DISCONNECTED)
        elif state.health.data_quality == "good":
            self._latched.discard(FaultType.PLC_DISCONNECTED)

        if abs(state.physical.stress) >= float(self._t["stress_critical_n"]):
            self._latched.add(FaultType.STRESS_CRITICAL)
        else:
            self._latched.discard(FaultType.STRESS_CRITICAL)

        priority = (
            FaultType.PLC_DISCONNECTED,
            FaultType.STRESS_CRITICAL,
            FaultType.AP_SIGNAL,
            FaultType.RTT_SPIKE,
            FaultType.STRESS_WARNING,
            FaultType.MODEL_MISMATCH,
        )
        active = tuple(item for item in priority if item in self._latched)
        if active:
            self._healthy_streak = 0
            primary = active[0]
            reason = f"检测到 {primary.value}，活动故障={','.join(x.value for x in active)}"
        else:
            self._healthy_streak += 1
            primary = FaultType.NONE
            reason = f"连续健康样本 {self._healthy_streak}"
        return AnalysisResult(primary, active, self._healthy_streak, reason)

    def _update_latched(
        self,
        fault: FaultType,
        trigger: bool,
        trigger_samples: int,
        recover: bool,
        recover_samples: int,
    ) -> None:
        trigger_key = f"{fault.value}:trigger"
        recover_key = f"{fault.value}:recover"
        self._counts[trigger_key] = self._counts.get(trigger_key, 0) + 1 if trigger else 0
        self._counts[recover_key] = self._counts.get(recover_key, 0) + 1 if recover else 0
        if self._counts[trigger_key] >= trigger_samples:
            self._latched.add(fault)
        if self._counts[recover_key] >= recover_samples:
            self._latched.discard(fault)


class MAPEKPlanner:
    """按安全优先级生成确定性 Plan。"""

    def __init__(
        self,
        config: Mapping[str, Any],
        impedance_config: Mapping[str, Any],
    ):
        self._config = config
        self._impedance = impedance_config

    def plan(
        self,
        state: TwinState,
        analysis: AnalysisResult,
    ) -> AdaptationPlan:
        fault = analysis.primary_fault
        if fault in (FaultType.PLC_DISCONNECTED, FaultType.STRESS_CRITICAL):
            return AdaptationPlan(
                AdaptationAction.SAFE_STOP,
                f"最高优先级故障 {fault.value}，锁存安全停车",
            )
        if fault is FaultType.AP_SIGNAL:
            current = state.network.active_ap
            backup = 2 if current == 1 else 1
            current_rssi = (
                state.network.ap1_rssi if current == 1 else state.network.ap2_rssi
            )
            backup_rssi = (
                state.network.ap1_rssi if backup == 1 else state.network.ap2_rssi
            )
            margin = float(self._config["route"]["minimum_margin_db"])
            if backup_rssi >= current_rssi + margin:
                return AdaptationPlan(
                    AdaptationAction.REROUTE,
                    f"备用 AP{backup} 信号高 {backup_rssi-current_rssi:.1f} dB",
                    target_ap=backup,
                )
        if fault in (FaultType.RTT_SPIKE, FaultType.STRESS_WARNING):
            md = state.control.md
            bd_max = float(self._impedance["B_base"]) + float(
                self._impedance["B_delta_max"]
            )
            kd_min = max(
                float(self._impedance["K_base"])
                - float(self._impedance["K_delta_max"]),
                1e-4,
            )
            return AdaptationPlan(
                AdaptationAction.ADAPT_IMPEDANCE,
                f"{fault.value} 触发柔顺降级",
                target_impedance=(
                    md,
                    min(bd_max, 1.2 * state.control.bd),
                    max(kd_min, 0.6 * state.control.kd),
                ),
            )
        if fault is FaultType.MODEL_MISMATCH:
            return AdaptationPlan(
                AdaptationAction.MAINTENANCE_ALERT,
                "数字孪生预测残差持续超限，记录维护告警",
            )
        if analysis.healthy_streak >= int(self._config["thresholds"]["healthy_samples"]):
            base = (
                float(self._impedance["M_base"]),
                float(self._impedance["B_base"]),
                float(self._impedance["K_base"]),
            )
            rate = float(self._config["adaptation"]["restore_rate"])
            current = (state.control.md, state.control.bd, state.control.kd)
            target = tuple(
                value + rate * (reference - value)
                for value, reference in zip(current, base, strict=True)
            )
            if max(abs(a - b) for a, b in zip(current, target, strict=True)) > 1e-3:
                return AdaptationPlan(
                    AdaptationAction.RESTORE_IMPEDANCE,
                    "连续健康，平滑恢复基准阻抗",
                    target_impedance=target,
                )
        return AdaptationPlan(AdaptationAction.NONE, analysis.reason)
