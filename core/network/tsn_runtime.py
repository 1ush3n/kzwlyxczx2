from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class NetworkRuntimeState:
    ap1_rssi: float
    ap2_rssi: float
    active_ap: int
    active_route: str
    route_switch_count: int
    link_occupancy: float


class TSNRuntimeController:
    """面向在线运行的 TSN 路由控制器，GNN 采用惰性加载。"""

    ROUTES = {
        1: "Server-SW1-SW3-SW5-AP1-AGV",
        2: "Server-SW2-SW4-SW6-AP2-AGV",
    }
    AP_POSITIONS = {1: (5.0, 2.0), 2: (15.0, 2.0)}

    def __init__(self, config: Mapping[str, Any], project_root: Path):
        self._config = config
        self._project_root = project_root
        self._active_ap = 1
        self._switch_count = 0
        self._last_switch = 0.0
        self._attenuation: dict[int, tuple[float, float]] = {}
        self._disabled: set[int] = set()
        self._model: Any | None = None
        self._model_load_attempted = False

    def observe(self, agv_position: float) -> NetworkRuntimeState:
        rssi = {
            ap: self._calculate_rssi(agv_position, position)
            for ap, position in self.AP_POSITIONS.items()
        }
        now = time.monotonic()
        for ap, (loss, expiry) in list(self._attenuation.items()):
            if now >= expiry:
                del self._attenuation[ap]
            else:
                rssi[ap] -= loss
        for ap in self._disabled:
            rssi[ap] = -120.0
        return NetworkRuntimeState(
            ap1_rssi=rssi[1],
            ap2_rssi=rssi[2],
            active_ap=self._active_ap,
            active_route=self.ROUTES[self._active_ap],
            route_switch_count=self._switch_count,
            link_occupancy=0.0,
        )

    def select_ap(self, ap: int, force: bool = False) -> bool:
        if ap not in (1, 2) or ap in self._disabled:
            return False
        cooldown = float(self._config["route"]["switch_cooldown_sec"])
        now = time.monotonic()
        if not force and now - self._last_switch < cooldown:
            return False
        if ap != self._active_ap:
            self._active_ap = ap
            self._switch_count += 1
            self._last_switch = now
        return True

    def select_normal_route(self, agv_position: float) -> NetworkRuntimeState:
        state = self.observe(agv_position)
        preferred = self._predict_ap_with_gnn(agv_position)
        if preferred is None:
            preferred = 1 if state.ap1_rssi >= state.ap2_rssi else 2
        self.select_ap(preferred)
        return self.observe(agv_position)

    def inject_ap_attenuation(
        self,
        ap: int,
        attenuation_db: float,
        duration_sec: float,
    ) -> None:
        if ap not in (1, 2):
            raise ValueError(f"AP 编号必须为 1 或 2，实际为 {ap}")
        self._attenuation[ap] = (
            max(0.0, float(attenuation_db)),
            time.monotonic() + max(0.0, float(duration_sec)),
        )

    def set_link_available(self, ap: int, available: bool) -> None:
        if available:
            self._disabled.discard(ap)
        else:
            self._disabled.add(ap)

    def _predict_ap_with_gnn(self, agv_position: float) -> int | None:
        if not self._model_load_attempted:
            self._load_model()
        if self._model is None:
            return None
        try:
            import torch
            from tsn_net.topology import TSNTopology

            topology = TSNTopology(num_nodes=10, num_ap=2, agv_idx=9)
            topology.update_roaming_rssi(float(agv_position))
            data = topology.get_pyg_data()
            with torch.inference_mode():
                hidden = self._model.encode(data)
                # hidden shape: [N, H]，从服务器节点自回归选择到 AGV 的路径。
                current = 0
                visited = {current}
                path = [current]
                for _ in range(topology.num_nodes):
                    mask = torch.zeros(topology.num_nodes, dtype=torch.bool)
                    for neighbor in topology.get_neighbors(current):
                        if neighbor not in visited:
                            mask[neighbor] = True
                    if not torch.any(mask):
                        break
                    logits = self._model.get_routing_logits(hidden, current, 9, mask)
                    current = int(torch.argmax(logits).item())
                    path.append(current)
                    if current == 9:
                        return 1 if 7 in path else 2 if 8 in path else None
                    visited.add(current)
        except (ImportError, RuntimeError, ValueError, IndexError):
            return None
        return None

    def _load_model(self) -> None:
        self._model_load_attempted = True
        checkpoint = self._project_root / str(self._config["gnn_checkpoint"])
        if not checkpoint.exists():
            return
        try:
            import torch
            from agent.gnn_actor_critic import GNNActorCritic

            model = GNNActorCritic(node_dim=3, edge_dim=4, hidden_dim=64)
            state_dict = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
        except (ImportError, RuntimeError, ValueError):
            self._model = None

    @staticmethod
    def _calculate_rssi(
        agv_x: float,
        ap_position: tuple[float, float],
    ) -> float:
        distance = math.sqrt((agv_x - ap_position[0]) ** 2 + ap_position[1] ** 2)
        rssi = -30.0 - 30.0 * math.log10(distance + 1.0)
        return max(-95.0, min(-20.0, rssi))
