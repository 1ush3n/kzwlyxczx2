from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.twin.models import TwinState


class AASExporter:
    """生成 AAS 3.0 基本结构的轻量级 JSON 快照。"""

    def __init__(self, static_config: Mapping[str, Any]):
        self._static = static_config

    def export(self, state: TwinState, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "assetAdministrationShells": [
                {
                    "id": str(self._static["aas_id"]),
                    "idShort": state.identity.asset_id,
                    "assetInformation": {
                        "assetKind": "Instance",
                        "globalAssetId": str(self._static["global_asset_id"]),
                    },
                    "submodels": [
                        "Identification",
                        "TechnicalData",
                        "OperationalData",
                        "Communication",
                        "ConditionMonitoring",
                        "Documentation",
                    ],
                }
            ],
            "submodels": {
                "Identification": {
                    "assetId": state.identity.asset_id,
                    "assetType": state.identity.asset_type,
                    "version": state.identity.version,
                    "manufacturer": str(self._static["manufacturer"]),
                },
                "TechnicalData": dict(self._static["technical_data"]),
                "OperationalData": {
                    "physical": state.to_dict()["physical"],
                    "control": state.to_dict()["control"],
                },
                "Communication": state.to_dict()["network"],
                "ConditionMonitoring": {
                    "health": state.to_dict()["health"],
                    "prediction": state.to_dict()["prediction"],
                    "synchronization": state.to_dict()["synchronization"],
                },
                "Documentation": {
                    "description": str(self._static["documentation"]),
                    "format": "AAS 3.0 lightweight JSON snapshot",
                },
            },
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(output_path)
