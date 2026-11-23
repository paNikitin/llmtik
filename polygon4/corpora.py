"""Открытый учебный набор Полигона 4 (только сценарии train_01..train_12)."""
from __future__ import annotations

import json
from pathlib import Path

from .spec import ScenarioSpec

_DATA_PATH = Path(__file__).with_name("train_scenarios.json")


def build_scenarios(split: str = "train") -> list[ScenarioSpec]:
    if split != "train":
        raise ValueError("в студенческий пакет входят только учебные сценарии train")
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return [ScenarioSpec.from_dict(item) for item in raw]
