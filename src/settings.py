from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    brand: str
    conversations_path: str
    golden_path: str
    metrics_path: str
    min_intent_confidence: float
    min_retrieval_similarity: float
    top_k: int


def load_settings(path: str | Path = "config.yaml") -> Settings:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Settings(
        brand=str(config["brand"]),
        conversations_path=str(config["paths"]["conversations"]),
        golden_path=str(config["paths"]["golden_set"]),
        metrics_path=str(config["paths"]["metrics"]),
        min_intent_confidence=float(config["model"]["min_intent_confidence"]),
        min_retrieval_similarity=float(config["model"]["min_retrieval_similarity"]),
        top_k=int(config["model"]["top_k"]),
    )
