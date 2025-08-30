from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class FeatureConfig:
	mandatory_columns: List[str]
	optional_columns: List[str]
	id_column: str
	timestamp_column: str
	amount_column: str
	category_column: Optional[str] = None


@dataclass
class ModelConfig:
	algorithm: str = "isolation_forest"
	contamination: float = 0.02
	random_state: int = 42
	threshold: float = 0.8


@dataclass
class AppConfig:
	features: FeatureConfig
	model: ModelConfig

	@staticmethod
	def from_yaml(path: Path) -> "AppConfig":
		with open(path, "r", encoding="utf-8") as f:
			cfg = yaml.safe_load(f)
		features = FeatureConfig(
			mandatory_columns=cfg["features"]["mandatory_columns"],
			optional_columns=cfg["features"].get("optional_columns", []),
			id_column=cfg["features"]["id_column"],
			timestamp_column=cfg["features"]["timestamp_column"],
			amount_column=cfg["features"]["amount_column"],
			category_column=cfg["features"].get("category_column"),
		)
		model = ModelConfig(**cfg["model"]) if "model" in cfg else ModelConfig()
		return AppConfig(features=features, model=model)