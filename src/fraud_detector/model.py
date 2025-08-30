from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import ModelConfig
from .utils import ensure_dir


@dataclass
class AnomalyModel:
	config: ModelConfig
	model: IsolationForest | None = None

	def fit(self, X: pd.DataFrame) -> None:
		self.model = IsolationForest(
			contamination=self.config.contamination,
			random_state=self.config.random_state,
		)
		self.model.fit(X)

	def score(self, X: pd.DataFrame) -> np.ndarray:
		if self.model is None:
			raise RuntimeError("Model not trained")
		# sklearn returns higher is less anomalous; invert to [0,1] suspicious score
		d_scores = -self.model.score_samples(X)
		# min-max normalize per batch
		mn, mx = float(np.min(d_scores)), float(np.max(d_scores))
		if mx - mn <= 1e-12:
			return np.zeros_like(d_scores)
		return (d_scores - mn) / (mx - mn)

	def save(self, model_dir: Path) -> Path:
		ensure_dir(model_dir)
		path = Path(model_dir) / "isolation_forest.joblib"
		joblib.dump({"cfg": self.config.__dict__, "model": self.model}, path)
		return path

	@staticmethod
	def load(model_dir: Path) -> "AnomalyModel":
		path = Path(model_dir) / "isolation_forest.joblib"
		obj = joblib.load(path)
		m = AnomalyModel(ModelConfig(**obj["cfg"]))
		m.model = obj["model"]
		return m