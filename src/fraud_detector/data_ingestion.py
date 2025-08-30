from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from .utils import read_csv


class DataIngestion:
	def __init__(self, source: Path, kind: str = "csv") -> None:
		self.source = Path(source)
		self.kind = kind

	def load(self) -> pd.DataFrame:
		if self.kind == "csv":
			return read_csv(self.source)
		elif self.kind == "json":
			return pd.read_json(self.source, lines=False)
		# Placeholder for SQL ingestion
		raise ValueError(f"Unsupported source kind: {self.kind}")