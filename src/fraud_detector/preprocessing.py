from __future__ import annotations

from typing import Iterable
import pandas as pd

from .config import FeatureConfig
from .utils import sha256_text


def anonymize_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
	for col in columns:
		if col in df.columns:
			df[col] = df[col].astype(str).map(sha256_text)
	return df


def preprocess(df: pd.DataFrame, features: FeatureConfig) -> pd.DataFrame:
	missing = [c for c in features.mandatory_columns if c not in df.columns]
	if missing:
		raise ValueError(f"Missing mandatory columns: {missing}")
	# parse timestamp
	df[features.timestamp_column] = pd.to_datetime(df[features.timestamp_column], errors="coerce")
	# cast amount
	df[features.amount_column] = pd.to_numeric(df[features.amount_column], errors="coerce")
	# drop rows with invalid basic fields
	df = df.dropna(subset=[features.timestamp_column, features.amount_column])
	# anonymize PII-like fields
	pii_candidates = [features.id_column, "ip_address", "device_fingerprint"]
	return anonymize_columns(df, [c for c in pii_candidates if c in df.columns])