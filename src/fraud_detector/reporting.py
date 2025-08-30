from __future__ import annotations

from pathlib import Path
import pandas as pd


def generate_reasons(row: pd.Series) -> list[str]:
	reasons: list[str] = []
	if row.get("amount_zscore", 0) > 3:
		reasons.append("High z-score amount")
	if row.get("amount_over_max_ratio", 0) > 1.2:
		reasons.append("Exceeds historical max")
	if row.get("tod_prop_for_user", 1) < 0.05:
		reasons.append("Unusual time of day")
	if row.get("cat_prop_for_user", 1) < 0.05:
		reasons.append("Rare category for user")
	return reasons or ["Anomalous pattern"]


def write_flags_csv(df: pd.DataFrame, out_path: Path) -> Path:
	out = Path(out_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out, index=False)
	return out