from __future__ import annotations

from pathlib import Path
import hashlib
import orjson
import pandas as pd


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def sha256_text(text: str) -> str:
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
	return orjson.loads(Path(path).read_bytes())


def save_json(obj: dict, path: Path) -> None:
	Path(path).write_bytes(orjson.dumps(obj))


def read_csv(path: Path) -> pd.DataFrame:
	return pd.read_csv(path)