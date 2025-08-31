from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .config import AppConfig
from .data_ingestion import DataIngestion
from .preprocessing import preprocess
from .profiling import create_transaction_features, select_feature_matrix
from .model import AnomalyModel
from .reporting import generate_reasons, write_flags_csv
try:
    from .dashboard import run_streamlit_dashboard
except ImportError:
    def run_streamlit_dashboard():
        print("❌ Streamlit is not installed. Please run: pip install streamlit")
        return


def cmd_train(args: argparse.Namespace) -> None:
	cfg = AppConfig.from_yaml(Path(args.config))
	df = DataIngestion(Path(args.data), kind=args.kind).load()
	df = preprocess(df, cfg.features)
	feat = create_transaction_features(df, cfg.features)
	X = select_feature_matrix(feat, cfg.features)
	model = AnomalyModel(cfg.model)
	model.fit(X)
	model.save(Path(args.model_dir))
	print("Model trained and saved.")


def cmd_score(args: argparse.Namespace) -> None:
	cfg = AppConfig.from_yaml(Path(args.config))
	model = AnomalyModel.load(Path(args.model_dir))
	df = DataIngestion(Path(args.data), kind=args.kind).load()
	df = preprocess(df, cfg.features)
	feat = create_transaction_features(df, cfg.features)
	X = select_feature_matrix(feat, cfg.features)
	scores = model.score(X)
	feat["suspicion_score"] = scores
	flags = feat[feat["suspicion_score"] >= cfg.model.threshold].copy()
	flags["reasons"] = flags.apply(generate_reasons, axis=1)
	write_flags_csv(flags, Path(args.out))
	print(f"Flags written to {args.out}")


def cmd_dashboard(args: argparse.Namespace) -> None:
	run_streamlit_dashboard()


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(prog="fraud-detector")
	sub = p.add_subparsers(dest="cmd", required=True)

	pt = sub.add_parser("train")
	pt.add_argument("--data", required=True)
	pt.add_argument("--kind", default="csv")
	pt.add_argument("--model_dir", required=True)
	pt.add_argument("--config", default="config.yaml")
	pt.set_defaults(func=cmd_train)

	ps = sub.add_parser("score")
	ps.add_argument("--data", required=True)
	ps.add_argument("--kind", default="csv")
	ps.add_argument("--model_dir", required=True)
	ps.add_argument("--out", required=True)
	ps.add_argument("--config", default="config.yaml")
	ps.set_defaults(func=cmd_score)

	pd = sub.add_parser("dashboard")
	pd.add_argument("--flags", required=False, help="Optional: path to flags file")
	pd.set_defaults(func=cmd_dashboard)
	return p


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()