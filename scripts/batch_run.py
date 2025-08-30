from __future__ import annotations

from pathlib import Path
import argparse

from src.fraud_detector.synthetic import generate_synthetic, save_csv
from src.fraud_detector.cli import cmd_train, cmd_score


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--out_dir", default="data")
	parser.add_argument("--model_dir", default="models")
	parser.add_argument("--flags_out", default="reports/flags.csv")
	args = parser.parse_args()

	data_dir = Path(args.out_dir)
	data_dir.mkdir(parents=True, exist_ok=True)
	csv = data_dir / "sample.csv"
	df = generate_synthetic()
	save_csv(df, csv)

	train_ns = argparse.Namespace(data=str(csv), kind="csv", model_dir=args.model_dir, config="config.yaml")
	cmd_train(train_ns)

	score_ns = argparse.Namespace(data=str(csv), kind="csv", model_dir=args.model_dir, out=args.flags_out, config="config.yaml")
	cmd_score(score_ns)


if __name__ == "__main__":
	main()