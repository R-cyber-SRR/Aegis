from __future__ import annotations

import pandas as pd
import numpy as np

from .config import FeatureConfig


def _time_of_day(hour: int) -> str:
	if 5 <= hour < 12:
		return "morning"
	if 12 <= hour < 17:
		return "afternoon"
	if 17 <= hour < 22:
		return "evening"
	return "night"


def build_profiles(df: pd.DataFrame, features: FeatureConfig) -> pd.DataFrame:
	g = df.groupby(features.id_column)
	profiles = g[features.amount_column].agg([
		("user_amount_mean", "mean"),
		("user_amount_std", "std"),
		("user_amount_max", "max"),
		("user_tx_count", "count"),
	])
	# time of day preferences
	temp = df.copy()
	temp["hour"] = temp[features.timestamp_column].dt.hour
	temp["tod"] = temp["hour"].map(_time_of_day)
	tod_pivot = (
		temp.pivot_table(index=features.id_column, columns="tod", values=features.amount_column, aggfunc="count", fill_value=0)
		.add_prefix("user_tod_")
	)
	profiles = profiles.join(tod_pivot, how="left").fillna(0.0)
	# transaction type/category frequencies
	if features.category_column and features.category_column in df.columns:
		cat_pivot = (
			df.pivot_table(index=features.id_column, columns=features.category_column, values=features.amount_column, aggfunc="count", fill_value=0)
			.add_prefix("user_cat_")
		)
		profiles = profiles.join(cat_pivot, how="left").fillna(0.0)
	return profiles.reset_index()


def create_transaction_features(df: pd.DataFrame, features: FeatureConfig) -> pd.DataFrame:
	profiles = build_profiles(df, features)
	feat = df.copy()
	feat = feat.merge(profiles, on=features.id_column, how="left")
	# compute dynamic features
	feat["amount_to_mean_ratio"] = feat[features.amount_column] / (feat["user_amount_mean"].replace(0, np.nan))
	feat["amount_zscore"] = (feat[features.amount_column] - feat["user_amount_mean"]) / (feat["user_amount_std"].replace(0, np.nan))
	feat["amount_over_max_ratio"] = feat[features.amount_column] / (feat["user_amount_max"].replace(0, np.nan))
	feat["tx_frequency"] = feat["user_tx_count"]
	feat["hour"] = feat[features.timestamp_column].dt.hour
	feat["tod"] = feat["hour"].map(_time_of_day)
	for label in ["morning", "afternoon", "evening", "night"]:
		col = f"user_tod_{label}"
		if col not in feat.columns:
			feat[col] = 0.0
	feat["tod_prop_for_user"] = feat.apply(lambda r: r.get(f"user_tod_{r['tod']}", 0.0) / (r["tx_frequency"] if r["tx_frequency"] else np.nan), axis=1)
	# category propensity
	if features.category_column and features.category_column in feat.columns:
		feat["cat_prop_for_user"] = feat.apply(lambda r: r.get(f"user_cat_{r[features.category_column]}", 0.0) / (r["tx_frequency"] if r["tx_frequency"] else np.nan), axis=1)
	else:
		feat["cat_prop_for_user"] = np.nan
	# replace inf/NaN
	feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
	return feat


def select_feature_matrix(feat_df: pd.DataFrame, features: FeatureConfig) -> pd.DataFrame:
	cols = [
		"amount_to_mean_ratio",
		"amount_zscore",
		"amount_over_max_ratio",
		"tx_frequency",
		"hour",
		"user_tod_morning",
		"user_tod_afternoon",
		"user_tod_evening",
		"user_tod_night",
		"tod_prop_for_user",
		"cat_prop_for_user",
	]
	available = [c for c in cols if c in feat_df.columns]
	return feat_df[available]