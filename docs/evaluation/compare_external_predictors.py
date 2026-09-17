#!/usr/bin/env python3
"""Compare PALM/ESM scores with external numeric or categorical predictors.

The script writes both strict complete-case metrics and pairwise matched metrics
against a declared reference score. It never fills a missing external-tool
prediction, so every AUROC reports its exact evaluable cohort.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def split_columns(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def categorical_score(value: object, pathogenic: set[str], benign: set[str]) -> float:
    if pd.isna(value):
        return np.nan
    calls = {
        call.strip().upper()
        for part in str(value).replace(";", ",").replace("|", ",").split(",")
        for call in [part]
        if call.strip() and call.strip() != "."
    }
    pathogenic_votes = len(calls & pathogenic)
    benign_votes = len(calls & benign)
    if pathogenic_votes == benign_votes == 0 or pathogenic_votes == benign_votes:
        return np.nan
    return float(pathogenic_votes > benign_votes)


def metric(label: pd.Series, score: pd.Series, name: str, cohort: str) -> dict[str, object]:
    usable = pd.DataFrame({"label": label, "score": score}).dropna()
    if len(usable) < 2 or usable["label"].nunique() < 2:
        return {
            "cohort": cohort,
            "predictor": name,
            "n": len(usable),
            "pathogenic": int(usable["label"].sum()) if len(usable) else 0,
            "benign": int((usable["label"] == 0).sum()) if len(usable) else 0,
            "auroc": np.nan,
            "average_precision": np.nan,
            "orientation": "unavailable",
        }
    auroc = roc_auc_score(usable["label"], usable["score"])
    orientation = "higher_is_pathogenic"
    if auroc < 0.5:
        auroc = 1 - auroc
        usable["score"] = -usable["score"]
        orientation = "lower_is_pathogenic"
    return {
        "cohort": cohort,
        "predictor": name,
        "n": len(usable),
        "pathogenic": int(usable["label"].sum()),
        "benign": int((usable["label"] == 0).sum()),
        "auroc": auroc,
        "average_precision": average_precision_score(usable["label"], usable["score"]),
        "orientation": orientation,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--numeric-columns", required=True, help="Comma-separated continuous scores")
    parser.add_argument("--categorical-columns", default="", help="Comma-separated categorical calls")
    parser.add_argument("--reference-score", required=True, help="Score used for pairwise matched comparisons")
    parser.add_argument(
        "--pathogenic-values",
        default="D,A,H,M,P,PATHOGENIC,LIKELY_PATHOGENIC",
    )
    parser.add_argument(
        "--benign-values",
        default="T,B,L,N,NEUTRAL,BENIGN,LIKELY_BENIGN",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.input, sep="\t", low_memory=False)
    numeric = split_columns(args.numeric_columns)
    categorical = split_columns(args.categorical_columns)
    predictors = numeric + categorical
    if args.reference_score not in predictors:
        raise ValueError("--reference-score must also be listed as a numeric or categorical column.")
    required = {args.label_column, args.reference_score, *predictors}
    if missing := required - set(table.columns):
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    labels = pd.to_numeric(table[args.label_column], errors="coerce")
    if not set(labels.dropna().unique()).issubset({0, 1}):
        raise ValueError("The label column must contain only 0, 1, or missing values.")
    score_table = pd.DataFrame(index=table.index)
    for column in numeric:
        score_table[column] = pd.to_numeric(table[column], errors="coerce")
    pathogenic = {value.upper() for value in split_columns(args.pathogenic_values)}
    benign = {value.upper() for value in split_columns(args.benign_values)}
    for column in categorical:
        score_table[column] = table[column].map(
            lambda value: categorical_score(value, pathogenic, benign)
        )

    strict_mask = labels.notna() & score_table[predictors].notna().all(axis=1)
    strict_rows = [
        metric(labels[strict_mask], score_table.loc[strict_mask, predictor], predictor, "strict_complete_case")
        for predictor in predictors
    ]
    strict = pd.DataFrame(strict_rows).sort_values("auroc", ascending=False)
    strict.to_csv(args.output_dir / "strict_complete_case_metrics.tsv", sep="\t", index=False)

    pairwise_rows = []
    for predictor in predictors:
        if predictor == args.reference_score:
            continue
        mask = labels.notna() & score_table[args.reference_score].notna() & score_table[predictor].notna()
        pairwise_rows.append(
            metric(labels[mask], score_table.loc[mask, args.reference_score], args.reference_score, f"matched_with:{predictor}")
        )
        pairwise_rows.append(
            metric(labels[mask], score_table.loc[mask, predictor], predictor, f"matched_with:{args.reference_score}")
        )
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(args.output_dir / "pairwise_matched_metrics.tsv", sep="\t", index=False)

    scored = pd.concat([table, score_table.add_prefix("benchmark_score__")], axis=1)
    scored.to_csv(args.output_dir / "scored_predictors.tsv.gz", sep="\t", index=False, compression="gzip")
    print("Strict complete-case metrics")
    print(strict.to_string(index=False))


if __name__ == "__main__":
    main()
