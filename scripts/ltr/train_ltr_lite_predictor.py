#!/usr/bin/env python3
"""Train a lightweight, explainable LTR-lite predictor.

This script learns a cheap request-length ranking score from existing serving
logs. It intentionally avoids a neural predictor in the first pass: the model is
a small ridge regressor implemented with NumPy, using transparent features from
the request manifest.

The exported score file can be consumed directly by the server-side vLLM `ltr`
policy via `VLLM_LTR_SCORE_FILE`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

import numpy as np


NUMERIC_FEATURES = [
    "target_max_new_tokens",
    "prompt_chars",
    "prompt_words",
    "prompt_lines",
    "prompt_digits",
    "prompt_punctuation",
    "prompt_ascii_ratio",
    "prompt_cjk_chars",
]

CATEGORICAL_FIELDS = [
    "group",
    "lang",
    "domain",
    "task_family",
    "topic",
    "source",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
    return rows


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    for idx, row in enumerate(rows, start=1):
        for key in ("id", "prompt", "target_max_new_tokens"):
            if key not in row:
                raise ValueError(f"{path}:{idx} missing required key: {key}")
    return rows


def expand_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(item) for item in matches)
        else:
            paths.append(Path(pattern))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            seen.add(resolved)
            deduped.append(path)
    return deduped


def maybe_build_tokenizer(tokenizer_path: str | None):
    if tokenizer_path is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def count_input_tokens(prompt: str, tokenizer) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def prompt_stats(prompt: str) -> dict[str, float]:
    chars = len(prompt)
    ascii_chars = sum(1 for char in prompt if ord(char) < 128)
    cjk_chars = sum(1 for char in prompt if "\u4e00" <= char <= "\u9fff")
    punctuation = sum(1 for char in prompt if not char.isalnum() and not char.isspace())
    return {
        "prompt_chars": float(chars),
        "prompt_words": float(len(re.findall(r"\S+", prompt))),
        "prompt_lines": float(prompt.count("\n") + 1),
        "prompt_digits": float(sum(1 for char in prompt if char.isdigit())),
        "prompt_punctuation": float(punctuation),
        "prompt_ascii_ratio": float(ascii_chars / max(chars, 1)),
        "prompt_cjk_chars": float(cjk_chars),
    }


def collect_labels(
    raw_log_paths: list[Path],
    valid_request_ids: set[str],
    *,
    label_strategy: str | None,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    values: dict[str, list[float]] = defaultdict(list)
    used_logs: list[str] = []

    for path in raw_log_paths:
        if "gpu_metrics" in path.name:
            continue
        try:
            rows = load_jsonl(path)
        except ValueError:
            continue
        used_in_file = 0
        for row in rows:
            request_id = str(row.get("request_id", ""))
            if request_id not in valid_request_ids:
                continue
            if "output_tokens" not in row:
                continue
            if label_strategy is not None and row.get("strategy") != label_strategy:
                continue
            values[request_id].append(float(row["output_tokens"]))
            used_in_file += 1
        if used_in_file:
            used_logs.append(str(path))

    labels: dict[str, dict[str, float]] = {}
    for request_id, samples in values.items():
        arr = np.asarray(samples, dtype=float)
        labels[request_id] = {
            "mean_output_tokens": float(arr.mean()),
            "std_output_tokens": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "num_label_samples": float(len(arr)),
        }
    return labels, used_logs


def build_feature_space(requests: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
    categories: dict[str, set[str]] = {field: set() for field in CATEGORICAL_FIELDS}
    tags: set[str] = set()
    for row in requests:
        for field in CATEGORICAL_FIELDS:
            categories[field].add(str(row.get(field, "unknown")))
        for tag in row.get("tags", []):
            tags.add(str(tag))

    category_values = {field: sorted(values) for field, values in categories.items()}
    category_values["tag"] = sorted(tags)

    feature_names = list(NUMERIC_FEATURES)
    feature_names.append("prompt_tokens")
    for field in CATEGORICAL_FIELDS:
        feature_names.extend(f"{field}={value}" for value in category_values[field])
    feature_names.extend(f"tag={value}" for value in category_values["tag"])
    return feature_names, category_values


def featurize_request(
    row: dict[str, Any],
    *,
    feature_names: list[str],
    category_values: dict[str, list[str]],
    tokenizer,
) -> np.ndarray:
    prompt = str(row["prompt"])
    numeric = prompt_stats(prompt)
    numeric["target_max_new_tokens"] = float(row["target_max_new_tokens"])
    prompt_tokens = count_input_tokens(prompt, tokenizer)
    numeric["prompt_tokens"] = float(prompt_tokens if prompt_tokens is not None else numeric["prompt_chars"])

    values: dict[str, float] = {name: float(numeric.get(name, 0.0)) for name in NUMERIC_FEATURES}
    values["prompt_tokens"] = numeric["prompt_tokens"]
    for field in CATEGORICAL_FIELDS:
        current = str(row.get(field, "unknown"))
        for category in category_values[field]:
            values[f"{field}={category}"] = 1.0 if current == category else 0.0
    row_tags = {str(tag) for tag in row.get("tags", [])}
    for tag in category_values["tag"]:
        values[f"tag={tag}"] = 1.0 if tag in row_tags else 0.0
    return np.asarray([values[name] for name in feature_names], dtype=float)


def stable_split(ids: list[str], validation_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * validation_fraction))) if len(shuffled) >= 5 else 0
    return shuffled[n_val:], shuffled[:n_val]


def standardize_train(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (X - mean) / scale, mean, scale


def standardize_apply(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - mean) / scale


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    penalty = np.eye(X_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    lhs = X_aug.T @ X_aug + penalty
    rhs = X_aug.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def predict(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    return X_aug @ coef


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    a = rankdata(y_true)
    b = rankdata(y_pred)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a @ a) * (b @ b)))
    return float((a @ b) / denom) if denom else float("nan")


def kendall_tau_b(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    if n < 2:
        return float("nan")
    concordant = discordant = ties_true = ties_pred = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_true = y_true[i] - y_true[j]
            d_pred = y_pred[i] - y_pred[j]
            if d_true == 0 and d_pred == 0:
                continue
            if d_true == 0:
                ties_true += 1
            elif d_pred == 0:
                ties_pred += 1
            elif d_true * d_pred > 0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_true) * (concordant + discordant + ties_pred))
    return float((concordant - discordant) / denom) if denom else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(float(np.mean(err * err)))),
        "kendall_tau_b": kendall_tau_b(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
    }


def request_hash(row: dict[str, Any]) -> str:
    text = json.dumps(
        {
            "id": row.get("id"),
            "prompt": row.get("prompt"),
            "target_max_new_tokens": row.get("target_max_new_tokens"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument(
        "--raw-log-glob",
        action="append",
        default=[],
        help="Raw request log glob. Can be passed multiple times.",
    )
    parser.add_argument("--output-score-file", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--label-strategy", default="fcfs")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--limit-output", type=int, default=None)
    args = parser.parse_args()

    if not args.raw_log_glob:
        args.raw_log_glob = ["logs/raw/*.jsonl"]
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")

    requests = load_requests(args.request_file)
    request_by_id = {str(row["id"]): row for row in requests}
    raw_log_paths = expand_globs(args.raw_log_glob)
    labels, used_logs = collect_labels(
        raw_log_paths,
        set(request_by_id),
        label_strategy=args.label_strategy if args.label_strategy else None,
    )
    labeled_ids = sorted(labels)
    if len(labeled_ids) < 8:
        raise ValueError(f"not enough labeled requests: {len(labeled_ids)}")

    tokenizer = maybe_build_tokenizer(args.tokenizer_path)
    feature_names, category_values = build_feature_space(requests)

    X_all_by_id = {
        request_id: featurize_request(
            row,
            feature_names=feature_names,
            category_values=category_values,
            tokenizer=tokenizer,
        )
        for request_id, row in request_by_id.items()
    }

    train_ids, val_ids = stable_split(labeled_ids, args.validation_fraction, args.seed)
    X_train_raw = np.vstack([X_all_by_id[request_id] for request_id in train_ids])
    y_train = np.asarray([labels[request_id]["mean_output_tokens"] for request_id in train_ids], dtype=float)
    X_train, feature_mean, feature_scale = standardize_train(X_train_raw)
    coef = fit_ridge(X_train, y_train, args.alpha)

    def predict_ids(ids: list[str]) -> np.ndarray:
        raw = np.vstack([X_all_by_id[request_id] for request_id in ids])
        return predict(standardize_apply(raw, feature_mean, feature_scale), coef)

    train_pred = predict_ids(train_ids)
    report: dict[str, Any] = {
        "model_type": "numpy_ridge_ltr_lite",
        "score_convention": "lower score means predicted shorter request and earlier scheduling",
        "request_file": str(args.request_file),
        "raw_log_globs": args.raw_log_glob,
        "used_logs": used_logs,
        "label_strategy": args.label_strategy,
        "num_requests_in_manifest": len(requests),
        "num_labeled_requests": len(labeled_ids),
        "num_train_requests": len(train_ids),
        "num_validation_requests": len(val_ids),
        "alpha": args.alpha,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "feature_count": len(feature_names),
        "train_metrics": regression_metrics(y_train, train_pred),
    }

    if val_ids:
        y_val = np.asarray([labels[request_id]["mean_output_tokens"] for request_id in val_ids], dtype=float)
        val_pred = predict_ids(val_ids)
        report["validation_metrics"] = regression_metrics(y_val, val_pred)
        for baseline_name in ("target_max_new_tokens", "estimated_total_tokens"):
            baseline_scores = []
            for request_id in val_ids:
                row = request_by_id[request_id]
                target = float(row["target_max_new_tokens"])
                if baseline_name == "estimated_total_tokens":
                    target += X_all_by_id[request_id][feature_names.index("prompt_tokens")]
                baseline_scores.append(target)
            report[f"validation_{baseline_name}_ranking"] = {
                "kendall_tau_b": kendall_tau_b(y_val, np.asarray(baseline_scores)),
                "spearman": spearman_corr(y_val, np.asarray(baseline_scores)),
            }

    coef_by_feature = {
        feature_name: float(value)
        for feature_name, value in zip(feature_names, coef[1:], strict=True)
    }
    report["top_standardized_coefficients"] = [
        {"feature": feature, "coefficient": coef_by_feature[feature]}
        for feature in sorted(coef_by_feature, key=lambda name: abs(coef_by_feature[name]), reverse=True)[:30]
    ]
    sample_counts = [labels[request_id]["num_label_samples"] for request_id in labeled_ids]
    report["label_samples_per_request"] = {
        "min": float(min(sample_counts)),
        "mean": float(sum(sample_counts) / len(sample_counts)),
        "max": float(max(sample_counts)),
    }

    model = {
        "model_type": "numpy_ridge_ltr_lite",
        "feature_names": feature_names,
        "category_values": category_values,
        "intercept": float(coef[0]),
        "coefficients": [float(value) for value in coef[1:]],
        "feature_mean": [float(value) for value in feature_mean],
        "feature_scale": [float(value) for value in feature_scale],
        "alpha": args.alpha,
        "request_file": str(args.request_file),
        "label_strategy": args.label_strategy,
    }

    output_requests = requests[: args.limit_output] if args.limit_output is not None else requests
    output_ids = [str(row["id"]) for row in output_requests]
    output_scores = predict_ids(output_ids)

    args.output_score_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_score_file.open("w", encoding="utf-8") as f:
        for row, score in zip(output_requests, output_scores, strict=True):
            request_id = str(row["id"])
            label = labels.get(request_id)
            out = {
                "request_id": request_id,
                "score": float(score),
                "score_source": "ltr_lite_numpy_ridge",
                "lower_is_shorter": True,
                "target_max_new_tokens": int(row["target_max_new_tokens"]),
                "request_hash": request_hash(row),
            }
            if label is not None:
                out["label_mean_output_tokens"] = label["mean_output_tokens"]
                out["label_num_samples"] = int(label["num_label_samples"])
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"trained LTR-lite on {len(train_ids)} requests; validation={len(val_ids)}")
    print(f"wrote scores: {args.output_score_file}")
    print(f"wrote model: {args.model_output}")
    print(f"wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
