#!/usr/bin/env python3
"""Validate and upload PALM checkpoints to one Hugging Face model repository."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi


ANNOTATION_FILENAME = "palm_soft_target_epoch8.pt"
HEAD_FILENAMES = [
    f"pathogenicity/random_split_fold_{fold}.pt" for fold in range(1, 6)
]
EXPECTED_HEAD_EPOCHS = [3, 4, 4, 3, 4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "label_mode": checkpoint.get("label_mode"),
        "specific_class_loss": checkpoint.get("specific_class_loss"),
        "num_domain_classes": int(checkpoint.get("num_domain_classes", -1)),
        "num_site_classes": int(checkpoint.get("num_site_classes", -1)),
    }


def head_metadata(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "feature_dim": int(checkpoint.get("feature_dim", -1)),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="For example organization/PALM")
    parser.add_argument("--annotation-checkpoint", type=Path, required=True)
    parser.add_argument("--pathogenicity-heads", type=Path, nargs=5, required=True)
    parser.add_argument(
        "--model-card",
        type=Path,
        default=Path(__file__).with_name("README.md"),
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", help="Prefer the HF_TOKEN environment variable.")
    parser.add_argument("--execute", action="store_true", help="Upload after validation.")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    annotation = checkpoint_metadata(args.annotation_checkpoint)
    if annotation["epoch"] != 8:
        raise ValueError(f"Annotation checkpoint epoch is {annotation['epoch']}, not 8.")
    if annotation["label_mode"] != "sparse_set_softmax_with_instance_heads":
        raise ValueError("Annotation checkpoint has the wrong label mode.")
    if annotation["specific_class_loss"] != "uniform_soft_target_cross_entropy":
        raise ValueError("Annotation checkpoint has the wrong soft-target loss.")
    head_details = [head_metadata(path) for path in args.pathogenicity_heads]
    epochs = [details["epoch"] for details in head_details]
    if epochs != EXPECTED_HEAD_EPOCHS:
        raise ValueError(f"Expected selected head epochs {EXPECTED_HEAD_EPOCHS}, found {epochs}.")
    feature_dimensions = {details["feature_dim"] for details in head_details}
    if feature_dimensions != {329920}:
        raise ValueError(f"Unexpected pathogenicity feature dimensions: {feature_dimensions}")

    artifacts = [(args.annotation_checkpoint, ANNOTATION_FILENAME)] + list(
        zip(args.pathogenicity_heads, HEAD_FILENAMES)
    )
    manifest = {
        "model": "PALM soft-target overlap-aware epoch 8",
        "annotation_checkpoint": annotation,
        "pathogenicity_ensemble": {
            "split": "random stratified five-fold ClinVar",
            "aggregation": "unweighted arithmetic mean of probabilities",
            "selected_epochs": EXPECTED_HEAD_EPOCHS,
            "feature_dim": 329920,
        },
        "files": [
            {
                "filename": destination,
                "source_name": source.name,
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
            for source, destination in artifacts
        ],
    }
    print(json.dumps(manifest, indent=2))
    if not args.execute:
        print("Validation only. Add --execute to create/upload the Hugging Face repository.")
        return

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        path_or_fileobj=args.model_card,
        path_in_repo="README.md",
        commit_message="Add PALM model card",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        path_or_fileobj=io.BytesIO((json.dumps(manifest, indent=2) + "\n").encode()),
        path_in_repo="checkpoint_manifest.json",
        commit_message="Add checkpoint provenance manifest",
    )
    for source, destination in artifacts:
        print(f"Uploading {source} -> {args.repo_id}/{destination}", flush=True)
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            path_or_fileobj=source,
            path_in_repo=destination,
            commit_message=f"Upload {destination}",
        )
    print(f"Uploaded PALM checkpoints to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

