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


ANNOTATION_FILENAME = "palm_post_ep3_six_head.pt"
HEAD_FILENAMES = [
    f"pathogenicity/gene_protein_disjoint_fold_{fold}.pt" for fold in range(1, 6)
]
EXPECTED_HEAD_EPOCHS = [1, 1, 1, 1, 2]
EXPECTED_FEATURE_DIM = 329_968
FINAL_EPOCH = 3

# Files published by the four-head epoch-8 release. --replace-legacy deletes
# them, so the repository holds exactly one model.
LEGACY_FILENAMES = ["palm_soft_target_epoch8.pt"] + [
    f"pathogenicity/random_split_fold_{fold}.pt" for fold in range(1, 6)
]


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
        "num_repeat_classes": int(checkpoint.get("num_repeat_classes", -1)),
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
    parser.add_argument(
        "--replace-legacy",
        action="store_true",
        help="Also delete the four-head epoch-8 files from the repository.",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    annotation = checkpoint_metadata(args.annotation_checkpoint)
    if annotation["epoch"] != FINAL_EPOCH:
        raise ValueError(
            f"Annotation checkpoint epoch is {annotation['epoch']}, not {FINAL_EPOCH}."
        )
    if annotation["num_repeat_classes"] <= 0:
        raise ValueError("Annotation checkpoint has no repeat head; this is a four-head model.")
    if annotation["label_mode"] != "sparse_set_softmax_with_instance_heads":
        raise ValueError("Annotation checkpoint has the wrong label mode.")
    if annotation["specific_class_loss"] != "uniform_soft_target_cross_entropy":
        raise ValueError("Annotation checkpoint has the wrong soft-target loss.")
    head_details = [head_metadata(path) for path in args.pathogenicity_heads]
    epochs = [details["epoch"] for details in head_details]
    if epochs != EXPECTED_HEAD_EPOCHS:
        raise ValueError(f"Expected selected head epochs {EXPECTED_HEAD_EPOCHS}, found {epochs}.")
    feature_dimensions = {details["feature_dim"] for details in head_details}
    if feature_dimensions != {EXPECTED_FEATURE_DIM}:
        raise ValueError(f"Unexpected pathogenicity feature dimensions: {feature_dimensions}")

    artifacts = [(args.annotation_checkpoint, ANNOTATION_FILENAME)] + list(
        zip(args.pathogenicity_heads, HEAD_FILENAMES)
    )
    manifest = {
        "model": "PALM overlap-aware six-head, epoch 3 (post-ep3 final)",
        "annotation_checkpoint": annotation,
        "pathogenicity_ensemble": {
            "split": "gene/protein-disjoint five-fold ClinVar",
            "aggregation": "unweighted arithmetic mean of probabilities",
            "selected_epochs": EXPECTED_HEAD_EPOCHS,
            "feature_dim": EXPECTED_FEATURE_DIM,
            "pooled_out_of_fold": {
                "n": 151823,
                "roc_auc": 0.9229,
                "pr_auc_average_precision": 0.8669,
            },
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
        if args.replace_legacy:
            print("\nWould also delete these four-head release files:")
            for filename in LEGACY_FILENAMES:
                print(f"  {filename}")
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
    if args.replace_legacy:
        existing = set(
            api.list_repo_files(repo_id=args.repo_id, repo_type="model", revision=args.revision)
        )
        for filename in LEGACY_FILENAMES:
            if filename not in existing:
                print(f"Legacy file already absent: {filename}")
                continue
            print(f"Deleting legacy {args.repo_id}/{filename}", flush=True)
            api.delete_file(
                repo_id=args.repo_id,
                repo_type="model",
                revision=args.revision,
                path_in_repo=filename,
                commit_message=f"Remove four-head release file {filename}",
            )
    print(f"Uploaded PALM checkpoints to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

