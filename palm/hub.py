"""Resolve PALM model artifacts from local storage or Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


HF_PREFIX = "hf://"
CHECKPOINT_DIR_ENV = "PALM_CHECKPOINT_DIR"
DEFAULT_HF_REPO_ID = "Wangwpi/PALM"
DEFAULT_ANNOTATION_FILENAME = "palm_soft_target_epoch8.pt"
DEFAULT_ANNOTATION_REFERENCE = (
    f"{HF_PREFIX}{DEFAULT_HF_REPO_ID}::{DEFAULT_ANNOTATION_FILENAME}"
)
DEFAULT_PATHOGENICITY_HEAD_REFERENCES = tuple(
    f"{HF_PREFIX}{DEFAULT_HF_REPO_ID}::pathogenicity/random_split_fold_{fold}.pt"
    for fold in range(1, 6)
)


def hf_reference(repo_id: str, filename: str) -> str:
    """Construct PALM's unambiguous Hugging Face artifact reference."""
    if not repo_id or not filename:
        raise ValueError("Both repo_id and filename are required.")
    return f"{HF_PREFIX}{repo_id}::{filename}"


def parse_hf_reference(reference: str) -> tuple[str, str]:
    """Parse ``hf://organization/repository::path/in/repository.pt``."""
    if not reference.startswith(HF_PREFIX):
        raise ValueError(f"Not a Hugging Face reference: {reference}")
    payload = reference[len(HF_PREFIX) :]
    if "::" not in payload:
        raise ValueError(
            "A Hugging Face reference must use "
            "hf://organization/repository::filename.pt"
        )
    repo_id, filename = payload.split("::", 1)
    if repo_id.count("/") != 1 or not filename:
        raise ValueError(f"Malformed Hugging Face reference: {reference}")
    return repo_id, filename


def checkpoint_dir_candidates(
    reference: str, checkpoint_dir: str | Path | None = None
) -> list[Path]:
    """List local files that could satisfy ``reference`` inside a checkpoint directory.

    The directory is taken from ``checkpoint_dir`` or, when that is ``None``,
    from the ``PALM_CHECKPOINT_DIR`` environment variable. Both the full path
    used inside the Hugging Face repository (``pathogenicity/fold_1.pt``) and
    the bare file name (``fold_1.pt``) are accepted, so a directory populated by
    ``hf download`` and a flat directory of files both work.
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.environ.get(CHECKPOINT_DIR_ENV) or None
    if checkpoint_dir is None:
        return []
    root = Path(checkpoint_dir).expanduser()
    relative = parse_hf_reference(reference)[1] if reference.startswith(HF_PREFIX) else reference
    candidates = [root / relative]
    name = Path(relative).name
    if name != relative:
        candidates.append(root / name)
    return candidates


def resolve_artifact(
    reference: str | Path,
    *,
    checkpoint_dir: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return a local path for a local file or Hugging Face artifact.

    Resolution order: an existing local path is used as-is; otherwise a
    ``checkpoint_dir`` (or ``PALM_CHECKPOINT_DIR``) is searched; otherwise an
    ``hf://`` reference is downloaded from the Hugging Face Hub and cached.
    """
    text = str(reference)
    direct = Path(text).expanduser()
    if not text.startswith(HF_PREFIX) and direct.is_file():
        return direct.resolve()
    for candidate in checkpoint_dir_candidates(text, checkpoint_dir):
        if candidate.is_file():
            return candidate.resolve()
    if text.startswith(HF_PREFIX):
        repo_id, filename = parse_hf_reference(text)
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=None if cache_dir is None else str(cache_dir),
                token=token,
                local_files_only=local_files_only,
            )
        )
    searched = checkpoint_dir_candidates(text, checkpoint_dir)
    locations = ", ".join(str(item) for item in [direct, *searched])
    raise FileNotFoundError(
        f"Checkpoint not found. Looked in: {locations}. Pass a path to a local "
        "file, set a checkpoint directory (--checkpoint-dir or "
        f"{CHECKPOINT_DIR_ENV}), or use hf://organization/repository::filename.pt"
    )
