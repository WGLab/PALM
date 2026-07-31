"""Resolve PALM model artifacts from local storage or Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


HF_PREFIX = "hf://"
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


def resolve_artifact(
    reference: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return a local path for a local file or Hugging Face artifact."""
    text = str(reference)
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
    path = Path(reference).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. For Hugging Face use "
            "hf://organization/repository::filename.pt"
        )
    return path.resolve()
