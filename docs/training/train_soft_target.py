#!/usr/bin/env python3
"""Train the final eight-epoch PALM soft-target annotation model.

Expected FASTA headers contain an optional whitespace-delimited field such as
``InterPro=IPR000001:10-40;IPR000002:35-60``. Coordinates are one-based and
inclusive. Multiple intervals/classes may cover the same residue.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from palm.model import PALM, sparse_uniform_target_loss


INTERPRO_PATTERN = re.compile(r"(IPR\d{6}):(\d+)-(\d+)")


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if distributed() else 1


def main_process() -> bool:
    return rank() == 0


def log(message: str, path: Path | None) -> None:
    if not main_process():
        return
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if path is not None:
        with path.open("a") as handle:
            handle.write(line + "\n")


def setup_distributed() -> torch.device:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=120),
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_maps(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    domain: dict[str, int] = {}
    site: dict[str, int] = {}
    with path.open() as handle:
        next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed InterPro row: {line[:100]}")
            ipr, annotation_type = fields[:2]
            target = domain if annotation_type == "Domain" else site
            if ipr not in target:
                target[ipr] = len(target)
    if not domain or not site:
        raise ValueError(f"Empty domain/site map from {path}")
    return domain, site


def sparse_sets(sets: Sequence[set[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max([len(value) for value in sets] + [1])
    labels = torch.full((len(sets), width), -100, dtype=torch.long)
    occurrence = torch.zeros(len(sets), dtype=torch.long)
    for index, values in enumerate(sets):
        if values:
            ordered = sorted(values)
            labels[index, : len(ordered)] = torch.tensor(ordered)
            occurrence[index] = 1 if len(ordered) == 1 else 2
    return labels, occurrence


def residue_labels(
    length: int,
    interpro: str,
    domain_map: dict[str, int],
    site_map: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    domains = [set() for _ in range(length)]
    sites = [set() for _ in range(length)]
    for ipr, start_text, end_text in INTERPRO_PATTERN.findall(interpro):
        start = max(0, int(start_text) - 1)
        end = min(length - 1, int(end_text) - 1)
        if end < start:
            continue
        target, label_map = (domains, domain_map) if ipr in domain_map else (sites, site_map)
        if ipr not in label_map:
            continue
        label = label_map[ipr]
        for position in range(start, end + 1):
            target[position].add(label)
    domain_sets, domain_occurrence = sparse_sets(domains)
    site_sets, site_occurrence = sparse_sets(sites)
    return domain_sets, site_sets, domain_occurrence, site_occurrence


def stream_fasta(path: str) -> Iterator[tuple[str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].strip(), []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def header_interpro(header: str) -> str:
    index = header.find("InterPro=")
    return "" if index < 0 else header[index + len("InterPro=") :].split(None, 1)[0]


@dataclass
class Sample:
    sequence: str
    domain_sets: torch.Tensor
    site_sets: torch.Tensor
    domain_occurrence: torch.Tensor
    site_occurrence: torch.Tensor


class FastaShardDataset(IterableDataset):
    def __init__(
        self,
        files: Sequence[str],
        domain_map: dict[str, int],
        site_map: dict[str, int],
        max_len: int,
        drop_unannotated: bool,
        shuffle_files: bool,
        seed: int,
        epoch: int,
    ) -> None:
        self.files = list(files)
        self.domain_map = domain_map
        self.site_map = site_map
        self.max_len = int(max_len)
        self.drop_unannotated = bool(drop_unannotated)
        self.shuffle_files = bool(shuffle_files)
        self.seed = int(seed)
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[Sample]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        global_worker = rank() * workers + worker_id
        global_workers = world_size() * workers
        files = list(self.files)
        if self.shuffle_files:
            random.Random(self.seed + self.epoch).shuffle(files)
        for file_index, path in enumerate(files):
            if file_index % global_workers != global_worker:
                continue
            for header, sequence in stream_fasta(path):
                sequence = sequence[: self.max_len]
                interpro = header_interpro(header)
                if not sequence or (self.drop_unannotated and not interpro):
                    continue
                yield Sample(
                    sequence,
                    *residue_labels(
                        len(sequence), interpro, self.domain_map, self.site_map
                    ),
                )


def pad_sets(items: Sequence[torch.Tensor]) -> torch.Tensor:
    length = max(item.shape[0] for item in items)
    width = max(item.shape[1] for item in items)
    output = torch.full((len(items), length, width), -100, dtype=torch.long)
    for index, item in enumerate(items):
        output[index, : item.shape[0], : item.shape[1]] = item
    return output


def collate(samples: Sequence[Sample]) -> dict[str, object]:
    return {
        "sequences": [sample.sequence for sample in samples],
        "domain_sets": pad_sets([sample.domain_sets for sample in samples]),
        "site_sets": pad_sets([sample.site_sets for sample in samples]),
        "domain_occurrence": pad_sequence(
            [sample.domain_occurrence for sample in samples],
            batch_first=True,
            padding_value=-100,
        ),
        "site_occurrence": pad_sequence(
            [sample.site_occurrence for sample in samples],
            batch_first=True,
            padding_value=-100,
        ),
    }


def match_length(values: torch.Tensor, length: int) -> torch.Tensor:
    if values.shape[1] == length:
        return values
    if values.shape[1] > length:
        return values[:, :length]
    shape = list(values.shape)
    shape[1] = length - values.shape[1]
    return torch.cat([values, torch.full(shape, -100, dtype=values.dtype)], dim=1)


def batch_loss(
    output,
    batch: dict[str, object],
    device: torch.device,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    length = output.residue_mask.shape[1]
    domain_occurrence = match_length(batch["domain_occurrence"], length).to(device)
    site_occurrence = match_length(batch["site_occurrence"], length).to(device)
    domain_sets = match_length(batch["domain_sets"], length).to(device)
    site_sets = match_length(batch["site_sets"], length).to(device)

    domain_occurrence_sum = F.cross_entropy(
        output.domain_occurrence_logits.float().reshape(-1, 3),
        domain_occurrence.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    site_occurrence_sum = F.cross_entropy(
        output.site_occurrence_logits.float().reshape(-1, 3),
        site_occurrence.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    domain_class_sum, domain_count = sparse_uniform_target_loss(
        output.domain_class_logits, domain_sets
    )
    site_class_sum, site_count = sparse_uniform_target_loss(
        output.site_class_logits, site_sets
    )
    domain_occurrence_count = domain_occurrence.ne(-100).sum()
    site_occurrence_count = site_occurrence.ne(-100).sum()
    components = {
        "domain_occurrence": domain_occurrence_sum / domain_occurrence_count.clamp_min(1),
        "site_occurrence": site_occurrence_sum / site_occurrence_count.clamp_min(1),
        "domain_class": domain_class_sum / domain_count.clamp_min(1),
        "site_class": site_class_sum / site_count.clamp_min(1),
    }
    loss = sum(weights[key] * value for key, value in components.items())
    return loss, {key: float(value.detach()) for key, value in components.items()}


def occurrence_counts(logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    length = logits.shape[1]
    truth = match_length(truth, length).to(logits.device)
    valid = truth.ne(-100)
    predicted = logits.argmax(dim=-1).ne(0)
    positive = truth.ne(0)
    return torch.tensor(
        [
            int((predicted & positive & valid).sum()),
            int((predicted & ~positive & valid).sum()),
            int((~predicted & positive & valid).sum()),
        ],
        dtype=torch.float64,
        device=logits.device,
    )


def precision_recall_f1(counts: torch.Tensor) -> tuple[float, float, float]:
    tp, fp, fn = [float(value) for value in counts.cpu().tolist()]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def run_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    scaler,
    accumulation_steps: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    batches = 0
    domain_counts = torch.zeros(3, dtype=torch.float64, device=device)
    site_counts = torch.zeros(3, dtype=torch.float64, device=device)
    for step, batch in enumerate(loader, 1):
        with torch.set_grad_enabled(training):
            output = model(batch["sequences"], amp=amp)
            loss, _ = batch_loss(output, batch, device, weights)
            if training:
                scaler.scale(loss / accumulation_steps).backward()
                if step % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.detach())
        batches += 1
        domain_counts += occurrence_counts(
            output.domain_occurrence_logits, batch["domain_occurrence"]
        )
        site_counts += occurrence_counts(
            output.site_occurrence_logits, batch["site_occurrence"]
        )
    if training and batches % accumulation_steps:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    totals = torch.cat(
        [torch.tensor([loss_sum, batches], device=device), domain_counts, site_counts]
    )
    if distributed():
        dist.all_reduce(totals)
    domain_prf = precision_recall_f1(totals[2:5])
    site_prf = precision_recall_f1(totals[5:8])
    return {
        "loss": float(totals[0] / totals[1].clamp_min(1)),
        "domain_precision": domain_prf[0],
        "domain_recall": domain_prf[1],
        "domain_f1": domain_prf[2],
        "site_precision": site_prf[0],
        "site_recall": site_prf[1],
        "site_f1": site_prf[2],
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--ipr-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--esm-checkpoint", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-last-n-layers", type=int, default=4)
    parser.add_argument("--domain-occurrence-weight", type=float, default=1.0)
    parser.add_argument("--site-occurrence-weight", type=float, default=1.0)
    parser.add_argument("--domain-class-weight", type=float, default=1.0)
    parser.add_argument("--site-class-weight", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--keep-unannotated", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    device = setup_distributed()
    random.seed(args.seed + rank())
    torch.manual_seed(args.seed + rank())
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training.log"
    if main_process():
        log_path.write_text("")

    train_files = sorted(glob.glob(str(args.train_dir / "*.fasta")))
    validation_files = sorted(glob.glob(str(args.val_dir / "*.fasta")))
    if not train_files or not validation_files:
        raise FileNotFoundError("No .fasta shards found in train/validation directory.")
    domain_map, site_map = load_label_maps(args.ipr_tsv)
    model = PALM(
        args.esm_checkpoint,
        len(domain_map),
        len(site_map),
        max_len_residues=args.max_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_last_n_layers=args.lora_last_n_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.learning_rate
    )
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if amp and "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
    wrapped = (
        DistributedDataParallel(
            model, device_ids=[device.index], broadcast_buffers=False
        )
        if distributed()
        else model
    )
    weights = {
        "domain_occurrence": args.domain_occurrence_weight,
        "site_occurrence": args.site_occurrence_weight,
        "domain_class": args.domain_class_weight,
        "site_class": args.site_class_weight,
    }
    metadata = {
        "architecture": "overlap_aware_multi_task",
        "label_mode": "sparse_set_softmax_with_instance_heads",
        "specific_class_loss": "uniform_soft_target_cross_entropy",
        "num_domain_classes": len(domain_map),
        "num_site_classes": len(site_map),
        "ipr2domain": domain_map,
        "ipr2site": site_map,
        "esm_checkpoint": args.esm_checkpoint,
        "max_len": args.max_len,
        "use_all_layer_pooling": True,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_last_n_layers": args.lora_last_n_layers,
        "lora_target_modules": ["query", "key", "value", "dense"],
        "domain_instance_weight": args.domain_occurrence_weight,
        "site_instance_weight": args.site_occurrence_weight,
        "domain_class_weight": args.domain_class_weight,
        "site_class_weight": args.site_class_weight,
    }
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": collate,
        "persistent_workers": args.workers > 0,
    }
    log(
        f"training_files={len(train_files)} validation_files={len(validation_files)} "
        f"domain_classes={len(domain_map)} site_classes={len(site_map)}",
        log_path,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        datasets = [
            FastaShardDataset(
                files,
                domain_map,
                site_map,
                args.max_len,
                not args.keep_unannotated,
                shuffle,
                args.seed,
                epoch,
            )
            for files, shuffle in ((train_files, True), (validation_files, False))
        ]
        train_loader = DataLoader(datasets[0], drop_last=True, **loader_options)
        validation_loader = DataLoader(datasets[1], drop_last=False, **loader_options)
        train_metrics = run_epoch(
            wrapped,
            train_loader,
            device,
            amp,
            weights,
            optimizer,
            scaler,
            max(1, args.gradient_accumulation),
        )
        validation_metrics = run_epoch(
            wrapped,
            validation_loader,
            device,
            amp,
            weights,
            None,
            scaler,
            1,
        )
        log(f"epoch={epoch} train={train_metrics}", log_path)
        log(f"epoch={epoch} validation={validation_metrics}", log_path)
        if main_process():
            unwrapped = wrapped.module if distributed() else wrapped
            torch.save(
                {
                    **metadata,
                    "epoch": epoch,
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "train_metrics": train_metrics,
                    "validation_metrics": validation_metrics,
                },
                args.output_dir / f"epoch_{epoch}.pt",
            )
    if distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
