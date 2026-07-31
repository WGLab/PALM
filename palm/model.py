"""PALM soft-target overlap-aware residue annotation model.

The classes in this module reproduce the architecture stored in the final
``epoch_8.pt`` checkpoint.  The model has two occurrence heads (none, one, or
multiple annotations) and two annotation-class heads (InterPro domain and site
classes).  Overlapping class targets are represented by a uniform probability
distribution over every valid class at a residue.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel, AutoTokenizer


class ScalarMix(nn.Module):
    """Learned weighted average of ESM hidden layers."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(tuple(hidden_states), dim=0)
        weights = torch.softmax(self.layer_weights, dim=0)
        return self.gamma * (weights[:, None, None, None] * stacked).sum(dim=0)


class ResidueBiLSTMEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else int(embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(x)
        return encoded * mask.unsqueeze(-1)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim),
            nn.ReLU(),
            nn.LayerNorm(4 * in_dim),
            nn.Linear(4 * in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PALMOutput:
    domain_occurrence_logits: torch.Tensor
    site_occurrence_logits: torch.Tensor
    domain_class_logits: torch.Tensor
    site_class_logits: torch.Tensor
    residue_mask: torch.Tensor
    residue_embeddings: torch.Tensor


class PALM(nn.Module):
    """ESM2-LoRA + BiLSTM model used by the final PALM checkpoint."""

    def __init__(
        self,
        esm_checkpoint: str,
        num_domain_classes: int,
        num_site_classes: int,
        max_len_residues: int = 4096,
        lstm_hidden_dim: int | None = None,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        use_all_layer_pooling: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_last_n_layers: int = 4,
        lora_target_modules: Sequence[str] | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            esm_checkpoint, local_files_only=local_files_only
        )
        self.use_all_layer_pooling = bool(use_all_layer_pooling)
        base_esm = AutoModel.from_pretrained(
            esm_checkpoint,
            output_hidden_states=self.use_all_layer_pooling,
            local_files_only=local_files_only,
        )
        self.esm_config = base_esm.config
        self.embed_dim = int(base_esm.config.hidden_size)
        self.max_len_residues = int(max_len_residues)

        for parameter in base_esm.parameters():
            parameter.requires_grad_(False)
        if not hasattr(base_esm, "encoder") or not hasattr(base_esm.encoder, "layer"):
            raise RuntimeError("Expected an ESM model with esm.encoder.layer modules.")

        n_layers = len(base_esm.encoder.layer)
        targets = list(lora_target_modules or ["query", "key", "value", "dense"])
        layers = list(range(max(0, n_layers - int(lora_last_n_layers)), n_layers))
        if not layers:
            raise ValueError("lora_last_n_layers selected no ESM layers.")
        peft_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            target_modules=targets,
            layers_to_transform=layers,
            layers_pattern="layer",
        )
        self.esm = get_peft_model(base_esm, peft_config)
        self.n_layers = n_layers
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_last_n_layers = int(lora_last_n_layers)
        self.lora_target_modules = targets

        if self.use_all_layer_pooling:
            self.scalar_mix = ScalarMix(n_layers + 1)
        self.encoder = ResidueBiLSTMEncoder(
            self.embed_dim, lstm_hidden_dim, lstm_layers, lstm_dropout
        )
        output_dim = self.encoder.out_dim
        self.domain_instance_head = MLPHead(output_dim, 3)
        self.site_instance_head = MLPHead(output_dim, 3)
        self.domain_head = MLPHead(output_dim, int(num_domain_classes))
        self.site_type_head = MLPHead(output_dim, int(num_site_classes))

    def _sequence_cap(self) -> int:
        maximum = getattr(self.esm_config, "max_position_embeddings", None)
        if isinstance(maximum, int) and maximum > 0:
            return min(self.max_len_residues, max(1, maximum - 2))
        return self.max_len_residues

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def encode(self, sequences: Sequence[str], amp: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded PALM residue embeddings and a valid-residue mask."""
        if not sequences:
            raise ValueError("At least one protein sequence is required.")
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._sequence_cap() + 2,
        )
        tokens = {key: value.to(device, non_blocking=True) for key, value in tokens.items()}
        use_amp = bool(amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = self.esm(**tokens)
            hidden = (
                self.scalar_mix(outputs.hidden_states)
                if self.use_all_layer_pooling
                else outputs.last_hidden_state
            )
        lengths = (tokens["attention_mask"].sum(dim=1) - 2).clamp(min=0)
        longest = int(lengths.max().item())
        positions = torch.arange(longest, device=device).unsqueeze(0)
        mask = (positions < lengths.unsqueeze(1)).float()
        residues = hidden[:, 1 : 1 + longest, :]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            encoded = self.encoder(residues, mask)
        return encoded, mask

    def forward(self, sequences: Sequence[str], amp: bool = True) -> PALMOutput:
        encoded, mask = self.encode(sequences, amp=amp)
        use_amp = bool(amp and encoded.device.type == "cuda")
        with torch.autocast(device_type=encoded.device.type, enabled=use_amp):
            domain_occurrence = self.domain_instance_head(encoded) * mask.unsqueeze(-1)
            site_occurrence = self.site_instance_head(encoded) * mask.unsqueeze(-1)
            domain_class = self.domain_head(encoded) * mask.unsqueeze(-1)
            site_class = self.site_type_head(encoded) * mask.unsqueeze(-1)
        return PALMOutput(
            domain_occurrence,
            site_occurrence,
            domain_class,
            site_class,
            mask,
            encoded,
        )


def model_from_checkpoint(
    checkpoint: dict[str, object],
    *,
    max_len: int | None = None,
    local_files_only: bool = False,
) -> PALM:
    """Construct PALM from checkpoint metadata without loading its weights."""
    return PALM(
        esm_checkpoint=str(checkpoint["esm_checkpoint"]),
        num_domain_classes=int(checkpoint["num_domain_classes"]),
        num_site_classes=int(checkpoint["num_site_classes"]),
        max_len_residues=int(max_len or checkpoint.get("max_len", 4096)),
        lstm_hidden_dim=checkpoint.get("lstm_hidden_dim"),
        lstm_layers=int(checkpoint.get("lstm_layers", 1)),
        lstm_dropout=float(checkpoint.get("lstm_dropout", 0.0)),
        use_all_layer_pooling=bool(checkpoint.get("use_all_layer_pooling", True)),
        lora_r=int(checkpoint.get("lora_r", 8)),
        lora_alpha=int(checkpoint.get("lora_alpha", 16)),
        lora_dropout=float(checkpoint.get("lora_dropout", 0.05)),
        lora_last_n_layers=int(checkpoint.get("lora_last_n_layers", 4)),
        lora_target_modules=checkpoint.get(
            "lora_target_modules", ["query", "key", "value", "dense"]
        ),
        local_files_only=local_files_only,
    )


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "auto",
    max_len: int | None = None,
    local_files_only: bool = False,
    require_epoch8: bool = True,
) -> tuple[PALM, dict[str, object]]:
    """Load the final PALM model and return it with its checkpoint metadata."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # The training checkpoint also contains optimizer state and is about 2 GB.
    # Memory mapping prevents those tensors from being materialized during
    # inference and substantially lowers peak host-memory use.
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("label_mode") != "sparse_set_softmax_with_instance_heads":
        raise ValueError("Checkpoint is not a PALM soft-target overlap-aware checkpoint.")
    if checkpoint.get("specific_class_loss") != "uniform_soft_target_cross_entropy":
        raise ValueError("Checkpoint was not trained with the final uniform soft-target loss.")
    if require_epoch8 and int(checkpoint.get("epoch", -1)) != 8:
        raise ValueError(f"Expected epoch 8, found epoch {checkpoint.get('epoch')}.")
    model = model_from_checkpoint(
        checkpoint, max_len=max_len, local_files_only=local_files_only
    )
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}; unexpected={unexpected}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model.to(resolved_device).eval()
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key
        not in {
            "model_state_dict",
            "optimizer_state_dict",
            "scaler_state_dict",
        }
    }
    del checkpoint
    return model, metadata


def occurrence_probability(logits: torch.Tensor) -> torch.Tensor:
    """Probability that at least one annotation occurs: P(one) + P(multiple)."""
    return torch.softmax(logits.float(), dim=-1)[..., 1:].sum(dim=-1)


def sparse_uniform_target_loss(
    logits: torch.Tensor, label_sets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed soft-target loss and number of annotated residue targets.

    ``label_sets`` has shape ``[batch, residue, max_labels]`` with ``-100``
    padding.  The loss is the average negative log probability of every true
    class at an annotated residue.
    """
    valid = label_sets.ge(0)
    annotated = valid.any(dim=-1)
    count = annotated.sum()
    if not bool(count):
        return logits.sum() * 0.0, count
    safe_labels = label_sets.clamp_min(0)
    selected = torch.gather(
        torch.log_softmax(logits.float(), dim=-1), -1, safe_labels
    )
    per_residue = -(selected * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
    return per_residue[annotated].sum(), count


def freeze(model: nn.Module) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()
