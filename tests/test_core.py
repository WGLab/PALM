from pathlib import Path

import torch

from palm.hub import hf_reference, parse_hf_reference
from palm.model import sparse_uniform_target_loss
from palm.pathogenicity import PathogenicityHead, crop_around_position
from palm.predictor import mutate_sequence


def test_mutate_sequence_uses_one_based_positions():
    assert mutate_sequence("ACDE", 2, "W") == "AWDE"


def test_crop_retains_mutation_position():
    crop, position, start = crop_around_position("A" * 20, 18, 8)
    assert len(crop) == 8
    assert start + position == 18


def test_uniform_target_loss_prefers_probability_shared_across_true_classes():
    labels = torch.tensor([[[0, 1]]])
    uniform_logits = torch.tensor([[[1.0, 1.0, -8.0]]])
    single_logits = torch.tensor([[[8.0, -8.0, -8.0]]])
    uniform_loss, count = sparse_uniform_target_loss(uniform_logits, labels)
    single_loss, _ = sparse_uniform_target_loss(single_logits, labels)
    assert count.item() == 1
    assert uniform_loss < single_loss


def test_pathogenicity_head_shape():
    head = PathogenicityHead(12, [8, 4], 0.1)
    assert head(torch.zeros(3, 12)).shape == (3,)


def test_hugging_face_reference_round_trip():
    reference = hf_reference("example/PALM", "pathogenicity/fold_1.pt")
    assert reference == "hf://example/PALM::pathogenicity/fold_1.pt"
    assert parse_hf_reference(reference) == (
        "example/PALM",
        "pathogenicity/fold_1.pt",
    )


def test_label_maps_split_repeat_out_of_sites(tmp_path):
    """Repeat entries must land in their own vocabulary, not in the site head.

    In the four-head release every InterPro type that was not ``Domain`` fell
    into the site map, which is how 392 Repeat classes ended up in a site
    vocabulary that never trained on them.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))
    from training.train_soft_target import load_label_maps

    table = tmp_path / "ipr.tsv"
    table.write_text(
        "ipr\ttype\tname\n"
        "IPR000001\tDomain\tKringle\n"
        "IPR000002\tRepeat\tArmadillo\n"
        "IPR000003\tActive_site\tSerine protease\n"
        "IPR000004\tConserved_site\tSomething conserved\n"
    )
    domain, site, repeat = load_label_maps(table)
    assert list(domain) == ["IPR000001"]
    assert list(repeat) == ["IPR000002"]
    assert list(site) == ["IPR000003", "IPR000004"]


def test_palm_output_carries_six_tracks():
    from palm.model import PALMOutput

    assert PALMOutput.__dataclass_fields__.keys() >= {
        "domain_occurrence_logits",
        "site_occurrence_logits",
        "repeat_occurrence_logits",
        "domain_class_logits",
        "site_class_logits",
        "repeat_class_logits",
    }


def test_bundled_class_table_covers_three_heads():
    import pandas as pd

    from palm.predictor import BUNDLED_CLASS_TABLE

    table = pd.read_csv(BUNDLED_CLASS_TABLE, sep="\t")
    counts = table.groupby("head").size().to_dict()
    assert counts == {"domain": 17951, "site": 991, "repeat": 392}
