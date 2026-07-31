import torch

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

