import torch
from torch import nn

from coldstart.data.schema import SLOTS
from coldstart.models.base_dnn import BaseDNN


def test_base_dnn_slot_pooling_and_logits():
    model = BaseDNN()
    assert model.embedding.weight.shape == (11135, 10)
    linear = [layer for layer in model.mlp if isinstance(layer, nn.Linear)]
    assert [tuple(layer.weight.shape) for layer in linear] == [
        (256, 80), (128, 256), (1, 128)
    ]
    assert all(layer.bias is None for layer in linear)

    # Encode each ID in coordinate 0. A deliberately nonzero row 0 exercises
    # the padding mask even when weights come from an external checkpoint.
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.weight[:11, 0] = torch.arange(11)
        model.embedding.weight[0, 0] = 99
        for layer in linear:
            layer.weight.zero_()
        linear[0].weight[0, ::10] = torch.arange(1, 9)
        linear[0].weight[1, ::10] = -1
        linear[1].weight[0, :2] = 1
        linear[2].weight[0, 0] = -1

    # Slot sums are [1, 2, 0, 4, 5, 6, 15, 19]. Weight by position so
    # swapping (for example) title and genres also changes the expected logit.
    features = torch.tensor([[1, 2, 0, 4, 5, 6, 7, 0, 8, 0, 0, 9, 10, 0, 0, 0]])
    batch = torch.cat([features, torch.zeros_like(features)])
    expected = torch.tensor([-(1 + 4 + 16 + 25 + 36 + 105 + 152), 0.0])
    torch.testing.assert_close(model(batch), expected, rtol=0, atol=0)
    assert model(features).shape == (1,)


def test_base_dnn_uses_extended_slot_schema():
    slots = (*SLOTS, ("semantic_id", 2))
    model = BaseDNN(slots=slots)
    assert model.mlp[0].in_features == 90
    features = torch.zeros((3, 18), dtype=torch.long)
    features[:, -2:] = torch.tensor([12, 13])
    logits = model(features)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
