import pytest
import torch
from torch import nn

from coldstart.data.schema import SLOTS
from coldstart.models.base_dnn import BaseDNN
from coldstart.models.generators.attribute import MetaGenerator


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
    assert BaseDNN(slots=iter(slots)).feature_width == model.feature_width
    assert model.mlp[0].in_features == 90
    features = torch.zeros((3, 18), dtype=torch.long)
    features[:, -2:] = torch.tensor([12, 13])
    logits = model(features)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_movie_embedding_replaces_only_movie_slot():
    # Reorder slots so the test also checks that replacement follows the slot name.
    model = BaseDNN(slots=(("movie", 1), ("user", 1), ("other", 2)), vocab_size=10,
                    embedding_dim=1).requires_grad_(False)
    with torch.no_grad():
        model.embedding.weight[:, 0] = torch.arange(10)
        model.embedding.weight[0, 0] = 10
        for layer in model.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
        model.mlp[0].weight[0] = torch.tensor([1, 2, 3])
        model.mlp[2].weight[0, 0] = 1
        model.mlp[4].weight[0, 0] = 1

    features = torch.tensor([[1, 2, 3, 0], [4, 1, 1, 2]])
    movie_vectors = model.embedding(features[:, 0]).requires_grad_()
    logits = model(features, movie_embedding=movie_vectors)
    torch.testing.assert_close(logits, model(features), rtol=0, atol=0)
    features[:, 0] = 7
    torch.testing.assert_close(model(features, movie_embedding=movie_vectors),
                               torch.tensor([14., 15.]), rtol=0, atol=0)
    logits.sum().backward()
    torch.testing.assert_close(movie_vectors.grad, torch.ones_like(movie_vectors))
    assert all(param.grad is None for param in model.parameters())
    for wrong_shape in ((2,), (1, 1), (2, 2)):
        with pytest.raises(ValueError, match="movie_embedding"):
            model(features, movie_embedding=torch.zeros(wrong_shape))


def test_attribute_initial_uses_masked_item_slots_and_signed_calibration():
    embedding = nn.Embedding(12, 1, padding_idx=0).requires_grad_(False)
    with torch.no_grad():
        embedding.weight[:, 0] = torch.arange(12)
        embedding.weight[0, 0] = 99
    attributes = torch.tensor([[1, 3, 0, 4, 0, 0, 2, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    generator = MetaGenerator(1, "attribute")
    assert torch.equal(generator.initial(attributes, embedding), torch.zeros(2, 1))
    adapted = torch.tensor([[-1.], [2.]])
    histories = torch.tensor([[2.], [0.]])
    assert torch.equal(generator.calibrate(adapted, histories), adapted)
    with torch.no_grad():
        generator.attribute.weight.copy_(torch.tensor([[-1., 2., 3.]]))
        generator.calibration.weight.fill_(-3)
    assert torch.equal(generator.initial(attributes, embedding), torch.tensor([[19.], [0.]]))
    assert torch.equal(generator.calibrate(adapted, histories), torch.tensor([[-7.], [2.]]))
    with pytest.raises(ValueError, match="attributes"):
        generator.initial(torch.zeros(2, 16, dtype=torch.long), embedding)
    with pytest.raises(ValueError, match="history"):
        generator.calibrate(adapted, histories[:1])


def test_shared_initial_starts_at_zero_and_learns_across_movies():
    embedding = nn.Embedding(12, 2).requires_grad_(False)
    attributes = torch.tensor([[0] * 11, [5] * 11])
    generator = MetaGenerator(2, "shared")
    initial = generator.initial(attributes, embedding)
    assert torch.equal(initial, torch.zeros(2, 2))
    initial.sum().backward()
    shared, = generator.initial_parameters()
    assert torch.equal(shared.grad, torch.full_like(shared, 2.0))
    assert generator.calibration.weight.grad is None
