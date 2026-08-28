import jax
import jax.numpy as jnp
import numpy as np
import pytest

from common import (
    PAD_CHAR,
    VOCAB_SIZE,
    build_tokenized_dataset,
    create_grain_loader,
    itos,
    load_names,
    stoi,
)
import ngrams
import nn


def test_tokenization_roundtrip():
    chars = [PAD_CHAR] + [chr(ord("a") + i) for i in range(26)]
    assert len(chars) == VOCAB_SIZE
    for c in chars:
        token = stoi(c)
        assert 0 <= token < VOCAB_SIZE
        assert itos(token) == c


def test_build_tokenized_dataset():
    sample_names = ["em", "bob"]
    ngram_size = 3
    xs, ys = build_tokenized_dataset(sample_names, ngram_size=ngram_size)

    # For "em": tokens = [., ., e, m, .] -> transitions: (.., e), (.e, m), (em, .) -> 3 transitions
    # For "bob": tokens = [., ., b, o, b, .] -> transitions: (.., b), (.b, o), (bo, b), (ob, .) -> 4 transitions
    # Total = 7
    assert xs.shape == (7, ngram_size - 1)
    assert ys.shape == (7,)
    assert xs.dtype == jnp.int32
    assert ys.dtype == jnp.int32

    # First example for "em" should be context (., .) -> (0, 0) and target 'e' -> stoi('e')
    assert (xs[0] == jnp.array([0, 0])).all()
    assert ys[0] == stoi("e")


def test_grain_loader():
    xs = jnp.arange(20).reshape(10, 2)
    ys = jnp.arange(10)
    batch_size = 4

    loader = create_grain_loader(xs, ys, batch_size=batch_size, seed=0, repeat=False, shuffle=False)
    batches = list(loader)

    assert len(batches) == 3  # 4, 4, 2
    assert batches[0]["x"].shape == (4, 2)
    assert batches[0]["y"].shape == (4,)
    assert batches[2]["x"].shape == (2, 2)


def test_ngram_counts_and_smoothing():
    sample_names = ["emma", "olivia"]
    ngram_size = 3
    xs, ys = build_tokenized_dataset(sample_names, ngram_size=ngram_size)

    counts = ngrams.fit_ngram_counts(xs, ys, ngram_size=ngram_size)
    assert counts.shape == (VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE)
    assert float(counts.sum()) == len(xs)

    smoothed = counts + 1.0
    probs = smoothed / smoothed.sum(axis=-1, keepdims=True)
    assert probs.shape == (VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-6)

    nll, perp = ngrams.evaluate_nll(jnp.log(probs), xs, ys, ngram_size=ngram_size)
    assert nll > 0.0
    assert perp >= 1.0


def test_ngram_sampling():
    sample_names = ["emma", "olivia", "ava"]
    ngram_size = 3
    xs, ys = build_tokenized_dataset(sample_names, ngram_size=ngram_size)
    counts = ngrams.fit_ngram_counts(xs, ys, ngram_size=ngram_size) + 1.0
    probs = counts / counts.sum(axis=-1, keepdims=True)

    key = jax.random.key(42)
    names = ngrams.sample(probs, key, ngram_size=ngram_size, num_samples=5, max_length=15)
    assert len(names) == 5
    assert all(isinstance(n, str) for n in names)
    assert all(len(n) <= 15 for n in names)


def test_mlp_param_shapes():
    key = jax.random.key(0)

    # Embedding table mode
    model_emb = nn.init_mlp_params(key, ngram_size=3, emb_size=10, hidden_dim=64)
    assert "emb" in model_emb
    assert model_emb["emb"].shape == (VOCAB_SIZE, 10)
    assert model_emb["W1"].shape == (2 * 10, 64)
    assert model_emb["b1"].shape == (64,)
    assert model_emb["W2"].shape == (64, VOCAB_SIZE)
    assert model_emb["b2"].shape == (VOCAB_SIZE,)

    # One-hot mode
    model_onehot = nn.init_mlp_params(key, ngram_size=3, emb_size=None, hidden_dim=64)
    assert "emb" not in model_onehot
    assert model_onehot["W1"].shape == (2 * VOCAB_SIZE, 64)


def test_mlp_forward_and_vmap():
    key = jax.random.key(0)
    model = nn.init_mlp_params(key, ngram_size=3, emb_size=10, hidden_dim=64)

    # Single example forward
    x_single = jnp.array([0, 5], dtype=jnp.int32)
    logits_single = nn.forward(x_single, model)
    assert logits_single.shape == (VOCAB_SIZE,)

    # Batched forward
    x_batch = jnp.array([[0, 5], [5, 13], [13, 0]], dtype=jnp.int32)
    logits_batch = nn.compute_batch_logits(x_batch, model)
    assert logits_batch.shape == (3, VOCAB_SIZE)


def test_mlp_train_step_decreases_loss():
    key = jax.random.key(0)
    model = nn.init_mlp_params(key, ngram_size=3, emb_size=10, hidden_dim=64)

    x_batch = jnp.array([[0, 5], [5, 13], [13, 1], [1, 0]], dtype=jnp.int32)
    y_batch = jnp.array([5, 13, 1, 0], dtype=jnp.int32)

    _, initial_loss = nn.train_step(model, x_batch, y_batch, lr=0.1, reg_weight=0.0)

    cur_model = model
    for _ in range(30):
        cur_model, final_loss = nn.train_step(cur_model, x_batch, y_batch, lr=0.2, reg_weight=0.0)

    assert float(final_loss) < float(initial_loss)


def test_mlp_sampling():
    key = jax.random.key(0)
    model = nn.init_mlp_params(key, ngram_size=3, emb_size=10, hidden_dim=64)

    sample_key = jax.random.key(42)
    names = nn.sample_nn(model, sample_key, ngram_size=3, num_samples=5, max_length=15)
    assert len(names) == 5
    assert all(isinstance(n, str) for n in names)
    assert all(len(n) <= 15 for n in names)
