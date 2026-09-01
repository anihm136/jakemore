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


def test_modular_mlp_param_shapes():
    key = jax.random.key(0)

    # Embedding table mode with batchnorm
    config_emb = nn.NNConfig(ngram_size=3, emb_size=10, hidden_dim=64, use_batchnorm=True)
    model_emb = nn.build_mlp(config_emb)
    params_emb, state_emb = model_emb.init(key)

    # Embedding: weight (27, 10)
    assert params_emb[0]["weight"].shape == (VOCAB_SIZE, 10)
    # Flatten: {}
    assert params_emb[1] == {}
    # Linear: w (2 * 10, 64), bias omitted due to batchnorm
    assert params_emb[2]["w"].shape == (2 * 10, 64)
    assert "b" not in params_emb[2]
    # BatchNorm1d: gamma (64,), beta (64,)
    assert params_emb[3]["gamma"].shape == (64,)
    assert params_emb[3]["beta"].shape == (64,)
    assert state_emb[3]["running_mean"].shape == (64,)
    assert state_emb[3]["running_var"].shape == (64,)
    # Tanh: {}
    assert params_emb[4] == {}
    # Output Linear: w (64, 27), b (27,)
    assert params_emb[5]["w"].shape == (64, VOCAB_SIZE)
    assert params_emb[5]["b"].shape == (VOCAB_SIZE,)

    # One-hot mode without batchnorm
    config_onehot = nn.NNConfig(ngram_size=3, emb_size=None, hidden_dim=64, use_batchnorm=False)
    model_onehot = nn.build_mlp(config_onehot)
    params_onehot, _ = model_onehot.init(key)
    # OneHot: {}
    assert params_onehot[0] == {}
    # Flatten: {}
    assert params_onehot[1] == {}
    # Linear: w (2 * 27, 64), b (64,)
    assert params_onehot[2]["w"].shape == (2 * VOCAB_SIZE, 64)
    assert params_onehot[2]["b"].shape == (64,)


def test_modular_mlp_forward():
    key = jax.random.key(0)
    config = nn.NNConfig(ngram_size=3, emb_size=10, hidden_dim=64, use_batchnorm=True)
    model = nn.build_mlp(config)
    params, state = model.init(key)

    # Single example forward
    x_single = jnp.array([0, 5], dtype=jnp.int32)
    logits_single, _ = model(params, state, x_single, training=False)
    assert logits_single.shape == (VOCAB_SIZE,)

    # Batched forward
    x_batch = jnp.array([[0, 5], [5, 13], [13, 0]], dtype=jnp.int32)
    logits_batch, _ = model(params, state, x_batch, training=False)
    assert logits_batch.shape == (3, VOCAB_SIZE)


def test_modular_mlp_batchnorm_construction():
    config_bn = nn.NNConfig(ngram_size=3, emb_size=10, hidden_dim=64, use_batchnorm=True)
    model_bn = nn.build_mlp(config_bn)

    # Layers: Embedding, Flatten, Linear(bias=False), BatchNorm1d, Tanh, Linear(bias=True)
    assert len(model_bn.layers) == 6
    assert isinstance(model_bn.layers[2], nn.Linear)
    assert model_bn.layers[2].bias is False
    assert isinstance(model_bn.layers[3], nn.BatchNorm1d)
    assert model_bn.layers[3].momentum == 0.001

    config_nobn = nn.NNConfig(ngram_size=3, emb_size=10, hidden_dim=64, use_batchnorm=False)
    model_nobn = nn.build_mlp(config_nobn)
    assert len(model_nobn.layers) == 5
    assert model_nobn.layers[2].bias is True


def test_modular_mlp_train_and_sample():
    config = nn.NNConfig(
        ngram_size=3,
        emb_size=10,
        hidden_dim=32,
        use_batchnorm=True,
        bn_momentum=0.001,
        learning_rate=0.1,
    )
    model = nn.build_mlp(config)
    key = jax.random.key(0)
    params, state = model.init(key)

    step_fn = nn.make_train_step(model)

    x_batch = jnp.array([[0, 5], [5, 13], [13, 1], [1, 0]], dtype=jnp.int32)
    y_batch = jnp.array([5, 13, 1, 0], dtype=jnp.int32)

    p_cur, s_cur = params, state
    p_cur, s_cur, initial_loss = step_fn(p_cur, s_cur, x_batch, y_batch, lr=0.1, reg_weight=0.0)

    for _ in range(20):
        p_cur, s_cur, final_loss = step_fn(p_cur, s_cur, x_batch, y_batch, lr=0.2, reg_weight=0.0)

    assert float(final_loss) < float(initial_loss)

    # Test sampling with modular model
    sample_names = nn.sample_nn(
        model,
        p_cur,
        s_cur,
        key,
        context_len=2,
        num_samples=3,
        max_length=10,
    )
    assert len(sample_names) == 3
    assert all(isinstance(n, str) for n in sample_names)

    # Test evaluate with modular model
    mean_nll, perplexity = nn.evaluate(model, p_cur, s_cur, x_batch, y_batch, batch_size=2)
    assert mean_nll > 0.0
    assert perplexity >= 1.0

