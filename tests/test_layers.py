"""Unit tests for modular neural network layers in layers.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from layers import (
    BatchNorm1d,
    Embedding,
    Flatten,
    Linear,
    OneHot,
    Sequential,
    Tanh,
    calibrate_batchnorm_stats,
)


def test_linear_shapes_and_bias():
    key = jax.random.key(0)
    lin_bias = Linear(in_features=8, out_features=16, bias=True)
    p_bias, s_bias = lin_bias.init(key)
    assert p_bias["w"].shape == (8, 16)
    assert p_bias["b"].shape == (16,)
    assert s_bias == {}

    lin_nobias = Linear(in_features=8, out_features=16, bias=False)
    p_nobias, s_nobias = lin_nobias.init(key)
    assert p_nobias["w"].shape == (8, 16)
    assert "b" not in p_nobias

    x = jnp.ones((4, 8))
    out, _ = lin_bias(p_bias, s_bias, x)
    assert out.shape == (4, 16)


def test_batchnorm_training_normalization():
    key = jax.random.key(0)
    bn = BatchNorm1d(num_features=4, eps=1e-5, momentum=0.001)
    p, s = bn.init(key)

    # Synthetic batch with non-zero mean and non-unit variance
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (1000, 4)) * 5.0 + 10.0

    out, new_s = bn(p, s, x, training=True)

    # Check batch statistics of normalized output
    np.testing.assert_allclose(jnp.mean(out, axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(jnp.var(out, axis=0), 1.0, atol=1e-4)

    # Verify running stats moved toward batch stats
    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    expected_rm = (1.0 - 0.001) * 0.0 + 0.001 * batch_mean
    expected_rv = (1.0 - 0.001) * 1.0 + 0.001 * batch_var
    np.testing.assert_allclose(new_s["running_mean"], expected_rm, atol=1e-6)
    np.testing.assert_allclose(new_s["running_var"], expected_rv, atol=1e-6)


def test_batchnorm_momentum_ema_tracking():
    key = jax.random.key(0)
    alpha = 0.001
    bn = BatchNorm1d(num_features=2, eps=1e-5, momentum=alpha)
    p, s = bn.init(key)

    x1 = jnp.array([[2.0, 4.0], [4.0, 6.0]])  # mean = [3.0, 5.0], var = [1.0, 1.0]
    _, s1 = bn(p, s, x1, training=True)

    expected_rm1 = (1.0 - alpha) * 0.0 + alpha * jnp.array([3.0, 5.0])
    expected_rv1 = (1.0 - alpha) * 1.0 + alpha * jnp.array([1.0, 1.0])
    np.testing.assert_allclose(s1["running_mean"], expected_rm1, atol=1e-6)
    np.testing.assert_allclose(s1["running_var"], expected_rv1, atol=1e-6)

    x2 = jnp.array([[10.0, 20.0], [20.0, 30.0]])  # mean = [15.0, 25.0], var = [25.0, 25.0]
    _, s2 = bn(p, s1, x2, training=True)

    expected_rm2 = (1.0 - alpha) * expected_rm1 + alpha * jnp.array([15.0, 25.0])
    expected_rv2 = (1.0 - alpha) * expected_rv1 + alpha * jnp.array([25.0, 25.0])
    np.testing.assert_allclose(s2["running_mean"], expected_rm2, atol=1e-6)
    np.testing.assert_allclose(s2["running_var"], expected_rv2, atol=1e-6)


def test_batchnorm_inference_uses_running_stats():
    key = jax.random.key(0)
    bn = BatchNorm1d(num_features=2, eps=1e-5, momentum=0.001)
    p, _ = bn.init(key)

    # Set known frozen running statistics
    s_eval = {
        "running_mean": jnp.array([5.0, 10.0]),
        "running_var": jnp.array([4.0, 9.0]),
    }

    x_test = jnp.array([[5.0, 10.0], [7.0, 16.0]])
    out, out_s = bn(p, s_eval, x_test, training=False)

    # (5 - 5) / sqrt(4 + eps) = 0
    # (7 - 5) / sqrt(4) = 1.0
    # (10 - 10) / sqrt(9) = 0
    # (16 - 10) / sqrt(9) = 2.0
    expected = jnp.array([[0.0, 0.0], [1.0, 2.0]])
    np.testing.assert_allclose(out, expected, atol=1e-3)

    # State must not be updated during inference
    np.testing.assert_allclose(out_s["running_mean"], s_eval["running_mean"])
    np.testing.assert_allclose(out_s["running_var"], s_eval["running_var"])


def test_batchnorm_single_sample_inference():
    key = jax.random.key(0)
    bn = BatchNorm1d(num_features=3, eps=1e-5, momentum=0.001)
    p, s = bn.init(key)

    # 1D single sample (e.g., autoregressive token generation)
    x_single = jnp.array([1.0, 2.0, 3.0])
    out, _ = bn(p, s, x_single, training=False)
    assert out.shape == (3,)
    assert not jnp.isnan(out).any()


def test_batchnorm_gradient_propagation():
    key = jax.random.key(0)
    bn = BatchNorm1d(num_features=3, eps=1e-5, momentum=0.001)
    p, s = bn.init(key)
    x = jax.random.normal(key, (8, 3))

    def dummy_loss(params):
        out, _ = bn(params, s, x, training=True)
        return jnp.sum(out**2)

    grads = jax.grad(dummy_loss)(p)
    assert "gamma" in grads
    assert "beta" in grads
    assert grads["gamma"].shape == (3,)
    assert grads["beta"].shape == (3,)
    assert not jnp.isnan(grads["gamma"]).any()
    assert not jnp.isnan(grads["beta"]).any()


def test_linear_bias_redundancy():
    """Confirms Linear(..., bias=False) followed by BatchNorm is mathematically equivalent."""
    key = jax.random.key(42)
    key_w, key_x, key_bn = jax.random.split(key, 3)

    w = jax.random.normal(key_w, (4, 4))
    b = jnp.array([1.5, -2.0, 3.0, -0.5])
    x = jax.random.normal(key_x, (16, 4))

    bn = BatchNorm1d(num_features=4, eps=1e-5, momentum=0.001)
    p_bn, s_bn = bn.init(key_bn)

    # With bias
    z_biased = x @ w + b
    out_biased, _ = bn(p_bn, s_bn, z_biased, training=True)

    # Without bias
    z_unbiased = x @ w
    out_unbiased, _ = bn(p_bn, s_bn, z_unbiased, training=True)

    np.testing.assert_allclose(out_biased, out_unbiased, atol=1e-5)


def test_calibrate_batchnorm_stats():
    key = jax.random.key(0)
    model = Sequential([
        Linear(in_features=4, out_features=4, bias=False),
        BatchNorm1d(num_features=4, eps=1e-5, momentum=0.001),
    ])
    params, state = model.init(key)

    data_batches = [
        jax.random.normal(jax.random.key(i), (10, 4)) for i in range(5)
    ]
    calibrated_state = calibrate_batchnorm_stats(params, state, model, data_batches)

    # Compute expected population mean/var directly
    all_data = jnp.concatenate(data_batches, axis=0)
    all_z = all_data @ params[0]["w"]
    expected_mean = jnp.mean(all_z, axis=0)
    expected_var = jnp.var(all_z, axis=0)

    np.testing.assert_allclose(calibrated_state[1]["running_mean"], expected_mean, atol=1e-5)
    np.testing.assert_allclose(calibrated_state[1]["running_var"], expected_var, atol=1e-5)


def test_sequential_pipeline_and_jit():
    key = jax.random.key(0)
    model = Sequential([
        Embedding(vocab_size=27, emb_dim=8),
        Flatten(),
        Linear(in_features=16, out_features=32, bias=False),
        BatchNorm1d(num_features=32, eps=1e-5, momentum=0.001),
        Tanh(),
        Linear(in_features=32, out_features=27, bias=True),
    ])
    params, state = model.init(key)

    def loss_fn(p, s, x, y):
        logits, new_s = model(p, s, x, training=True)
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.take_along_axis(logprobs, y[:, None], axis=1))
        return loss, new_s

    @jax.jit
    def train_step(p, s, x, y, lr):
        (loss, new_s), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, s, x, y)
        new_p = jax.tree.map(lambda param, grad: param - lr * grad, p, grads)
        return new_p, new_s, loss

    x = jnp.ones((16, 2), dtype=jnp.int32)
    y = jnp.zeros((16,), dtype=jnp.int32)

    p_cur, s_cur = params, state
    _, _, initial_loss = train_step(p_cur, s_cur, x, y, 0.1)

    for _ in range(10):
        p_cur, s_cur, final_loss = train_step(p_cur, s_cur, x, y, 0.1)

    assert float(final_loss) < float(initial_loss)

    # Test single-token inference
    single_x = jnp.array([1, 2], dtype=jnp.int32)
    logits, _ = model(p_cur, s_cur, single_x, training=False)
    assert logits.shape == (27,)
