"""Unit tests for telemetry.py and TensorBoard diagnostics."""

import os
import shutil
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from layers import BatchNorm1d, Embedding, Flatten, Linear, Sequential, Tanh
from telemetry import TensorBoardLogger, compute_diagnostics


def test_forward_with_intermediates():
    key = jax.random.key(0)
    model = Sequential([
        Embedding(vocab_size=27, emb_dim=8),
        Flatten(),
        Linear(in_features=16, out_features=32, bias=False),
        BatchNorm1d(num_features=32),
        Tanh(),
        Linear(in_features=32, out_features=27, bias=True),
    ])
    params, state = model.init(key)

    x = jnp.ones((4, 2), dtype=jnp.int32)
    out, intermediates = model.forward_with_intermediates(params, state, x, training=False)

    assert out.shape == (4, 27)
    assert len(intermediates) == 6
    assert intermediates[0][0] == "Embedding_0"
    assert intermediates[0][1].shape == (4, 2, 8)
    assert intermediates[1][0] == "Flatten_1"
    assert intermediates[1][1].shape == (4, 16)
    assert intermediates[4][0] == "Tanh_4"
    assert intermediates[4][1].shape == (4, 32)


def test_compute_diagnostics_ratios_and_metrics():
    key = jax.random.key(0)
    model = Sequential([
        Linear(in_features=4, out_features=4, bias=True),
        Tanh(),
    ])
    params, state = model.init(key)

    # Set mock weights and gradients with known std
    params[0]["w"] = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # std = 0.5
    params[0]["b"] = jnp.zeros(2)
    grads = [
        {
            "w": jnp.array([[0.01, 0.0], [0.0, 0.01]]),  # std = 0.005
            "b": jnp.zeros(2),
        },
        {},
    ]
    # Update model layers to match 2x2
    model.layers[0].in_features = 2
    model.layers[0].out_features = 2

    x = jnp.ones((2, 2))
    lr = 0.1
    diag = compute_diagnostics(model, params, state, grads, x, lr)

    scalars = diag["scalars"]
    histograms = diag["histograms"]

    assert "grad_to_weight/Linear_0_w" in scalars
    assert "update_to_weight_log10/Linear_0_w" in scalars

    # Expected: g_std = 0.005, w_std = 0.5 -> grad_to_weight = 0.01
    np.testing.assert_allclose(scalars["grad_to_weight/Linear_0_w"], 0.01, atol=1e-5)

    # Expected: ud = lr * g_std / w_std = 0.1 * 0.01 = 0.001 -> log10(0.001) = -3.0
    np.testing.assert_allclose(scalars["update_to_weight_log10/Linear_0_w"], -3.0, atol=1e-4)

    # Verify grouped_scalars mapping
    grouped = diag["grouped_scalars"]
    assert "grad_to_weight" in grouped
    assert "update_to_weight_log10" in grouped
    assert "Linear_0_w" in grouped["grad_to_weight"]
    np.testing.assert_allclose(grouped["grad_to_weight"]["Linear_0_w"], 0.01, atol=1e-5)
    np.testing.assert_allclose(grouped["update_to_weight_log10"]["Linear_0_w"], -3.0, atol=1e-4)

    # Histograms present
    assert "weights/Linear_0_w" in histograms
    assert "gradients/Linear_0_w" in histograms
    assert "activations/Tanh_1" in histograms


def test_tanh_saturation_detection():
    key = jax.random.key(0)
    model = Sequential([
        Linear(in_features=2, out_features=2, bias=False),
        Tanh(),
    ])
    params, state = model.init(key)
    grads = [{"w": jnp.zeros((2, 2))}, {}]

    # Extreme weights causing complete saturation (|h| > 0.99)
    params[0]["w"] = jnp.ones((2, 2)) * 100.0
    x = jnp.ones((4, 2))
    diag_saturated = compute_diagnostics(model, params, state, grads, x, lr=0.1)
    sat_pct = diag_saturated["scalars"]["tanh_saturation_pct/Tanh_1"]
    np.testing.assert_allclose(sat_pct, 100.0, atol=1e-4)
    sat_pct_grouped = diag_saturated["grouped_scalars"]["tanh_saturation_pct"]["Tanh_1"]
    np.testing.assert_allclose(sat_pct_grouped, 100.0, atol=1e-4)

    # Tiny weights causing zero saturation
    params[0]["w"] = jnp.zeros((2, 2))
    diag_unsat = compute_diagnostics(model, params, state, grads, x, lr=0.1)
    unsat_pct = diag_unsat["scalars"]["tanh_saturation_pct/Tanh_1"]
    np.testing.assert_allclose(unsat_pct, 0.0, atol=1e-4)
    unsat_pct_grouped = diag_unsat["grouped_scalars"]["tanh_saturation_pct"]["Tanh_1"]
    np.testing.assert_allclose(unsat_pct_grouped, 0.0, atol=1e-4)


def test_tensorboard_logger_writes_event_files():
    temp_dir = tempfile.mkdtemp()
    try:
        logger = TensorBoardLogger(log_dir=temp_dir)
        logger.log_scalar("test/loss", 1.23, step=1)
        logger.log_scalars({"metric/a": 0.5, "metric/b": 0.2}, step=1)
        logger.log_histograms({"hist/w": np.ones((10,))}, step=1)
        logger.log_step(
            step=2,
            loss=1.10,
            lr=0.05,
            diagnostics={
                "scalars": {"update_to_weight_log10/test": -3.0},
                "histograms": {"weights/test": np.zeros(5)},
            },
            log_histograms=True,
        )
        logger.close()

        files = os.listdir(temp_dir)
        assert len(files) >= 1
        # Root event file or subdirectories with events
        all_files = []
        for root, _, fs in os.walk(temp_dir):
            all_files.extend(fs)
        assert any("tfevents" in f for f in all_files)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_diagnostics_logged_in_one_plot_per_metric():
    """Verifies that diagnostic metrics are grouped so all layers share one plot."""
    from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer

    temp_dir = tempfile.mkdtemp()
    try:
        logger = TensorBoardLogger(log_dir=temp_dir)
        diag = {
            "grouped_scalars": {
                "grad_to_weight": {
                    "Linear_0_w": 0.01,
                    "Linear_2_w": 0.02,
                },
                "update_to_weight_log10": {
                    "Linear_0_w": -3.0,
                    "Linear_2_w": -3.1,
                },
            }
        }
        logger.log_step(step=1, loss=2.5, lr=0.1, diagnostics=diag)
        logger.close()

        mux = EventMultiplexer()
        mux.AddRunsFromDirectory(temp_dir)
        mux.Reload()

        runs = mux.Runs()
        # Find scalar tag names across all runs
        charts = {}
        for run, run_info in runs.items():
            for tag in run_info.get("scalars", []):
                charts.setdefault(tag, []).append(run)

        # In TensorBoard, all layers for grad_to_weight must be under the same tag
        assert "grad_to_weight" in charts
        assert len(charts["grad_to_weight"]) == 2
        assert any("Linear_0_w" in r for r in charts["grad_to_weight"])
        assert any("Linear_2_w" in r for r in charts["grad_to_weight"])

        assert "update_to_weight_log10" in charts
        assert len(charts["update_to_weight_log10"]) == 2
        assert any("Linear_0_w" in r for r in charts["update_to_weight_log10"])
        assert any("Linear_2_w" in r for r in charts["update_to_weight_log10"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_flat_scalars_grouped_into_one_plot_per_metric():
    """Verifies that flat scalars formatted as 'metric/series' are also grouped per metric."""
    from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer

    temp_dir = tempfile.mkdtemp()
    try:
        logger = TensorBoardLogger(log_dir=temp_dir)
        diag = {
            "scalars": {
                "grad_to_weight/Linear_0_w": 0.01,
                "grad_to_weight/Linear_2_w": 0.02,
                "update_to_weight_log10/Linear_0_w": -3.0,
                "update_to_weight_log10/Linear_2_w": -3.1,
            }
        }
        logger.log_step(step=1, loss=2.5, lr=0.1, diagnostics=diag)
        logger.close()

        mux = EventMultiplexer()
        mux.AddRunsFromDirectory(temp_dir)
        mux.Reload()

        runs = mux.Runs()
        charts = {}
        for run, run_info in runs.items():
            for tag in run_info.get("scalars", []):
                charts.setdefault(tag, []).append(run)

        assert "grad_to_weight" in charts
        assert len(charts["grad_to_weight"]) == 2
        assert any("Linear_0_w" in r for r in charts["grad_to_weight"])
        assert any("Linear_2_w" in r for r in charts["grad_to_weight"])

        assert "update_to_weight_log10" in charts
        assert len(charts["update_to_weight_log10"]) == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


