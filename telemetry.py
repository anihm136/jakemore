"""TensorBoard diagnostics logger for training dynamics in jakemore."""

import os
from typing import Any, Optional
import jax.numpy as jnp
import numpy as np
from tensorboardX import SummaryWriter

from layers import BatchNorm1d, Sequential


def compute_diagnostics(
    model: Sequential,
    params: list[dict[str, jnp.ndarray]],
    state: list[dict[str, jnp.ndarray]],
    grads: list[dict[str, jnp.ndarray]],
    x_batch: jnp.ndarray,
    lr: float,
) -> dict[str, Any]:
    """Computes training diagnostics: grad-to-weight, update-to-weight, and saturation."""
    scalars: dict[str, float] = {}
    grouped_scalars: dict[str, dict[str, float]] = {
        "grad_to_weight": {},
        "update_to_weight_log10": {},
        "weight_std": {},
        "grad_std": {},
        "grad_mean": {},
        "activation_std": {},
        "activation_mean": {},
        "tanh_saturation_pct": {},
        "batchnorm_running_mean": {},
        "batchnorm_running_var": {},
    }
    histograms: dict[str, np.ndarray] = {}

    eps = 1e-12

    # 1. Parameter and gradient statistics
    for idx, (layer, p, g) in enumerate(zip(model.layers, params, grads)):
        layer_name = f"{layer.__class__.__name__}_{idx}"
        for param_name, w in p.items():
            if param_name not in g:
                continue
            grad = g[param_name]
            w_np = np.asarray(w)
            g_np = np.asarray(grad)

            w_std = float(np.std(w_np))
            g_std = float(np.std(g_np))
            g_mean = float(np.mean(g_np))

            grad_to_weight = g_std / (w_std + eps)
            update_std = lr * g_std
            ud = update_std / (w_std + eps)
            ud_log10 = float(np.log10(ud + eps))

            key = f"{layer_name}_{param_name}"
            grouped_scalars["grad_to_weight"][key] = grad_to_weight
            grouped_scalars["update_to_weight_log10"][key] = ud_log10
            grouped_scalars["weight_std"][key] = w_std
            grouped_scalars["grad_std"][key] = g_std
            grouped_scalars["grad_mean"][key] = g_mean

            scalars[f"grad_to_weight/{key}"] = grad_to_weight
            scalars[f"update_to_weight_log10/{key}"] = ud_log10
            scalars[f"weight_std/{key}"] = w_std
            scalars[f"grad_std/{key}"] = g_std
            scalars[f"grad_mean/{key}"] = g_mean

            histograms[f"weights/{key}"] = w_np
            histograms[f"gradients/{key}"] = g_np

    # 2. Activation statistics & Tanh saturation
    _, intermediates = model.forward_with_intermediates(
        params, state, x_batch, training=False
    )
    for layer_name, act in intermediates:
        act_np = np.asarray(act)
        act_std = float(np.std(act_np))
        act_mean = float(np.mean(act_np))

        grouped_scalars["activation_std"][layer_name] = act_std
        grouped_scalars["activation_mean"][layer_name] = act_mean
        scalars[f"activation_std/{layer_name}"] = act_std
        scalars[f"activation_mean/{layer_name}"] = act_mean
        histograms[f"activations/{layer_name}"] = act_np

        if "Tanh" in layer_name:
            sat_pct = float(np.mean(np.abs(act_np) > 0.99) * 100.0)
            grouped_scalars["tanh_saturation_pct"][layer_name] = sat_pct
            scalars[f"tanh_saturation_pct/{layer_name}"] = sat_pct

    # 3. BatchNorm running statistics
    for idx, (layer, s) in enumerate(zip(model.layers, state)):
        if isinstance(layer, BatchNorm1d) and "running_mean" in s:
            layer_name = f"BatchNorm1d_{idx}"
            running_mean = float(np.mean(np.asarray(s["running_mean"])))
            running_var = float(np.mean(np.asarray(s["running_var"])))
            grouped_scalars["batchnorm_running_mean"][layer_name] = running_mean
            grouped_scalars["batchnorm_running_var"][layer_name] = running_var
            scalars[f"batchnorm/{layer_name}_running_mean"] = running_mean
            scalars[f"batchnorm/{layer_name}_running_var"] = running_var

    return {
        "scalars": scalars,
        "grouped_scalars": grouped_scalars,
        "histograms": histograms,
    }


class TensorBoardLogger:
    """Manages TensorBoard logging for jakemore training metrics."""

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag_or_dict: Any,
        tag_scalar_dict: Optional[dict[str, float]] = None,
        step: Optional[int] = None,
    ):
        """Logs scalar data.

        Supports:
          log_scalars(main_tag: str, tag_scalar_dict: dict[str, float], step: int)
          log_scalars(tag_scalar_dict: dict[str, float], step: int)
        """
        if isinstance(main_tag_or_dict, str):
            assert tag_scalar_dict is not None and step is not None
            self.writer.add_scalars(main_tag_or_dict, tag_scalar_dict, step)
        else:
            assert isinstance(main_tag_or_dict, dict)
            actual_step = tag_scalar_dict if isinstance(tag_scalar_dict, int) else step
            assert actual_step is not None
            for tag, val in main_tag_or_dict.items():
                self.writer.add_scalar(tag, val, actual_step)

    def log_multiline_scalars(
        self, main_tag: str, tag_scalar_dict: dict[str, float], step: int
    ):
        """Logs multiple series to a single plot under main_tag."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histograms(self, tag_array_dict: dict[str, np.ndarray], step: int):
        for tag, arr in tag_array_dict.items():
            self.writer.add_histogram(tag, arr, step)

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        diagnostics: Optional[dict[str, Any]] = None,
        log_histograms: bool = False,
    ):
        """Logs training step scalars, learning rate, and diagnostics in one plot per metric."""
        self.writer.add_scalar("loss/train", loss, step)
        self.writer.add_scalar("lr/value", lr, step)

        if diagnostics is not None:
            if "grouped_scalars" in diagnostics:
                for main_tag, series_dict in diagnostics["grouped_scalars"].items():
                    if series_dict:
                        self.log_multiline_scalars(main_tag, series_dict, step)
            elif "scalars" in diagnostics:
                # Group flat keys of the form 'metric/series' to plot all layers in one plot
                grouped: dict[str, dict[str, float]] = {}
                flat: dict[str, float] = {}
                for tag, val in diagnostics["scalars"].items():
                    if "/" in tag:
                        metric, series = tag.split("/", 1)
                        grouped.setdefault(metric, {})[series] = val
                    else:
                        flat[tag] = val
                for metric, series_dict in grouped.items():
                    self.log_multiline_scalars(metric, series_dict, step)
                for tag, val in flat.items():
                    self.writer.add_scalar(tag, val, step)

            if log_histograms and "histograms" in diagnostics:
                self.log_histograms(diagnostics["histograms"], step)

    def close(self):
        self.writer.close()
