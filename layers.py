"""Modular neural network layers for jakemore in pure JAX."""

from typing import Any, Protocol, Sequence
import jax
import jax.numpy as jnp


class Layer(Protocol):
    """Functional layer protocol for JAX."""

    def init(
        self, key: jax.Array
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        """Initializes trainable parameters and non-trainable state buffers."""
        ...

    def __call__(
        self,
        params: dict[str, jnp.ndarray],
        state: dict[str, jnp.ndarray],
        x: jnp.ndarray,
        **kwargs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Transforms input x given params and state."""
        ...


class Embedding:
    """Lookup table embedding character tokens to continuous vectors."""

    def __init__(self, vocab_size: int, emb_dim: int, weight_scale: float = 1.0):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.weight_scale = weight_scale

    def init(
        self, key: jax.Array
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        weight = (
            jax.random.normal(key, (self.vocab_size, self.emb_dim)) * self.weight_scale
        )
        return {"weight": weight}, {}

    def __call__(
        self,
        params: dict[str, jnp.ndarray],
        state: dict[str, jnp.ndarray],
        x: jnp.ndarray,
        **kwargs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return params["weight"][x], state


class OneHot:
    """One-hot categorical encoding layer."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def init(
        self, key: jax.Array
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        return {}, {}

    def __call__(
        self,
        params: dict[str, jnp.ndarray],
        state: dict[str, jnp.ndarray],
        x: jnp.ndarray,
        **kwargs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return jax.nn.one_hot(x, self.num_classes, dtype=jnp.float32), state


class Flatten:
    """Flattens trailing feature dimensions into a 1D vector per batch element."""

    def init(
        self, key: jax.Array
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        return {}, {}

    def __call__(
        self,
        params: dict[str, jnp.ndarray],
        state: dict[str, jnp.ndarray],
        x: jnp.ndarray,
        **kwargs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        if x.ndim >= 2:
            return x.reshape(x.shape[:-2] + (-1,)), state
        return x.ravel(), state


class Linear:
    """Affine linear transformation layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_scale: float = 1.0,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_scale = weight_scale

    def init(
        self, key: jax.Array
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        w = (
            jax.random.normal(key, (self.in_features, self.out_features))
            * self.weight_scale
        )
        params = {"w": w}
        if self.bias:
            params["b"] = jnp.zeros((self.out_features,), dtype=jnp.float32)
        return params, {}

    def __call__(
        self,
        params: dict[str, jnp.ndarray],
        state: dict[str, jnp.ndarray],
        x: jnp.ndarray,
        **kwargs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        out = x @ params["w"]
        if "b" in params:
            out = out + params["b"]
        return out, state


class BatchNorm1d:
    """Batch Normalization layer with momentum-based running statistics."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.001,
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

    def init(
        self, key: jax.Array
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        params = {
            "gamma": jnp.ones((self.num_features,), dtype=jnp.float32),
            "beta": jnp.zeros((self.num_features,), dtype=jnp.float32),
        }
        state = {
            "running_mean": jnp.zeros((self.num_features,), dtype=jnp.float32),
            "running_var": jnp.ones((self.num_features,), dtype=jnp.float32),
        }
        return params, state

    def __call__(
        self,
        params: dict[str, jnp.ndarray],
        state: dict[str, jnp.ndarray],
        x: jnp.ndarray,
        training: bool = True,
        **kwargs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        gamma = params["gamma"]
        beta = params["beta"]

        if training:
            reduction_axes = tuple(range(x.ndim - 1))
            batch_mean = jnp.mean(x, axis=reduction_axes)
            batch_var = jnp.var(x, axis=reduction_axes)

            running_mean = (1.0 - self.momentum) * state[
                "running_mean"
            ] + self.momentum * batch_mean
            running_var = (1.0 - self.momentum) * state[
                "running_var"
            ] + self.momentum * batch_var
            new_state = {"running_mean": running_mean, "running_var": running_var}

            x_norm = (x - batch_mean) / jnp.sqrt(batch_var + self.eps)
            out = gamma * x_norm + beta
            return out, new_state
        else:
            x_norm = (x - state["running_mean"]) / jnp.sqrt(
                state["running_var"] + self.eps
            )
            out = gamma * x_norm + beta
            return out, state


class Tanh:
    """Hyperbolic tangent activation function."""

    def init(
        self, key: jax.Array
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        return {}, {}

    def __call__(
        self,
        params: dict[str, jnp.ndarray],
        state: dict[str, jnp.ndarray],
        x: jnp.ndarray,
        **kwargs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return jnp.tanh(x), state


class Sequential:
    """Sequential composition of modular layers."""

    def __init__(self, layers: Sequence[Any]):
        self.layers = list(layers)

    def init(
        self, key: jax.Array
    ) -> tuple[list[dict[str, jnp.ndarray]], list[dict[str, jnp.ndarray]]]:
        params, state = [], []
        for layer in self.layers:
            key, subkey = jax.random.split(key)
            p, s = layer.init(subkey)
            params.append(p)
            state.append(s)
        return params, state

    def __call__(
        self,
        params: list[dict[str, jnp.ndarray]],
        state: list[dict[str, jnp.ndarray]],
        x: jnp.ndarray,
        training: bool = True,
    ) -> tuple[jnp.ndarray, list[dict[str, jnp.ndarray]]]:
        new_states = []
        h = x
        for layer, p, s in zip(self.layers, params, state):
            h, new_s = layer(p, s, h, training=training)
            new_states.append(new_s)
        return h, new_states

    def param_count(self, params: list[dict[str, jnp.ndarray]]) -> int:
        """Returns total number of scalar parameters in the model."""
        return sum(p.size for p in jax.tree.leaves(params))


def calibrate_batchnorm_stats(
    params: list[dict[str, jnp.ndarray]],
    state: list[dict[str, jnp.ndarray]],
    model: Sequential,
    data_batches: Sequence[jnp.ndarray],
) -> list[dict[str, jnp.ndarray]]:
    """Calculates exact population mean and variance for all BatchNorm layers over dataset."""
    new_state = [dict(s) for s in state]

    for layer_idx, layer in enumerate(model.layers):
        if not isinstance(layer, BatchNorm1d):
            continue

        # Forward data up to current BatchNorm input
        all_features = []
        for batch in data_batches:
            h = batch
            for prev_idx in range(layer_idx):
                h, _ = model.layers[prev_idx](
                    params[prev_idx], state[prev_idx], h, training=False
                )
            all_features.append(h)

        combined = jnp.concatenate(all_features, axis=0)
        reduction_axes = tuple(range(combined.ndim - 1))
        pop_mean = jnp.mean(combined, axis=reduction_axes)
        pop_var = jnp.var(combined, axis=reduction_axes)

        new_state[layer_idx]["running_mean"] = pop_mean
        new_state[layer_idx]["running_var"] = pop_var

    return new_state
