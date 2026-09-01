import os
import pickle
import time
from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp

from common import (
    VOCAB_SIZE,
    build_tokenized_dataset,
    create_grain_loader,
    itos,
    load_names,
)
from layers import (
    BatchNorm1d,
    Embedding,
    Flatten,
    Linear,
    OneHot,
    Sequential,
    Tanh,
)


@dataclass
class NNConfig:
    """Configuration for the MLP character language model."""

    data_path: str = "names.txt"
    save_dir: str = "models"
    ngram_size: int = 4
    emb_size: Optional[int] = 10  # None for one-hot; int for embedding table
    hidden_dim: int = 200
    learning_rate: float = 0.1
    lr_decay_step: float = 0.8  # Fraction of steps after which LR is decayed
    lr_decay_factor: float = 0.1
    num_steps: int = 10000
    batch_size: int = 32
    reg_weight: float = 0.0  # L2 regularization weight
    use_kaiming: bool = True
    use_batchnorm: bool = True
    bn_momentum: float = 0.001
    bn_eps: float = 1e-5
    train_ratio: float = 0.8
    seed: int = 0
    num_samples: int = 5
    max_length: int = 20
    log_interval: int = 200


def build_mlp(config: NNConfig) -> Sequential:
    """Constructs a modular Sequential MLP model based on NNConfig."""
    context_len = config.ngram_size - 1
    layers = []

    if config.emb_size is not None:
        layers.append(Embedding(VOCAB_SIZE, config.emb_size))
        in_dim = context_len * config.emb_size
    else:
        layers.append(OneHot(VOCAB_SIZE))
        in_dim = context_len * VOCAB_SIZE

    layers.append(Flatten())

    kaiming_scale = (
        (5 / 3) / (in_dim**0.5) if config.use_kaiming else (1.0 / (in_dim**0.5))
    )

    # When followed by BatchNorm, Linear bias is redundant and omitted
    layers.append(
        Linear(
            in_features=in_dim,
            out_features=config.hidden_dim,
            bias=not config.use_batchnorm,
            weight_scale=kaiming_scale,
        )
    )

    if config.use_batchnorm:
        layers.append(
            BatchNorm1d(
                num_features=config.hidden_dim,
                eps=config.bn_eps,
                momentum=config.bn_momentum,
            )
        )

    layers.append(Tanh())

    # Output projection to vocabulary logits
    layers.append(
        Linear(
            in_features=config.hidden_dim,
            out_features=VOCAB_SIZE,
            bias=True,
            weight_scale=0.01,
        )
    )

    return Sequential(layers)


def make_train_step(model: Sequential):
    """Creates a JIT-compiled train step closure for a modular Sequential model."""

    @jax.jit
    def step(
        params: list[dict[str, jnp.ndarray]],
        state: list[dict[str, jnp.ndarray]],
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
        lr: float,
        reg_weight: float,
    ) -> tuple[list[dict[str, jnp.ndarray]], list[dict[str, jnp.ndarray]], jnp.ndarray]:
        def loss_fn(p):
            logits, new_s = model(p, state, x_batch, training=True)
            logprobs = jax.nn.log_softmax(logits, axis=-1)
            target_logprobs = jnp.take_along_axis(logprobs, y_batch[:, None], axis=1)
            nll = -jnp.mean(target_logprobs)
            l2_weights = [
                0.5 * jnp.sum(layer_p["w"] ** 2) / layer_p["w"].size
                for layer_p in p
                if "w" in layer_p
            ]
            l2_loss = sum(l2_weights) if l2_weights else 0.0
            return nll + reg_weight * l2_loss, new_s

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return new_params, new_state, loss

    return step


def evaluate(
    model: Sequential,
    params: list[dict[str, jnp.ndarray]],
    state: list[dict[str, jnp.ndarray]],
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """Evaluates NLL and perplexity over dataset splits in batches."""
    total_nll = 0.0
    n_samples = xs.shape[0]

    for i in range(0, n_samples, batch_size):
        xb = xs[i : i + batch_size]
        yb = ys[i : i + batch_size]
        logits, _ = model(params, state, xb, training=False)
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        target_logprobs = jnp.take_along_axis(logprobs, yb[:, None], axis=1)
        total_nll += float(-jnp.sum(target_logprobs))

    mean_nll = total_nll / n_samples
    perplexity = float(jnp.exp(mean_nll))
    return mean_nll, perplexity


def sample_nn(
    model: Sequential,
    params: list[dict[str, jnp.ndarray]],
    state: list[dict[str, jnp.ndarray]],
    key: jax.Array,
    context_len: int,
    num_samples: int = 10,
    max_length: int = 20,
) -> list[str]:
    """Generates names autoregressively from the network."""
    generated_names = []
    for _ in range(num_samples):
        cur_tokens = [0] * context_len
        out_chars = []
        for _ in range(max_length):
            key, subkey = jax.random.split(key)
            context = jnp.array(cur_tokens[-context_len:], dtype=jnp.int32)
            logits, _ = model(params, state, context, training=False)
            next_token = int(jax.random.categorical(subkey, logits))
            if next_token == 0:
                break
            out_chars.append(itos(next_token))
            cur_tokens.append(next_token)
        generated_names.append("".join(out_chars))
    return generated_names


def main(config: NNConfig = NNConfig()):
    print("=" * 60)
    print(f"  jakemore: {config.ngram_size}-gram MLP Language Model (JAX)")
    print("=" * 60)

    # 1. Load dataset & build tokenized n-grams
    names = load_names(config.data_path)
    xs, ys = build_tokenized_dataset(names, ngram_size=config.ngram_size)

    # Train / test split
    split_idx = int(round(xs.shape[0] * config.train_ratio))
    train_xs, train_ys = xs[:split_idx], ys[:split_idx]
    test_xs, test_ys = xs[split_idx:], ys[split_idx:]

    print(
        f"Loaded {len(names):,} names ({xs.shape[0]:,} {config.ngram_size}-gram transitions)"
    )
    print(f"Train set: {train_xs.shape[0]:,} transitions")
    print(f"Test set:  {test_xs.shape[0]:,} transitions")
    print(
        f"Embedding: {f'{config.emb_size}-dim table' if config.emb_size is not None else 'One-hot'}"
    )
    print(f"Hidden Dim: {config.hidden_dim} (tanh)")
    print(
        f"BatchNorm:  {'Enabled (alpha=' + str(config.bn_momentum) + ')' if config.use_batchnorm else 'Disabled'}"
    )
    print(
        f"LR: {config.learning_rate} (decay {config.lr_decay_factor}x @ {int(config.lr_decay_step * 100)}%) | Steps: {config.num_steps:,} | Batch: {config.batch_size}\n"
    )

    # 2. Build modular model and initialize parameters and state
    model = build_mlp(config)
    init_key = jax.random.key(config.seed)
    init_key, master_key = jax.random.split(init_key)
    params, state = model.init(init_key)
    param_count = model.param_count(params)
    print(f"Model Parameters: {param_count:,}\n")

    # 3. Create JIT-compiled train step
    step_fn = make_train_step(model)

    train_loader = create_grain_loader(
        train_xs,
        train_ys,
        batch_size=config.batch_size,
        seed=config.seed,
        repeat=True,
    )
    train_iter = iter(train_loader)

    print("Training:")
    start = time.perf_counter()
    for step in range(1, config.num_steps + 1):
        batch = next(train_iter)
        xb = jnp.array(batch["x"], dtype=jnp.int32)
        yb = jnp.array(batch["y"], dtype=jnp.int32)

        lr = (
            config.learning_rate
            if (step / config.num_steps) < config.lr_decay_step
            else config.learning_rate * config.lr_decay_factor
        )

        params, state, loss = step_fn(
            params,
            state,
            xb,
            yb,
            lr=lr,
            reg_weight=config.reg_weight,
        )

        if step % config.log_interval == 0 or step == config.num_steps:
            print(
                f"  Step {step:6d}/{config.num_steps:,} | Train Loss: {loss.item():.4f}"
            )

    stop = time.perf_counter()
    train_time = stop - start

    # 4. Evaluate NLL & Perplexity
    train_nll, train_perplexity = evaluate(
        model, params, state, train_xs, train_ys
    )
    test_nll, test_perplexity = evaluate(
        model, params, state, test_xs, test_ys
    )

    print(f"\nTrain Time:         {train_time:.2f} s")
    print(
        f"Train NLL:          {train_nll:.4f} | Train Perplexity: {train_perplexity:.4f}"
    )
    print(
        f"Test NLL:           {test_nll:.4f} | Test Perplexity:  {test_perplexity:.4f}"
    )

    # 5. Generate sample names
    master_key, sample_key = jax.random.split(master_key)
    generated = sample_nn(
        model,
        params,
        state,
        sample_key,
        context_len=config.ngram_size - 1,
        num_samples=config.num_samples,
        max_length=config.max_length,
    )

    print("\nGenerated Names:")
    for idx, name in enumerate(generated, start=1):
        print(f"  {idx:2d}. {name}")

    # 6. Save model checkpoint
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, "nn_model.pkl")
    model_data = {
        "params": params,
        "state": state,
        "config": config,
        "train_nll": train_nll,
        "train_perplexity": train_perplexity,
        "test_nll": test_nll,
        "test_perplexity": test_perplexity,
    }
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to '{save_path}'\n")


if __name__ == "__main__":
    main()
