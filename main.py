import os
import pickle
import jax
import bigrams
import nn


def load_or_train_models(save_dir: str = "models", data_path: str = "names.txt"):
    bigram_path = os.path.join(save_dir, "bigram_model.pkl")
    nn_path = os.path.join(save_dir, "nn_model.pkl")

    if not os.path.exists(bigram_path):
        print(f"'{bigram_path}' not found. Training bigram count model...")
        bigrams.main(bigrams.BigramConfig(data_path=data_path, save_dir=save_dir))

    if not os.path.exists(nn_path):
        print(f"'{nn_path}' not found. Training neural network model...")
        nn.main(nn.NNConfig(data_path=data_path, save_dir=save_dir))

    with open(bigram_path, "rb") as f:
        bigram_model = pickle.load(f)

    with open(nn_path, "rb") as f:
        nn_model = pickle.load(f)

    return bigram_model, nn_model


def main():
    print("=" * 65)
    print("  jakemore: Model Evaluation & Generation (Saved Models)")
    print("=" * 65)

    bigram_model, nn_model = load_or_train_models(save_dir="models")

    # 1. Print Loss & Perplexity Comparison Table
    print("\n" + "=" * 65)
    print(f"  {'Metric':<30} | {'Count Model':<14} | {'Neural Network':<14}")
    print("  " + "-" * 61)
    print(f"  {'Negative Log-Likelihood (NLL)':<30} | {bigram_model['nll']:<14.4f} | {nn_model['nll']:<14.4f}")
    print(f"  {'Perplexity':<30} | {bigram_model['perplexity']:<14.4f} | {nn_model['perplexity']:<14.4f}")
    print("  " + "=" * 65)

    # 2. Generate 5 Random Names from Each Model
    key = jax.random.key(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    count_samples = bigrams.sample(
        bigram_model["probs"],
        subkey1,
        num_samples=5,
    )
    nn_samples = nn.sample_nn(
        nn_model["W"],
        subkey2,
        num_samples=5,
    )

    print("\nGenerated Sample Names (5 from each model):")
    print("  " + "-" * 61)
    print(f"  {'#':<4} | {'Count Model':<26} | {'Neural Network':<26}")
    print("  " + "-" * 61)
    for idx, (c_name, n_name) in enumerate(zip(count_samples, nn_samples), start=1):
        print(f"  {idx:<4} | {c_name:<26} | {n_name:<26}")
    print("  " + "-" * 61 + "\n")


if __name__ == "__main__":
    main()
