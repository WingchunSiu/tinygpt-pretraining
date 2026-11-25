import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.metrics.open() as file:
        records = [json.loads(line) for line in file if line.strip()]

    train = [record for record in records if record["type"] == "train"]
    validation = [record for record in records if record["type"] == "validation"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if train:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            [record["step"] for record in train],
            [record["loss"] for record in train],
            label="Training",
        )
        if validation:
            axis.plot(
                [record["step"] for record in validation],
                [record["loss"] for record in validation],
                marker="o",
                label="Validation",
            )
        axis.set(xlabel="Training step", ylabel="Cross-entropy loss")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.output_dir / "loss.png", dpi=160)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            [record["step"] for record in train],
            [record["tokens_per_second"] for record in train],
        )
        axis.set(xlabel="Training step", ylabel="Total tokens per second")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.output_dir / "throughput.png", dpi=160)
        plt.close(figure)

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
