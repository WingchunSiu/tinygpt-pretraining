import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tiny_shakespeare"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input:
        text = args.input.read_text()
    else:
        with urllib.request.urlopen(DATA_URL) as response:
            text = response.read().decode("utf-8")

    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    tokens = np.array([stoi[char] for char in text], dtype=np.uint16)

    train_end = int(len(tokens) * 0.9)
    val_end = int(len(tokens) * 0.95)
    splits = {
        "train": tokens[:train_end],
        "val": tokens[train_end:val_end],
        "test": tokens[val_end:],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_tokens in splits.items():
        split_tokens.tofile(args.output_dir / f"{split}.bin")

    metadata = {
        "tokenizer": "char",
        "chars": chars,
        "vocab_size": len(chars),
        "block_size": 512,
        "source": str(args.input) if args.input else DATA_URL,
        "num_tokens": {split: len(values) for split, values in splits.items()},
    }
    with (args.output_dir / "meta.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps(metadata["num_tokens"], indent=2))
    print(f"Vocabulary size: {len(chars)}")


if __name__ == "__main__":
    main()
