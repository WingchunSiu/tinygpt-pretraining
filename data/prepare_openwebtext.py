import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    subset = parser.add_mutually_exclusive_group(required=True)
    subset.add_argument("--subset", choices=["10k", "100k"])
    subset.add_argument("--num-documents", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="Skylion007/openwebtext")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    num_documents = 10000 if args.subset == "10k" else 100000
    if args.num_documents is not None:
        num_documents = args.num_documents
    if num_documents < 100:
        raise ValueError("Use at least 100 documents for train/val/test splits")

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    dataset = dataset.take(num_documents)

    train_end = int(num_documents * 0.98)
    val_end = int(num_documents * 0.99)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_counts = {split: 0 for split in ("train", "val", "test")}
    document_counts = {split: 0 for split in token_counts}

    with ExitStack() as stack:
        files = {
            split: stack.enter_context((args.output_dir / f"{split}.bin").open("wb"))
            for split in token_counts
        }
        for index, example in enumerate(dataset):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            tokens = tokenizer.encode(example["text"])
            tokens.append(tokenizer.eot_token)
            array = np.asarray(tokens, dtype=np.uint16)
            array.tofile(files[split])
            token_counts[split] += len(array)
            document_counts[split] += 1
            if (index + 1) % 1000 == 0:
                print(f"Processed {index + 1}/{num_documents} documents", flush=True)

    metadata = {
        "tokenizer": "tiktoken",
        "encoding": "gpt2",
        "vocab_size": tokenizer.n_vocab,
        "block_size": 512,
        "dataset": args.dataset_name,
        "seed": args.seed,
        "num_documents": document_counts,
        "num_tokens": token_counts,
    }
    with (args.output_dir / "meta.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
