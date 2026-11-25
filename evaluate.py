import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import TokenDataset
from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = TokenDataset(args.data_dir, args.split, config.block_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    amp_enabled = device.type == "cuda" and args.dtype != "float32"
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    loss_sum = 0.0
    item_count = 0
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with context:
                _, loss = model(x, y)
            loss_sum += loss.float().item() * x.size(0)
            item_count += x.size(0)

    if item_count == 0:
        raise ValueError("Evaluation dataset did not produce any batches")
    mean_loss = loss_sum / item_count
    results = {
        "split": args.split,
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "sequences": item_count,
    }
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
