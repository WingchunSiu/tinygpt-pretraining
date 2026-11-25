import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from data.dataset import TokenDataset
from data.tokenization import load_metadata
from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device_type = "cuda"
        device = torch.device("cuda", local_rank if distributed else 0)
        if distributed:
            torch.cuda.set_device(device)
            dist.init_process_group("nccl")
    else:
        device_type = "cpu"
        device = torch.device("cpu")
        if distributed:
            dist.init_process_group("gloo")

    return distributed, rank, local_rank, world_size, device_type, device


@torch.no_grad()
def evaluate(model, loader, device, autocast_context, max_batches):
    model.eval()
    loss_sum = 0.0
    item_count = 0
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context():
            _, loss = model(x, y)
        batch_size = x.size(0)
        loss_sum += loss.float().item() * batch_size
        item_count += batch_size
    model.train()
    if item_count == 0:
        raise ValueError("Validation dataset did not produce any batches")
    return loss_sum / item_count


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.log_interval <= 0:
        raise ValueError("epochs, batch_size, and log_interval must be positive")

    distributed, rank, local_rank, world_size, device_type, device = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    metadata = load_metadata(args.data_dir)
    block_size = metadata["block_size"]
    config = GPTConfig(
        vocab_size=metadata["vocab_size"],
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    train_dataset = TokenDataset(args.data_dir, "train", block_size)
    val_dataset = TokenDataset(args.data_dir, "val", block_size)
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
        if distributed
        else None
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device_type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = GPT(config).to(device)
    raw_model = model
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device_type == "cuda" else None,
        )

    optimizer = raw_model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
        device_type,
    )
    amp_enabled = device_type == "cuda" and args.dtype != "float32"
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    def autocast_context():
        if not amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    scaler = torch.amp.GradScaler(
        "cuda", enabled=device_type == "cuda" and args.dtype == "float16"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"

    def write_metric(record):
        if is_main:
            with metrics_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

    def save_checkpoint(step, epoch):
        if not is_main:
            return
        checkpoint = {
            "model": raw_model.state_dict(),
            "model_config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
            "train_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "step": step,
            "epoch": epoch,
        }
        torch.save(checkpoint, args.out_dir / "last.pt")

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    if args.max_iters is not None:
        if args.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        total_steps = min(total_steps, args.max_iters)

    if is_main:
        print(
            json.dumps(
                {
                    "parameters": sum(p.numel() for p in raw_model.parameters()),
                    "device": str(device),
                    "world_size": world_size,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                    "model_config": asdict(config),
                }
            ),
            flush=True,
        )

    step = 0
    last_logged_step = 0
    last_val_step = -1
    window_loss_sum = 0.0
    window_batches = 0
    window_tokens = 0
    window_start = time.perf_counter()
    stop = False
    epoch = 0

    def flush_training_log():
        nonlocal window_loss_sum, window_batches, window_tokens, window_start
        nonlocal last_logged_step
        if window_batches == 0:
            return
        elapsed = time.perf_counter() - window_start
        if distributed:
            stats = torch.tensor(
                [window_loss_sum, window_batches, window_tokens],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total_batches, total_tokens = stats.tolist()
        else:
            total_loss = window_loss_sum
            total_batches = window_batches
            total_tokens = window_tokens

        write_metric(
            {
                "type": "train",
                "step": step,
                "epoch": epoch,
                "loss": total_loss / total_batches,
                "tokens_per_second": total_tokens / elapsed,
                "world_size": world_size,
                "batch_size_per_gpu": args.batch_size,
            }
        )
        last_logged_step = step
        window_loss_sum = 0.0
        window_batches = 0
        window_tokens = 0
        window_start = time.perf_counter()

    while step < total_steps and not stop:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for x, y in train_loader:
            if step >= total_steps:
                stop = True
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context():
                _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            window_loss_sum += loss.detach().float().item()
            window_batches += 1
            window_tokens += x.size(0) * x.size(1)
            if step % args.log_interval == 0:
                flush_training_log()

        if last_logged_step != step:
            flush_training_log()

        if step >= total_steps:
            break

        val_loss = evaluate(
            model, val_loader, device, autocast_context, args.val_batches
        )
        write_metric(
            {
                "type": "validation",
                "step": step,
                "epoch": epoch,
                "loss": val_loss,
                "perplexity": math.exp(val_loss),
            }
        )
        last_val_step = step
        save_checkpoint(step, epoch)
        window_start = time.perf_counter()

    if last_val_step != step:
        val_loss = evaluate(model, val_loader, device, autocast_context, args.val_batches)
        write_metric(
            {
                "type": "validation",
                "step": step,
                "epoch": epoch,
                "loss": val_loss,
                "perplexity": math.exp(val_loss),
            }
        )
        save_checkpoint(step, epoch)

    if is_main:
        print(f"Saved checkpoint to {args.out_dir / 'last.pt'}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
