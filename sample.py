import argparse
from pathlib import Path

import torch

from data.tokenization import decode_tokens, encode_text
from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.temperature <= 0 or args.max_new_tokens <= 0:
        raise ValueError("temperature and max_new_tokens must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metadata = checkpoint["metadata"]
    prompt_tokens = encode_text(args.prompt, metadata)
    if not prompt_tokens:
        raise ValueError("The prompt must produce at least one token")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    top_k = args.top_k if args.top_k > 0 else None
    generated = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
    )
    print(decode_tokens(generated[0].tolist(), metadata))


if __name__ == "__main__":
    main()
