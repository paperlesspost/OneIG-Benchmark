#!/usr/bin/env python3
"""Print style key and first 10 elements of each embedding vector from a .pt file.

Usage:
    python print_embed.py /path/to/embeddings.pt
"""

import argparse
from pathlib import Path
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Print style key and first 10 components of each embedding vector.")
    parser.add_argument("pt_file", type=Path, help="Path to the .pt embedding file (e.g., CSD_embed.pt)")
    args = parser.parse_args()

    if not args.pt_file.is_file():
        raise FileNotFoundError(f"File not found: {args.pt_file}")

    # Load the checkpoint – it is expected to be a dict of {style_key: Tensor}
    data = torch.load(args.pt_file, map_location="cpu")
    if not isinstance(data, dict) or not data:
        raise ValueError("Expected a non‑empty dictionary mapping style_key to a tensor.")

    # Iterate through all style keys and their tensors
    for style_key, tensor in data.items():
        if not isinstance(tensor, torch.Tensor):
            print(f"⚠️ Skipping {style_key}: not a tensor")
            continue

        print(f"\n{style_key}")
        for idx, row in enumerate(tensor):
            # Use Python's list conversion for nice display
            values = row[:3].tolist()
            print(f"[ {idx} ] ==> {values}")

if __name__ == "__main__":
    main()
