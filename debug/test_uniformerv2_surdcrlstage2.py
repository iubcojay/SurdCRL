"""Standalone test script for surdcrl/model/uniformerv2_surdcrlstage2.py

It builds the orcausal model, creates random inputs (video, boxes, back,
prototype/global dicts), profiles FLOPs/Params with thop, computes per-layer
FLOPs via fvcore, runs one forward pass, and prints latency and output shape.

Usage:
  python test_uniformerv2_surdcrlstage2.py --num_class 20 --batch 1 --frames 8 --height 224 --width 224 --objects 4 --globalN 512 --bgN 20 --device auto
"""

import argparse
import time
import torch
import sys
import os

# Add parent directory to Python path to import surdcrl module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_inputs(batch: int, frames: int, height: int, width: int, objects: int, globalN: int, bgN: int, device: torch.device):
    # Video tensors [B, 3, T, H, W]
    video = torch.randn(batch, 3, frames, height, width, device=device)
    # Boxes [B, O, T, 4]
    boxes = torch.randint(low=0, high=height, size=(batch, objects, frames, 4), device=device)
    # Background frames [B, 3, T, H, W]
    back = torch.randn(batch, 3, frames, height, width, device=device)
    # Global prototype dictionary [N, 768]
    globaldict = torch.randn(globalN, 768, device=device)
    # Background confounder dictionary [bgN, 768]
    bg_confounder_dictionary = torch.randn(bgN, 768, device=device)
    # Background prior [bgN, 1]
    bg_prior = torch.rand(bgN, 1, device=device)
    return video, boxes, back, globaldict, bg_confounder_dictionary, bg_prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_class", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--objects", type=int, default=4)
    parser.add_argument("--globalN", type=int, default=512)
    parser.add_argument("--bgN", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Import model
    from surdcrl.model.uniformerv2_surdcrlstage2 import orcausal

    # Build model
    model = orcausal(num_class=args.num_class).to(device)
    model.eval()

    # Build inputs
    video, boxes, back, globaldict, bg_conf, bg_prior = build_inputs(
        args.batch, args.frames, args.height, args.width, args.objects, args.globalN, args.bgN, device
    )

    # Optional FLOPs/Params profiling with thop
    try:
        from thop import profile, clever_format
        gflops, params = profile(model, inputs=(video, boxes, back, globaldict, bg_conf, bg_prior))
        gflops = gflops / 1e9
        gflops, params = clever_format([gflops, params], "%.3f")
        print(f"Total FLOPs: {gflops} GFLOPs")
        print(f"Total Params: {params} params")
    except Exception as e:
        print(f"[Profile] Skipped THOP profiling: {e}")

    # Per-layer FLOPs by fvcore (optional)
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        flops = FlopCountAnalysis(model, (video, boxes, back, globaldict, bg_conf, bg_prior))
        print(flop_count_table(flops, max_depth=2))
    except Exception as e:
        print(f"[Fvcore] Skipped flop table: {e}")

    # Forward once and measure latency
    with torch.no_grad():
        t0 = time.time()
        out = model(video, boxes, back, globaldict, bg_conf, bg_prior)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.time()

    # Report
    if isinstance(out, torch.Tensor):
        print(f"Output tensor shape: {tuple(out.shape)}")
    elif isinstance(out, (list, tuple)):
        shapes = [tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__ for x in out]
        print(f"Output (list/tuple) shapes: {shapes}")
    else:
        print(f"Output type: {type(out).__name__}")

    print(f"Latency: {(t1 - t0) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
