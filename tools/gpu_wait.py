#!/usr/bin/env python3
"""Wait until the local CUDA device can run a tiny torch operation."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="seconds to wait; 0 waits forever")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def nvidia_smi_ok() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return False, f"nvidia-smi failed: {exc!r}"
    if result.returncode != 0:
        return False, result.stdout.strip()
    return True, result.stdout.strip()


def torch_cuda_ok(device: int) -> tuple[bool, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False"
        if torch.cuda.device_count() <= device:
            return False, f"cuda device {device} is not present"
        torch.cuda.set_device(device)
        probe = torch.ones((32, 32), device=f"cuda:{device}")
        _ = probe @ probe
        torch.cuda.synchronize()
        return True, torch.cuda.get_device_name(device)
    except Exception as exc:
        return False, repr(exc)


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    attempt = 1

    while True:
        smi_ok, smi_info = nvidia_smi_ok()
        torch_ok, torch_info = torch_cuda_ok(args.device) if smi_ok else (False, smi_info)
        if smi_ok and torch_ok:
            print(f"[gpu_wait] CUDA healthy: {torch_info}", flush=True)
            return 0

        elapsed = time.monotonic() - start
        print(
            f"[gpu_wait] attempt={attempt} elapsed={elapsed:.0f}s "
            f"smi_ok={smi_ok} torch_ok={torch_ok} reason={torch_info}",
            flush=True,
        )
        if args.timeout and elapsed >= args.timeout:
            return 1
        attempt += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
