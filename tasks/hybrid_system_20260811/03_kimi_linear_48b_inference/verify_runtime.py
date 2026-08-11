#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ELF_MACHINE_AARCH64 = 183
KNOWN_MESSAGE = "The package `nvidia-cusparselt-cu12` was built for a different platform"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    return parser.parse_args()


def aarch64_elf(path: Path) -> bool:
    header = path.read_bytes()[:20]
    return len(header) == 20 and header[:4] == b"\x7fELF" and int.from_bytes(header[18:20], "little") == ELF_MACHINE_AARCH64


def main() -> None:
    args = parse_args()
    result = subprocess.run([str(args.uv), "pip", "check", "--python", str(args.python)], text=True, capture_output=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        print(output)
        return
    lines = [line for line in output.splitlines() if line.startswith("The package `")]
    venv = args.python.absolute().parent.parent
    libraries = list(venv.glob("lib/python*/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0"))
    wheels = list(venv.glob("lib/python*/site-packages/nvidia_cusparselt_cu12-*.dist-info/WHEEL"))
    if len(lines) == 1 and lines[0] == KNOWN_MESSAGE and len(libraries) == 1 and aarch64_elf(libraries[0]) and any(
        "manylinux2014_sbsa" in path.read_text(encoding="utf-8") for path in wheels
    ):
        print(output)
        print("Accepted audited NVIDIA SBSA tag exception for an AArch64 ELF library.")
        return
    print(output, file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
