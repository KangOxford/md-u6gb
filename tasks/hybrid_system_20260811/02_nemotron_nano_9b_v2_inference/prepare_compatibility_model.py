#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


UPSTREAM_BLOCK = '''try:
    #from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")'''

REFERENCE_BLOCK = '''def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    """Pure PyTorch equivalent of mamba_ssm.ops.triton.layernorm_gated.rms_norm_ref."""
    input_dtype = x.dtype
    x = x.float()
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    z = z.float() if z is not None else None
    if z is not None and not norm_before_gate:
        x = x * nn.functional.silu(z)
    if group_size is None:
        variance = x.square().mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + eps) * weight
    else:
        if x.shape[-1] % group_size:
            raise ValueError("RMSNorm feature dimension is not divisible by group_size")
        original_shape = x.shape
        grouped = x.reshape(*original_shape[:-1], original_shape[-1] // group_size, group_size)
        variance = grouped.square().mean(dim=-1, keepdim=True)
        out = (grouped * torch.rsqrt(variance + eps)).reshape(original_shape) * weight
    if bias is not None:
        out = out + bias
    if z is not None and norm_before_gate:
        out = out * nn.functional.silu(z)
    return out.to(input_dtype)'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a non-overwriting Nemotron pure-PyTorch compatibility view.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not (source / "download_manifest.json").is_file():
        raise RuntimeError("Pinned source snapshot manifest is missing")
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing destination: {destination}")
    destination.mkdir(parents=True)

    patched_name = "modeling_nemotron_h.py"
    upstream_path = source / patched_name
    upstream_text = upstream_path.read_text(encoding="utf-8")
    if upstream_text.count(UPSTREAM_BLOCK) != 1:
        raise RuntimeError("Expected exactly one upstream mamba-ssm import block")
    patched_text = upstream_text.replace(UPSTREAM_BLOCK, REFERENCE_BLOCK)

    linked = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ".cache" in path.parts or path.name == patched_name:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(path, target.parent))
        linked.append(str(relative))
    (destination / patched_name).write_text(patched_text, encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(destination),
        "patch": "replace mandatory mamba_ssm gated RMSNorm import with the upstream reference-equivalent pure PyTorch formula",
        "upstream_modeling_sha256": sha256_bytes(upstream_text.encode()),
        "derived_modeling_sha256": sha256_bytes(patched_text.encode()),
        "linked_files": linked,
    }
    (destination / "compatibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"destination": str(destination), "linked_files": len(linked), "manifest": manifest}))


if __name__ == "__main__":
    main()
