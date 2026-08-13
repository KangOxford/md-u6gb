#!/bin/bash
# Shadow NVIDIA Transformer Engine inside this venv with an empty package.
#
# TE cannot run in this environment: its import loads cudnn, nvrtc and curand, and
# **libcurand exists nowhere here** -- not in the torch `nvidia/*` wheels, not on the
# system. LD_LIBRARY_PATH cannot fix a library that is absent.
#
# That would be nobody's problem except that peft's probe is unguarded:
#
#     def is_te_pytorch_available():
#         import transformer_engine                      # bare, no try/except
#         return hasattr(transformer_engine, "pytorch")
#
# and peft sits on the import path to `MiniMaxH3Transformer3DModel` via
# `PeftAdapterMixin`, so TE's RuntimeError comes out of `import diffusers`.
#
# With this stub the probe imports fine and finds no `pytorch` attribute, so peft
# reports TE-pytorch unavailable, which is the truth. Idempotent; safe to re-run.

set -euo pipefail
TASKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SP="$(echo "$TASKDIR"/venv/lib/python3.*/site-packages)"

if [ ! -d "$SP" ]; then
    echo "FATAL: venv site-packages not found at $SP" >&2
    exit 1
fi

mkdir -p "$SP/transformer_engine"
cat > "$SP/transformer_engine/__init__.py" <<'EOF'
"""Stub shadowing NVIDIA Transformer Engine; see code/setup_te_stub.sh for why.

Short version: real TE needs libcurand, which does not exist in this environment, so
it can never import. peft probes it without a try/except and that RuntimeError comes
out of `import diffusers`. This module imports cleanly and has no `pytorch`
attribute, so peft's `is_te_pytorch_available()` returns False -- the truth.
"""

__version__ = "0.0.0+stub-no-curand"
EOF

echo "[te-stub] wrote $SP/transformer_engine/__init__.py"
"$TASKDIR/venv/bin/python" -c "
import transformer_engine as te
from peft.import_utils import is_te_pytorch_available
assert not hasattr(te, 'pytorch'), 'stub is being shadowed by the real TE'
assert is_te_pytorch_available() is False
print('[te-stub] verified: peft reports TE-pytorch unavailable without raising')
"
