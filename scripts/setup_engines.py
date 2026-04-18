"""Idempotent engine bootstrap.

Run at Docker build time (to warm the Maia-2 weight cache) and on local
dev when you want to verify engines are wired up. Safe to re-run; does
nothing when everything is already in place.

Usage:
    python scripts/setup_engines.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys


def check_stockfish() -> bool:
    path = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
    if not path:
        print("  [stockfish] NOT FOUND — install via apt (Linux) or download a binary and set STOCKFISH_PATH")
        return False
    print(f"  [stockfish] OK — {path}")
    return True


def check_maia2() -> bool:
    if importlib.util.find_spec("maia2") is None:
        print("  [maia2]    NOT INSTALLED — `pip install maia2`")
        return False
    cache = os.environ.get("MAIA2_CACHE_DIR")
    if cache:
        os.makedirs(cache, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache)
        print(f"  [maia2]    cache at {cache}")
    try:
        from maia2 import inference, model  # type: ignore

        print("  [maia2]    loading rapid model (this may download weights)...")
        m = model.from_pretrained(type="rapid", device="cpu")
        _ = inference.prepare()
        print(f"  [maia2]    OK — {type(m).__name__} loaded")
        return True
    except Exception as exc:
        print(f"  [maia2]    FAILED at load — {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    print("Metropolis Chess Club — engine setup")
    ok_sf = check_stockfish()
    ok_maia = check_maia2()
    print()
    if ok_sf and ok_maia:
        print("All engines ready.")
        return 0
    print("Some engines are unavailable — the app will still run, but gameplay will")
    print("fall back to MockEngine or the remaining real engine.")
    return 0  # Non-zero would fail Docker builds; we want builds to succeed even if
              # weights couldn't be downloaded (they'll download on first use instead).


if __name__ == "__main__":
    sys.exit(main())
