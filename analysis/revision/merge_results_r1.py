#!/usr/bin/env python3
"""merge_results_r1.py - merge the per-analysis result fragments into results_r1.json.

Reads every results_a*.json in the fragments directory (results/revision/fragments/ by default;
results_r1.json itself is never an input), checks each fragment against the namespace table
below, raises on a duplicate key or on a key outside its fragment's namespace, and writes the
merged file (sorted keys, indent 2, trailing newline, allow_nan=False, read-back equality)
through r1_common.write_fragment.

A fragment file must be named results_aN.json with N in 1..9. Anything else that matches
results_a*.json (for example a stray copy such as 'results_a1 2.json') raises.
The first path segment of every key must be one of the namespaces allowed for that fragment:
a1 may also carry a6 and a7, a3 may also carry a8, a4 may also carry a5, because closely related
analyses may share a fragment.

Run the analysis scripts first (a1 to a5, a7 to a9; a9 reads the a8 fragment, so run a8 before a9).

    PYTHONDONTWRITEBYTECODE=1 python3 merge_results_r1.py
    PYTHONDONTWRITEBYTECODE=1 python3 merge_results_r1.py --dir <fragment dir> --out <file>

The output file must lie under the results directory (ACDF_RESULTS_DIR).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_common as R1  # noqa: E402

# fragment id -> first-segment namespaces its keys may use
NAMESPACES: Dict[str, Sequence[str]] = {
    "a1": ("a1", "a6", "a7"),
    "a2": ("a2",),
    "a3": ("a3", "a8"),
    "a4": ("a4", "a5"),
    "a5": ("a5",),
    "a6": ("a6",),
    "a7": ("a7",),
    "a8": ("a8",),
    "a9": ("a9",),
}
ALL_PREFIXES = tuple(f"a{i}." for i in range(1, 10))
FRAGMENT_RE = re.compile(r"^results_(a[1-9])\.json$")


def merge(fragment_dir: Path) -> Dict[str, Any]:
    """Return the merged dict. Raises R1.R1Error on any contract violation."""
    files = sorted(p for p in fragment_dir.glob("results_a*.json"))
    if not files:
        raise R1.R1Error(f"no results_a*.json fragments in {fragment_dir}")
    merged: Dict[str, Any] = {}
    owner: Dict[str, str] = {}
    for path in files:
        m = FRAGMENT_RE.match(path.name)
        if not m:
            raise R1.R1Error(f"unexpected fragment file name {path.name!r} (expected results_aN.json, "
                             f"N in 1..9); a stray copy of a fragment?")
        allowed = NAMESPACES[m.group(1)]
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise R1.R1Error(f"{path.name} is not a JSON object")
        for key, value in data.items():
            if not R1.KEY_RE.match(key):
                raise R1.R1Error(f"{path.name}: key {key!r} is not a flat lowercase dotted path")
            top = key.split(".", 1)[0]
            if "." not in key or top not in allowed:
                raise R1.R1Error(f"{path.name}: key {key!r} is outside its namespaces {list(allowed)}")
            if key in merged:
                raise R1.R1Error(f"duplicate key {key!r} in {path.name} and {owner[key]}")
            merged[key] = value
            owner[key] = path.name
    return merged


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Merge results_a*.json into results_r1.json.")
    ap.add_argument("--dir", default=str(R1.FRAGMENTS_OUT), help="directory holding results_a*.json")
    ap.add_argument("--out", default=str(R1.MERGED_JSON), help="merged output file")
    args = ap.parse_args(argv)
    fragment_dir = Path(args.dir).resolve()
    out = Path(args.out).resolve()

    merged = merge(fragment_dir)
    n = R1.write_fragment(out, merged, ALL_PREFIXES)
    per: Dict[str, int] = {}
    for key in merged:
        top = key.split(".", 1)[0]
        per[top] = per.get(top, 0) + 1
    print(f"[merge] {n} keys from {len(list(fragment_dir.glob('results_a*.json')))} fragment(s): "
          + ", ".join(f"{k}={v}" for k, v in sorted(per.items())))
    print(f"[merge] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
