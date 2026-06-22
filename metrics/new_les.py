#!/usr/bin/env python3
"""
Compute binned LES from per_room_perf.json.

Input JSON format (as in per_room_perf.json):
{
  "1": {"methodA": {"SR":..., "TS":..., "PoC":...}, ...},
  "2": {...},
  ...
}

Bins requested:
- 1-3 rooms:  [1,2,3]
- 4-6 rooms:  [4,5,6]
- 7-10 rooms: [7,8,10]
"""

from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from typing import Dict, Any, List, Tuple

EPS = 1e-8

BINS = {
    "1-3": [1, 2, 3],
    "4-6": [4, 5, 6],
    "7-10": [7, 8, 10],
}

def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    return float(x)

def compute_global_minmax(data: Dict[int, Dict[str, Dict[str, float]]]) -> Tuple[float, float, float, float]:
    ts_vals, poc_vals = [], []
    for room, methods in data.items():
        for m, metrics in methods.items():
            ts = safe_float(metrics.get("TS"))
            poc = safe_float(metrics.get("PoC"))
            if math.isfinite(ts):
                ts_vals.append(ts)
            if math.isfinite(poc):
                poc_vals.append(poc)

    if not ts_vals or not poc_vals:
        raise ValueError("Could not find TS/PoC values in the JSON.")

    return min(ts_vals), max(ts_vals), min(poc_vals), max(poc_vals)

def les(
    sr: float,
    ts: float,
    poc: float,
    ts_min: float,
    ts_max: float,
    poc_min: float,
    poc_max: float,
    w_sr: float,
    w_ts: float,
    w_poc: float,
) -> float:
    # Utilities (higher is better), globally min-max normalized
    denom_ts = (ts_max - ts_min) if ts_max > ts_min else 1.0
    denom_poc = (poc_max - poc_min) if poc_max > poc_min else 1.0

    u_ts = clip01(1.0 - (ts - ts_min) / denom_ts)
    u_poc = clip01(1.0 - (poc - poc_min) / denom_poc)

    # Weighted geometric mean; SR in [0,1]
    sr = clip01(sr)
    return 100.0 * (sr ** w_sr) * ((u_ts + EPS) ** w_ts) * ((u_poc + EPS) ** w_poc)

def aggregate_bin(
    data: Dict[int, Dict[str, Dict[str, float]]],
    rooms: List[int],
) -> Dict[str, Dict[str, float]]:
    """
    For each method, average SR/TS/PoC across the rooms present in this bin.
    (Unweighted mean across rooms; change here if you want episode-weighted.)
    """
    acc = defaultdict(lambda: {"SR": 0.0, "TS": 0.0, "PoC": 0.0, "_n": 0})
    for r in rooms:
        if r not in data:
            continue
        for method, metrics in data[r].items():
            acc[method]["SR"] += safe_float(metrics.get("SR"))
            acc[method]["TS"] += safe_float(metrics.get("TS"))
            acc[method]["PoC"] += safe_float(metrics.get("PoC"))
            acc[method]["_n"] += 1

    out = {}
    for method, v in acc.items():
        n = v["_n"]
        if n == 0:
            continue
        out[method] = {
            "SR": v["SR"] / n,
            "TS": v["TS"] / n,
            "PoC": v["PoC"] / n,
        }
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", type=str, required=True, help="Path to per_room_perf.json")
    ap.add_argument("--w_sr", type=float, default=0.5, help="LES weight for SR (default: 0.5)")
    ap.add_argument("--w_ts", type=float, default=0.25, help="LES weight for TS utility (default: 0.25)")
    ap.add_argument("--w_poc", type=float, default=0.25, help="LES weight for PoC utility (default: 0.25)")
    ap.add_argument("--topk", type=int, default=50, help="How many methods to print per bin")
    args = ap.parse_args()

    # Load
    with open(args.json_path, "r") as f:
        raw = json.load(f)

    # Convert room keys to int
    data: Dict[int, Dict[str, Dict[str, float]]] = {int(k): v for k, v in raw.items()}

    # Check weights sum to 1 (renormalize if slightly off)
    w_sum = args.w_sr + args.w_ts + args.w_poc
    if w_sum <= 0:
        raise ValueError("Weights must be positive.")
    w_sr, w_ts, w_poc = args.w_sr / w_sum, args.w_ts / w_sum, args.w_poc / w_sum

    # Global min/max for normalization
    ts_min, ts_max, poc_min, poc_max = compute_global_minmax(data)
    poc_min = 1.00

    print("Global normalization ranges:")
    print(f"  TS:  min={ts_min:.4f}, max={ts_max:.4f}")
    print(f"  PoC: min={poc_min:.4f}, max={poc_max:.4f}")
    print(f"Weights (renormalized): w_sr={w_sr:.3f}, w_ts={w_ts:.3f}, w_poc={w_poc:.3f}")
    print()

    # Compute per bin
    for bin_name, rooms in BINS.items():
        agg = aggregate_bin(data, rooms)

        rows = []
        for method, m in agg.items():
            score = les(
                sr=m["SR"], ts=m["TS"], poc=m["PoC"],
                ts_min=ts_min, ts_max=ts_max, poc_min=poc_min, poc_max=poc_max,
                w_sr=w_sr, w_ts=w_ts, w_poc=w_poc,
            )
            rows.append((score, method, m["SR"], m["TS"], m["PoC"]))

        rows.sort(reverse=True, key=lambda x: x[0])

        print("FOR OVERLEAF!!!")
        print(f"=== Bin {bin_name} (rooms {rooms}) ===")
        print(f"{'rank':>4}  {'method':<12}  {'LES':>8}  {'SR':>6}  {'TS':>10}  {'PoC':>8}")
        for i, (score, method, sr, ts, poc) in enumerate(rows[: args.topk], start=1):
            print(f"{i:>4}  {method:<12}  {100*sr:.2f} & {poc:.2f} & {ts:.2f} & {score:.2f}")
        print()

        print(f"=== Bin {bin_name} (rooms {rooms}) ===")
        print(f"{'rank':>4}  {'method':<12}  {'LES':>8}  {'SR':>6}  {'TS':>10}  {'PoC':>8}")
        for i, (score, method, sr, ts, poc) in enumerate(rows[: args.topk], start=1):
            print(f"{i:>4}  {method:<12}  {score:8.2f}  {sr:6.3f}  {ts:10.2f}  {poc:8.3f}")
        print()

if __name__ == "__main__":
    main()