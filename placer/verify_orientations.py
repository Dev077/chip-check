"""
Verify that orientation_optimizer's choices actually improve the proxy when
applied through PlacementCost.

Reads:
  - orientations.pt sidecar produced by OrientationOptimizer.optimize()
  - benchmark name from argv

Re-runs the placer (DREAM → legalize → spread), scores the placement once
with default-N orientations and once with the sidecar's chosen orientations,
and prints the diff. Confirms that if the real scoring harness reads the
sidecar, you win the projected gain.

Run:
    uv run python placer/verify_orientations.py ibm03
"""

import sys
from pathlib import Path

import torch

# Add repo root to path so the placer imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placer.dream_placer import DreamPlacer


def main():
    if len(sys.argv) != 2:
        print("usage: verify_orientations.py <benchmark_name>")
        sys.exit(1)

    name = sys.argv[1]
    bench_dir = (
        Path(__file__).resolve().parent.parent
        / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / name
    )
    print(f"Loading {bench_dir}")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # Run the placer; OrientationOptimizer writes orientations.pt as a side-effect.
    placement = DreamPlacer().place(benchmark)

    # Baseline: score with all-N (default).
    baseline = compute_proxy_cost(placement, benchmark, plc)
    print(
        f"BASELINE (no orientation applied):\n"
        f"  proxy={baseline['proxy_cost']:.4f}  "
        f"wl={baseline['wirelength_cost']:.4f}  "
        f"den={baseline['density_cost']:.4f}  "
        f"cong={baseline['congestion_cost']:.4f}"
    )

    # Apply orientations.pt via plc.update_macro_orientation, then re-score.
    sidecar = Path("orientations.pt")
    if not sidecar.exists():
        print(f"No sidecar at {sidecar}; orientation optimizer didn't run or failed.")
        sys.exit(1)

    data = torch.load(sidecar, weights_only=False)
    codes = data["orientations"]
    names = data["names"]

    applied = 0
    for hard_i, code in enumerate(codes.tolist()):
        if code == 0:
            continue  # N is the default; nothing to do
        plc_idx = benchmark.hard_macro_indices[hard_i]
        try:
            plc.update_macro_orientation(plc_idx, names[code])
            applied += 1
        except Exception as e:
            print(f"  WARN: update_macro_orientation({plc_idx}, {names[code]}) failed: {e}")

    print(f"Applied {applied} non-N orientations through plc.update_macro_orientation()")

    after = compute_proxy_cost(placement, benchmark, plc)
    print(
        f"AFTER orientation:\n"
        f"  proxy={after['proxy_cost']:.4f}  "
        f"wl={after['wirelength_cost']:.4f}  "
        f"den={after['density_cost']:.4f}  "
        f"cong={after['congestion_cost']:.4f}"
    )

    delta = baseline['proxy_cost'] - after['proxy_cost']
    print(f"\nproxy improvement from orientations: {delta:+.4f}  "
          f"({100*delta/baseline['proxy_cost']:+.2f}%)")


if __name__ == "__main__":
    main()