"""
Multi-seed DREAM + parallel-tempering SA + orientation optimizer.

This is the top-level placer designed for a 60-min-per-benchmark budget:

    MultiSeedDream  →  pick top-2 winners  →  ParallelTemperingSA  →  OrientationOptimizer
       ~8-12 min            (~free)           ~45 min                     ~1-2 min

The SA gets the top-2 DREAM winners as starting points for two of its
chains, so parallel-tempering swaps mix two basins as well as
temperatures. The orientation optimizer is the existing greedy pass run
on the SA's final placement — orientations are independent of position,
so doing them last loses nothing.

Wall-clock budget is honored throughout: if multi-seed DREAM runs over,
SA gets less; if SA is slow, the orientation pass still gets to run.

Usage via evaluate.py:
    python -m macro_place.evaluate <bench> --placer multi_seed_dream_sa
"""

import time
from typing import List, Optional, Sequence

import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir

# These three are the building blocks; we don't re-implement them, just compose.
from placer.dream_placer import DreamPlacer
from placer.parallel_sa import ParallelTemperingSA

try:
    from placer.orientation_optimizer import OrientationOptimizer
    _HAS_ORIENT = True
except ImportError:
    _HAS_ORIENT = False


class MultiSeedDreamSA:
    """
    Composed placer: multi-seed DREAM → PT SA → orientation optimizer.

    Configurable so you can tune per-benchmark (small designs don't need
    the full budget; big ones need it all).

    Parameters
    ----------
    time_budget_s : total wall-clock budget for the whole pipeline.
        Default 3300s (55 min) leaves margin under a 60-min harness cap.
    dream_seeds : seeds to try in the multi-seed DREAM stage.
    dream_iterations : per-seed DREAM iterations. The default of 60 is
        deliberately lower than DreamPlacer's default of 100 — we want to
        spend more budget on SA polish than on extra DREAM iters that
        plateau anyway.
    sa_num_chains : R for parallel tempering. 4 is the sweet spot for a
        Colab GPU box; 8 is fine but doubles memory.
    sa_top_k_starts : how many DREAM winners to seed SA chains from.
        2 means chain 0 gets the best winner, chain 1 gets the #2.
    time_split : (dream_frac, sa_frac, orient_frac). Must sum to 1.0.
        The orient pass is fast; default favors DREAM and SA.
    """

    def __init__(
        self,
        time_budget_s: float = 3300.0,
        dream_seeds: Sequence[int] = (42, 1042, 2042, 3042),
        dream_iterations: int = 100,
        sa_num_chains: int = 4,
        sa_top_k_starts: int = 2,
        time_split: tuple = (0.20, 0.78, 0.02),
        sa_kwargs: Optional[dict] = None,
        run_orientation: bool = True,
        verbose: bool = True,
    ):
        if abs(sum(time_split) - 1.0) > 1e-3:
            raise ValueError(f"time_split must sum to 1.0, got {sum(time_split)}")
        self.time_budget_s = float(time_budget_s)
        self.dream_seeds = list(dream_seeds)
        self.dream_iterations = int(dream_iterations)
        self.sa_num_chains = int(sa_num_chains)
        self.sa_top_k_starts = int(sa_top_k_starts)
        self.time_split = tuple(time_split)
        self.sa_kwargs = dict(sa_kwargs) if sa_kwargs else {}
        self.run_orientation = run_orientation and _HAS_ORIENT
        self.verbose = verbose

    # ------------------------------------------------------------ placer API

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = self._reload_plc_for(benchmark)
        return self.place_with_plc(benchmark, plc)

    def place_with_plc(self, benchmark: Benchmark, plc) -> torch.Tensor:
        from macro_place.objective import compute_proxy_cost

        t0 = time.perf_counter()
        deadline = t0 + self.time_budget_s
        dream_deadline = t0 + self.time_split[0] * self.time_budget_s
        sa_deadline    = t0 + (self.time_split[0] + self.time_split[1]) * self.time_budget_s
        # orientation gets whatever's left up to deadline

        if self.verbose:
            print(f"[MSDS] budget {self.time_budget_s:.0f}s "
                  f"(dream≤{self.time_split[0]*self.time_budget_s:.0f}s, "
                  f"sa≤{self.time_split[1]*self.time_budget_s:.0f}s, "
                  f"orient≤{self.time_split[2]*self.time_budget_s:.0f}s)")

        # ── Stage 1: multi-seed DREAM ─────────────────────────────────
        winners = self._run_dream_seeds(benchmark, plc, dream_deadline)
        if not winners:
            # Catastrophic: every DREAM seed failed. Fall back to the
            # benchmark's initial positions.
            if self.verbose:
                print("[MSDS] WARN: no DREAM seeds completed, returning initial placement")
            return benchmark.macro_positions.clone()

        winners.sort(key=lambda w: (w["overlaps"] > 0, w["proxy"]))
        top1 = winners[0]
        top_starts = [w["placement"] for w in winners[: self.sa_top_k_starts]]

        if self.verbose:
            print(f"\n[MSDS] DREAM stage done after {time.perf_counter()-t0:.1f}s")
            print(f"[MSDS] top seeds:")
            for r, w in enumerate(winners[: max(4, self.sa_top_k_starts)]):
                marker = "→ SA" if r < self.sa_top_k_starts else "    "
                print(
                    f"  {marker} #{r+1}: seed {w['seed']:>5} "
                    f"proxy={w['proxy']:.4f} overlaps={w['overlaps']}"
                )

        # ── Stage 2: parallel-tempering SA ────────────────────────────
        sa_budget = max(0.0, sa_deadline - time.perf_counter())
        if sa_budget < 10:
            if self.verbose:
                print(f"[MSDS] only {sa_budget:.1f}s left for SA, skipping")
            refined = top1["placement"]
        else:
            if self.verbose:
                print(f"\n[MSDS] starting SA with {sa_budget:.0f}s budget")
            sa_defaults = dict(
                num_chains=self.sa_num_chains,
                time_budget_s=sa_budget,
                seed=self.dream_seeds[0],
                verbose=self.verbose,
            )
            sa_defaults.update(self.sa_kwargs)
            sa = ParallelTemperingSA(**sa_defaults)
            extras = top_starts[1:] if len(top_starts) > 1 else None
            refined = sa.refine(top1["placement"], benchmark, plc=plc, extra_starts=extras)

        # ── Stage 3: orientation optimization ─────────────────────────
        final_placement = refined
        if self.run_orientation:
            orient_budget = max(0.0, deadline - time.perf_counter())
            if orient_budget < 5:
                if self.verbose:
                    print(f"[MSDS] only {orient_budget:.1f}s left for orient, skipping")
            else:
                if self.verbose:
                    print(f"\n[MSDS] running orientation optimizer "
                          f"({orient_budget:.0f}s budget)")
                try:
                    orienter = OrientationOptimizer()
                    # OrientationOptimizer typically mutates
                    # benchmark.macro_pin_offsets in-place and returns the
                    # same placement. If its signature differs we just
                    # fall back to refined.
                    orienter_result = orienter.place(benchmark)
                    # If the orienter returned a placement, use it; otherwise
                    # stick with `refined`.
                    if isinstance(orienter_result, torch.Tensor) and \
                            orienter_result.shape == refined.shape:
                        final_placement = orienter_result
                except Exception as e:
                    if self.verbose:
                        print(f"[MSDS] WARN: orientation step failed ({e}), "
                              "returning SA result without it")

        # ── Final summary ─────────────────────────────────────────────
        if self.verbose and plc is not None:
            try:
                final_costs = compute_proxy_cost(final_placement, benchmark, plc)
                print(f"\n[MSDS] FINAL  proxy={final_costs['proxy_cost']:.4f}  "
                      f"wl={final_costs['wirelength_cost']:.3f}  "
                      f"den={final_costs['density_cost']:.3f}  "
                      f"cong={final_costs['congestion_cost']:.3f}  "
                      f"overlaps={final_costs['overlap_count']}")
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            print(f"[MSDS] total wall time: {elapsed:.1f}s "
                  f"(budget {self.time_budget_s:.0f}s)")

        return final_placement

    # ---------------------------------------------------------- dream seeds

    def _run_dream_seeds(
        self, benchmark: Benchmark, plc, deadline: float
    ) -> List[dict]:
        """
        Run DREAM with each configured seed (subject to deadline).

        Returns a list of dicts: {seed, placement, proxy, overlaps, ...}
        sorted by (overlaps>0 first, then proxy ascending) implicitly by
        the caller.
        """
        from macro_place.objective import compute_proxy_cost

        results: List[dict] = []
        for seed in self.dream_seeds:
            if time.perf_counter() >= deadline:
                if self.verbose:
                    remaining = len(self.dream_seeds) - len(results)
                    print(f"[MSDS] DREAM deadline hit; skipping {remaining} seeds")
                break
            try:
                if self.verbose:
                    print(f"\n[MSDS] DREAM seed {seed} "
                          f"(deadline in {deadline - time.perf_counter():.0f}s)")
                placer = DreamPlacer(seed=seed, iterations=self.dream_iterations)
                placement = placer.place(benchmark)
                costs = compute_proxy_cost(placement, benchmark, plc)
                results.append({
                    "seed": seed,
                    "placement": placement,
                    "proxy": costs["proxy_cost"],
                    "overlaps": costs["overlap_count"],
                    "wl": costs["wirelength_cost"],
                    "den": costs["density_cost"],
                    "cong": costs["congestion_cost"],
                })
                if self.verbose:
                    print(f"[MSDS]   → proxy={costs['proxy_cost']:.4f} "
                          f"overlaps={costs['overlap_count']}")
            except Exception as e:
                if self.verbose:
                    print(f"[MSDS]   ! seed {seed} failed: {e}")
        return results

    # ---------------------------------------------------------- plc reload

    @staticmethod
    def _reload_plc_for(benchmark: Benchmark):
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        name = benchmark.name
        candidates = [
            repo_root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / name,
        ]
        ng45_root = repo_root / "external" / "MacroPlacement" / "Flows" / "NanGate45"
        if ng45_root.exists():
            candidates.append(ng45_root / name / "netlist" / "output_CT_Grouping")
        for cand in candidates:
            if (cand / "netlist.pb.txt").exists():
                _, plc = load_benchmark_from_dir(str(cand))
                return plc
        raise FileNotFoundError(
            f"MultiSeedDreamSA: couldn't locate netlist for '{name}'. "
            f"Tried: {[str(c) for c in candidates]}"
        )


if __name__ == "__main__":
    # Run on ibm01 with a short budget, just to verify end-to-end.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    bench_dir = (
        Path(__file__).resolve().parents[1]
        / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    )
    if not bench_dir.exists():
        print(f"Skip smoke: {bench_dir} not present")
        sys.exit(0)

    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placer = MultiSeedDreamSA(
        time_budget_s=180.0,        # 3-min for smoke test
        dream_seeds=(42, 1042),     # only 2 seeds
        dream_iterations=30,        # quick
        sa_num_chains=3,
        verbose=True,
    )
    final = placer.place_with_plc(benchmark, plc)
    print(f"\nDone, shape={final.shape}")
