"""
LKPlacer — Five-phase macro placer with electrostatic GP front-end.

Phase α  Focused Electrostatic Global Placement  (gp.run_global_placement)
Phase 0  Legalize hard macros
Phase 1  Build FastEvaluator (bit-exact mirror of PlacementCost)
Phase 2  Lin-Kernighan k-opt + grid sweep
Phase 3  LAHC polish (true cost via fast evaluator, mixed hard/soft moves
         including partner-centroid biased proposals)
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark

from . import gp as gp_mod
from .cong_attack import direct_congestion_attack
from .evaluator import FastEvaluator
from .lahc import lahc_polish
from .legalize import _legalize
from .lk import _macro_priority, lk_swap_pass
from .regional import regional_polish
from .subgradient import true_cost_subgradient


# ────────────────────────────────────────────────────────────────────────────
# LKPlacer orchestrator
# ────────────────────────────────────────────────────────────────────────────


def _load_plc(name: str):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {
        "ariane133": "ariane133", "ariane136": "ariane136",
        "nvdla": "nvdla", "mempool_tile": "mempool_tile",
    }
    d = ng45.get(name.replace("_ng45", ""))
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


class LKPlacer:
    """Five-phase macro placer.  See module docstring."""

    def __init__(
        self,
        seed: int = 42,
        time_budget_s: float = 3000.0,
        # Phase α₁ (electrostatic GP)
        run_gp: bool = True,
        gp_pop_size: int = 4,
        gp_steps: int = 500,
        gp_budget_s: float = 90.0,
        # Phase α₂ (true-cost subgradient)
        run_alpha2: bool = True,
        alpha2_budget_s: float = 60.0,
        # Phase 2 LK
        lk_passes: int = 3,
        lk_neighbors: int = 24,
        lk_chain_depth: int = 4,
        # Phase 2.5 direct congestion attack
        run_cong_attack: bool = False,  # disabled: greedy moves trap LAHC in tighter basin
        cong_attack_passes: int = 3,
        cong_attack_budget_s: float = 60.0,
        # Phase 4 hierarchical regional polish
        run_regional: bool = True,
        regional_grid_sizes: Tuple[int, ...] = (3, 5, 7),
        regional_budget_frac: float = 0.45,
        regional_min_macros_for_phase: int = 30,
        regional_list_len: int = 60,
        regional_move_radius_frac: float = 0.12,
        # Phase 4 GPU variant: K parallel chains, batched proxy eval
        regional_use_gpu: bool = False,
        regional_n_chains: int = 8,
        # Phase 3 LAHC
        lahc_list_len: int = 100,
        verbose: bool = True,
    ):
        self.seed = seed
        self.time_budget_s = time_budget_s
        self.run_gp = run_gp
        self.gp_pop_size = gp_pop_size
        self.gp_steps = gp_steps
        self.gp_budget_s = gp_budget_s
        self.run_alpha2 = run_alpha2
        self.alpha2_budget_s = alpha2_budget_s
        self.lk_passes = lk_passes
        self.lk_neighbors = lk_neighbors
        self.lk_chain_depth = lk_chain_depth
        self.run_cong_attack = run_cong_attack
        self.cong_attack_passes = cong_attack_passes
        self.cong_attack_budget_s = cong_attack_budget_s
        self.run_regional = run_regional
        self.regional_grid_sizes = tuple(regional_grid_sizes)
        self.regional_budget_frac = regional_budget_frac
        self.regional_min_macros_for_phase = regional_min_macros_for_phase
        self.regional_list_len = regional_list_len
        self.regional_move_radius_frac = regional_move_radius_frac
        self.regional_use_gpu = regional_use_gpu
        self.regional_n_chains = regional_n_chains
        self.lahc_list_len = lahc_list_len
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[lk_placer] {msg}", flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        from macro_place.objective import compute_proxy_cost

        t0 = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        plc = _load_plc(benchmark.name)

        # ── Phase α — Focused Electrostatic GP ──
        if self.run_gp:
            try:
                self._log(f"Phase α: focused electrostatic global placement (pop={self.gp_pop_size}, steps={self.gp_steps}, budget={self.gp_budget_s:.0f}s)")
                gp_positions = gp_mod.run_global_placement(
                    benchmark, plc,
                    pop_size=self.gp_pop_size,
                    n_steps=self.gp_steps,
                    time_budget_s=self.gp_budget_s,
                    seed=self.seed,
                    verbose=self.verbose,
                )
                benchmark.macro_positions = torch.from_numpy(gp_positions).float()
            except Exception as e:
                self._log(f"Phase α: SKIPPED due to exception: {e}")

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        mov_np = benchmark.get_movable_mask()[:n_hard].cpu().numpy().astype(bool)
        init = benchmark.macro_positions[:n_hard].cpu().numpy().astype(np.float64)

        # ── Phase 0 — legalize ──
        self._log("Phase 0: legalizing hard macros")
        hard_legal = _legalize(init, sizes_np, mov_np, cw, ch)
        pos_full = benchmark.macro_positions.numpy().astype(np.float64).copy()
        pos_full[:n_hard] = hard_legal
        benchmark.macro_positions = torch.from_numpy(pos_full).float()

        # ── Phase 1 — FastEvaluator ──
        self._log("Phase 1: building FastEvaluator")
        ev = FastEvaluator(benchmark, plc)
        c0 = ev.proxy_cost()
        self._log(f"  fast baseline: proxy={c0['proxy_cost']:.4f} wl={c0['wirelength_cost']:.4f} den={c0['density_cost']:.4f} cong={c0['congestion_cost']:.4f}")
        true_c = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
        self._log(f"  oracle: {true_c['proxy_cost']:.4f}  overlaps={true_c['overlap_count']}")
        best_true = float(true_c["proxy_cost"]) if true_c["overlap_count"] == 0 else float("inf")
        best_pos = ev.positions.copy() if true_c["overlap_count"] == 0 else None

        # ── Phase α₂ — Stochastic true-cost subgradient ──
        if self.run_alpha2:
            self._log(f"Phase α₂: stochastic true-cost subgradient (budget={self.alpha2_budget_s:.0f}s)")
            out = true_cost_subgradient(
                ev,
                time_budget_s=self.alpha2_budget_s,
                seed=self.seed,
                verbose=self.verbose,
            )
            self._log(f"  α₂: best fast={out['proxy_cost']:.4f}  iters={out['iters']}  accepted={out['accepted']}")
            tc = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
            self._log(f"  oracle: {tc['proxy_cost']:.4f}  overlaps={tc['overlap_count']}")
            if tc["overlap_count"] == 0 and tc["proxy_cost"] < best_true:
                best_true = float(tc["proxy_cost"])
                best_pos = ev.positions.copy()
            elif best_pos is not None:
                ev.restore(best_pos)

        # ── Phase 2 — LK ──
        # Reserve at least 30% of total budget (or 60s, whichever is more) for LAHC,
        # not a cap on LK itself.  LK converges naturally after 2-3 passes anyway.
        min_lahc_s = max(60.0, self.time_budget_s * 0.30)
        for p in range(self.lk_passes):
            if time.time() - t0 > self.time_budget_s - min_lahc_s:
                self._log(f"Phase 2 pass {p}: reserving {min_lahc_s:.0f}s for LAHC, skipping further passes")
                break
            self._log(f"Phase 2 pass {p}: macro priority queue")
            order = _macro_priority(ev)
            cur_cost, n_acc = lk_swap_pass(
                ev, order,
                chain_depth=self.lk_chain_depth,
                n_neighbors_per_macro=self.lk_neighbors,
                log_every=max(1, len(order) // 6),
            )
            self._log(f"  pass {p}: fast proxy={cur_cost:.4f} accepted={n_acc}")
            tc = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
            self._log(f"  true oracle: {tc['proxy_cost']:.4f}  overlaps={tc['overlap_count']}")
            if tc["overlap_count"] == 0 and tc["proxy_cost"] < best_true:
                best_true = float(tc["proxy_cost"])
                best_pos = ev.positions.copy()

        # ── Phase 2.5 — Direct congestion attack ──
        if self.run_cong_attack and best_pos is not None:
            ev.restore(best_pos)
            self._log(f"Phase 2.5: direct congestion attack (budget={self.cong_attack_budget_s:.0f}s)")
            out = direct_congestion_attack(
                ev,
                n_passes=self.cong_attack_passes,
                time_budget_s=self.cong_attack_budget_s,
                verbose=self.verbose,
            )
            self._log(f"  cong-attack: improvement={out['improvement']:+.4f} accepted={out['accepted']}")
            tc = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
            self._log(f"  oracle: {tc['proxy_cost']:.4f}  overlaps={tc['overlap_count']}")
            if tc["overlap_count"] == 0 and tc["proxy_cost"] < best_true:
                best_true = float(tc["proxy_cost"])
                best_pos = ev.positions.copy()
            elif best_pos is not None:
                ev.restore(best_pos)

        # ── Phase 4 — Hierarchical Regional Polish ──
        # Runs before LAHC: gives LAHC a better starting point by doing
        # region-by-region focused optimization that single-macro LAHC moves
        # cannot do.  Disabled automatically for small designs where LAHC
        # already explores the full space efficiently.
        if (
            self.run_regional
            and best_pos is not None
            and ev.n_hard >= self.regional_min_macros_for_phase
        ):
            ev.restore(best_pos)
            total_remaining = max(60.0, self.time_budget_s - (time.time() - t0))
            regional_budget = total_remaining * self.regional_budget_frac
            if self.regional_use_gpu:
                self._log(
                    f"Phase 4 (GPU): K={self.regional_n_chains} parallel chains  "
                    f"grids={self.regional_grid_sizes}  budget={regional_budget:.0f}s "
                    f"(of {total_remaining:.0f}s remaining)"
                )
                # Lazy-import: regional_gpu.py has a top-level reference to
                # gp.macro_routing_demand which doesn't exist, so loading it
                # eagerly would break LKPlacer for the common (CPU) path.
                from . import regional_gpu as regional_gpu_mod
                # Re-rank chains by the bit-exact CPU FastEvaluator before
                # picking the winner — the GPU surrogate uses RUDY pin
                # congestion (not Steiner-tree like the proxy) so ranking
                # by GPU cost alone can be off.
                _ranking_ev = ev
                def _true_proxy_of(pos_np):
                    _ranking_ev.restore(pos_np)
                    return _ranking_ev.proxy_cost()["proxy_cost"]
                best_positions_np, out = regional_gpu_mod.regional_polish_gpu(
                    benchmark, ev.positions.copy(), plc,
                    n_chains=self.regional_n_chains,
                    region_grids=self.regional_grid_sizes,
                    list_len=self.regional_list_len,
                    move_radius_frac=self.regional_move_radius_frac,
                    time_budget_s=regional_budget,
                    seed=self.seed,
                    verbose=self.verbose,
                    rerank_with_true_proxy_cb=_true_proxy_of,
                )
                self._log(
                    f"  regional-gpu: best chain={out['best_chain']} "
                    f"true_proxy={out['proxy_cost']:.4f}  iters={out['iters']}  "
                    f"accepted={out['accepted']}\n"
                    f"               gpu_costs ={[f'{c:.4f}' for c in out['all_chain_costs_gpu']]}\n"
                    f"               true_costs={[f'{c:.4f}' for c in out['all_chain_costs_true']]}"
                )
                ev.restore(best_positions_np)
            else:
                self._log(
                    f"Phase 4: regional polish  "
                    f"grids={self.regional_grid_sizes}  budget={regional_budget:.0f}s "
                    f"(of {total_remaining:.0f}s remaining)"
                )
                out = regional_polish(
                    ev,
                    region_grids=self.regional_grid_sizes,
                    time_budget_s=regional_budget,
                    list_len=self.regional_list_len,
                    move_radius_frac=self.regional_move_radius_frac,
                    seed=self.seed,
                    verbose=self.verbose,
                )
                self._log(f"  regional: best fast={out['proxy_cost']:.4f}  iters={out['iters']}  accepted={out['accepted']}")
            tc = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
            self._log(f"  oracle: {tc['proxy_cost']:.4f}  overlaps={tc['overlap_count']}")
            if tc["overlap_count"] == 0 and tc["proxy_cost"] < best_true:
                best_true = float(tc["proxy_cost"])
                best_pos = ev.positions.copy()
            elif best_pos is not None:
                ev.restore(best_pos)

        # ── Phase 3 — LAHC ──
        if best_pos is not None:
            ev.restore(best_pos)
        remaining = max(60.0, self.time_budget_s - (time.time() - t0))
        self._log(f"Phase 3: LAHC polish, budget={remaining:.0f}s")
        out = lahc_polish(
            ev,
            list_len=self.lahc_list_len,
            time_budget_s=remaining,
            seed=self.seed,
            verbose=self.verbose,
        )
        self._log(f"  LAHC: best={out['proxy_cost']:.4f}  iters={out['iters']}")
        tc = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
        if tc["overlap_count"] == 0 and tc["proxy_cost"] < best_true:
            best_true = float(tc["proxy_cost"])
            best_pos = ev.positions.copy()

        self._log(f"DONE  best_true={best_true:.4f}  time={time.time()-t0:.1f}s")
        return torch.from_numpy(best_pos if best_pos is not None else ev.positions).float()
