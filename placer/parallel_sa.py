"""
Parallel-tempering polish SA for macro placement.

Designed as a post-process for an already-legal placement (e.g. from
multi-seed DREAM → legalize → spread). The SA's job is to refine, not to
fix illegal layouts, so we treat zero-overlap as a hard constraint: any
move that would create overlap is rejected outright instead of penalized.

Architecture
------------
- R chains run in parallel at geometrically-spaced temperatures.
  Hot chains explore broadly, cold chains converge to local optima.
  Every PT_SWAP_INTERVAL iterations we propose swaps between
  adjacent-temperature chains. The cold chain ends up holding the best
  states found anywhere in the ensemble.

- Each chain owns its own CostEstimator (stateful). Position, density,
  congestion, and HPWL state live inside the estimator; chains never
  share mutable state, PT swap is the only cross-chain interaction.

- Move portfolio per step (weighted random pick):
    * translate     — Gaussian step around current position
    * pair-swap     — swap centers of two random hard movable macros
    * block-move    — translate a cluster of K nearby macros together
  All moves enforce zero-overlap as a hard reject. Out-of-canvas is also
  a hard reject.

  Orientation flips are intentionally NOT done in SA — they're handled
  by the existing OrientationOptimizer as a separate greedy pass run
  after SA finishes.

- Wall-clock budget: the run loops until either max_iterations OR
  time_budget_s elapses (whichever first). Best-so-far is always
  returned. Two of the R chains can be seeded from different starting
  placements ("top-2 from DREAM") to give PT swaps real basin diversity.

Cost model
----------
Internal objective is the bare proxy cost from CostEstimator (no overlap
penalty term — overlaps are rejected, not penalized). The estimator's
proxy cost is a tensor-grid approximation of the real PlacementCost
evaluator, so the final winner is re-scored through compute_proxy_cost
after the SA finishes. If SA's "best" scores worse than its input under
the real evaluator (estimator drift), we return the input.

Usage
-----
    from placer.parallel_sa import ParallelTemperingSA
    sa = ParallelTemperingSA(num_chains=4, time_budget_s=2400)
    refined = sa.refine(placement, benchmark, plc=plc)
"""

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

try:
    from utils.cost_utils import CostEstimator
except ImportError:
    from chip_check.utils.cost_utils import CostEstimator


# ─── ChainState: everything one PT chain needs ──────────────────────────────


@dataclass
class ChainState:
    """All mutable state for one PT chain."""
    estimator: CostEstimator
    temperature: float
    step_frac: float                      # translation step as fraction of canvas
    current_cost: float
    best_cost: float
    best_placement: torch.Tensor

    # Diagnostics
    moves_proposed: int = 0
    moves_accepted: int = 0
    rejected_overlap: int = 0
    rejected_oob: int = 0
    rejected_metropolis: int = 0


# ─── ParallelTemperingSA ────────────────────────────────────────────────────


class ParallelTemperingSA:
    """
    Parallel-tempering SA. Acts as a refinement pass on a legal placement.

    Parameters
    ----------
    num_chains : R, number of PT chains. 4 is a reasonable default.
    t_max, t_min : temperature endpoints (geometric ladder between them).
    pt_swap_interval : how many iterations between PT swap attempts.
    max_iterations : per-chain iteration cap. The wall-clock budget
        usually hits first.
    time_budget_s : if set, SA returns no later than this many seconds.
    move_weights : (translate, swap, block) probabilities.
        Must sum to 1.0; we re-normalize otherwise.
    block_k : how many neighbors get pulled into a "block" move.
    auto_calibrate : if True, take a sample of trial moves before SA
        proper starts and use the observed cost-delta distribution to
        set T_max / T_min and step sizes. Eliminates the need to tune
        temperatures for each design.
    calibration_samples : how many trial moves to use for calibration.
    min_step_frac, max_step_frac : floor and ceiling on translation
        step size, as fractions of canvas. The cold chain wouldn't
        learn anything with step_frac=0; the hot chain shouldn't be
        teleporting either.
    step_alpha : how aggressively step size tracks temperature.
        step ∝ T^alpha with alpha=0.5 (square root) being the default —
        cold chains still take meaningful steps, hot chains don't go
        wild. alpha=1 reproduces the naive linear-in-T behaviour;
        alpha=0 turns off temperature-step coupling entirely.
    extra_starts : optional list of additional starting placements (e.g.
        the #2 winner from multi-seed DREAM). The first `len(extra_starts)`
        chains will be initialized from those; the rest start from the
        primary placement.
    """

    def __init__(
        self,
        num_chains: int = 4,
        t_max: float = 1.0,
        t_min: float = 1e-3,
        pt_swap_interval: int = 100,
        max_iterations: int = 200_000,
        time_budget_s: Optional[float] = None,
        move_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),
        block_k: int = 5,
        auto_calibrate: bool = True,
        calibration_samples: int = 200,
        min_step_frac: float = 0.002,
        max_step_frac: float = 0.10,
        step_alpha: float = 0.5,
        seed: int = 42,
        verbose: bool = True,
        log_interval: int = 500,
    ):
        if num_chains < 1:
            raise ValueError("num_chains must be >= 1")
        self.num_chains = num_chains
        # Initial (pre-calibration) temperature ladder. May be replaced
        # by _calibrate(). Geometric spacing is correct in either case.
        self._t_max_init = t_max
        self._t_min_init = t_min
        if num_chains == 1:
            self.temperatures = [t_max]
        else:
            ratio = (t_min / t_max) ** (1.0 / (num_chains - 1))
            self.temperatures = [t_max * (ratio ** i) for i in range(num_chains)]
        self.pt_swap_interval = pt_swap_interval
        self.max_iterations = max_iterations
        self.time_budget_s = time_budget_s

        s = sum(move_weights)
        self.move_weights = tuple(w / s for w in move_weights)
        self._move_cdf = [self.move_weights[0]]
        for w in self.move_weights[1:]:
            self._move_cdf.append(self._move_cdf[-1] + w)
        self.block_k = block_k

        self.auto_calibrate = auto_calibrate
        self.calibration_samples = calibration_samples
        self.min_step_frac = min_step_frac
        self.max_step_frac = max_step_frac
        self.step_alpha = step_alpha

        self.seed = seed
        self.verbose = verbose
        self.log_interval = log_interval

    # -------------------------------------------------------------- main API

    def refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc=None,
        extra_starts: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Refine `placement` and return an improved (still legal) placement.

        If `plc` is given, the final result is sanity-checked against the
        real PlacementCost evaluator and we'll return `placement` instead
        of our SA's output if it would actually be a regression.
        """
        rng = random.Random(self.seed)
        torch.manual_seed(self.seed)

        num_hard = benchmark.num_hard_macros
        if num_hard == 0:
            return placement

        # Pre-index movable hard macros once.
        movable_hard = [
            i for i in range(num_hard)
            if not bool(benchmark.macro_fixed[i].item())
        ]
        if not movable_hard:
            if self.verbose:
                print("[PT-SA] no movable hard macros, skipping")
            return placement

        # Start the wall-clock timer *now* so calibration and chain init
        # both count against the budget. Otherwise calibration with N=200
        # trial moves would silently blow past short budgets.
        global_t0 = time.perf_counter()

        # Set up per-chain state.
        chains = self._init_chains(placement, benchmark, extra_starts)

        # Auto-calibrate temperatures from a sample of trial moves on a
        # scratch chain. Replaces self.temperatures if it fires.
        if self.auto_calibrate and self.num_chains > 1:
            self._calibrate_temperatures(chains[0], benchmark, movable_hard, rng)
            for ci, ch in enumerate(chains):
                ch.temperature = self.temperatures[ci]

        # Pre-compute per-chain translation step size. Formula:
        #   step_frac = clamp(min_step_frac, max_step_frac,
        #                     base * (T / T_geometric_mean)^step_alpha)
        # The 'base' is fixed at 2% of canvas regardless of the temperature
        # range — the calibration step already sized T to the cost
        # landscape, so we don't double-couple the step here.
        T_geomean = math.exp(sum(math.log(t) for t in self.temperatures) / len(self.temperatures))
        base_frac = 0.02
        self._step_fracs = []
        for T in self.temperatures:
            raw = base_frac * (T / T_geomean) ** self.step_alpha
            clipped = max(self.min_step_frac, min(self.max_step_frac, raw))
            self._step_fracs.append(clipped)
        for ci, ch in enumerate(chains):
            ch.step_frac = self._step_fracs[ci]

        global_best_cost = min(c.best_cost for c in chains)
        global_best_chain = int(torch.tensor([c.best_cost for c in chains]).argmin().item())

        if self.verbose:
            temps_str = ", ".join(f"{t:.4g}" for t in self.temperatures)
            steps_str = ", ".join(f"{s:.4f}" for s in self._step_fracs)
            print(f"[PT-SA] {self.num_chains} chains, T in [{temps_str}]")
            print(f"[PT-SA] step fractions (per chain): [{steps_str}]")
            print(f"[PT-SA] move mix: translate={self.move_weights[0]:.2f} "
                  f"swap={self.move_weights[1]:.2f} block={self.move_weights[2]:.2f}")
            print(f"[PT-SA] initial costs: " +
                  ", ".join(f"c{i}={c.current_cost:.4f}" for i, c in enumerate(chains)))

        # ── main loop ─────────────────────────────────────────────────
        for it in range(self.max_iterations):
            # Wall-clock guard.
            if self.time_budget_s is not None and (it & 0x3F) == 0:
                # Check every 64 iters to keep overhead negligible.
                elapsed = time.perf_counter() - global_t0
                if elapsed >= self.time_budget_s:
                    if self.verbose:
                        print(f"[PT-SA] time budget {self.time_budget_s:.0f}s "
                              f"hit at iter {it}, stopping")
                    break

            # One move per chain per iteration.
            for chain in chains:
                self._step_chain(chain, benchmark, movable_hard, rng)

            # Periodic parallel-tempering swap attempts.
            if (it + 1) % self.pt_swap_interval == 0:
                self._pt_swap_step(chains, rng)

            # Track global best.
            for ci, chain in enumerate(chains):
                if chain.best_cost < global_best_cost:
                    global_best_cost = chain.best_cost
                    global_best_chain = ci

            if self.verbose and (it + 1) % self.log_interval == 0:
                elapsed = time.perf_counter() - global_t0
                acc_rate = [
                    c.moves_accepted / max(1, c.moves_proposed) for c in chains
                ]
                print(
                    f"[PT-SA] iter {it+1:6d}  t={elapsed:6.1f}s  "
                    f"best={global_best_cost:.4f} (chain {global_best_chain})  "
                    f"accept=[{', '.join(f'{a:.2f}' for a in acc_rate)}]"
                )

        # ── pick the best across all chains ───────────────────────────
        best_chain = min(chains, key=lambda c: c.best_cost)
        sa_placement = best_chain.best_placement.clone()

        # ── safety net: re-score with the real evaluator ──────────────
        if plc is not None:
            try:
                input_costs = compute_proxy_cost(placement, benchmark, plc)
                sa_costs = compute_proxy_cost(sa_placement, benchmark, plc)
                if self.verbose:
                    print(
                        f"[PT-SA] real-eval input proxy={input_costs['proxy_cost']:.4f} "
                        f"sa proxy={sa_costs['proxy_cost']:.4f} "
                        f"(estimator best={global_best_cost:.4f})"
                    )
                if sa_costs["proxy_cost"] > input_costs["proxy_cost"] or sa_costs["overlap_count"] > 0:
                    if self.verbose:
                        reason = "regression" if sa_costs["proxy_cost"] > input_costs["proxy_cost"] else "overlap"
                        print(f"[PT-SA] {reason} under real evaluator — returning input placement")
                    return placement
            except Exception as e:
                if self.verbose:
                    print(f"[PT-SA] WARN: real-eval safety check failed: {e}; trusting SA")

        return sa_placement

    def _calibrate_temperatures(
        self,
        scratch_chain: ChainState,
        benchmark: Benchmark,
        movable_hard: List[int],
        rng: random.Random,
    ):
        """
        Sample N trial translate-moves and use the cost-delta distribution
        to set T_max / T_min, replacing self.temperatures with a calibrated
        geometric ladder.

        Why this matters: the right temperature range depends on the cost
        landscape, which depends on the design. Fixed defaults like
        T_max=1.0 are usually wrong by an order of magnitude. We pick T_max
        so the hottest chain accepts ~80-95% of moves, and T_min so the
        coldest chain accepts only good (delta < 0) moves.

        Side effect: all moves used here are rolled back, so the scratch
        chain returns to its initial state.
        """
        n = self.calibration_samples
        if n < 10 or len(movable_hard) == 0:
            return  # not enough budget or nothing to move

        est = scratch_chain.estimator
        old_cost = scratch_chain.current_cost
        positive_deltas: List[float] = []

        # Snapshot the chain so we can roll back wholesale rather than
        # try to undo each move individually (which is fiddly given
        # `_apply_position_move`'s internal rollback path).
        snapshot = self._snapshot_chain(scratch_chain)

        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height
        sample_step = 0.02  # canvas fraction — match the geometric-mean step

        for _ in range(n):
            idx = rng.choice(movable_hard)
            old_pos = est.placement[idx].clone()
            step_x = rng.gauss(0, 1) * canvas_w * sample_step
            step_y = rng.gauss(0, 1) * canvas_h * sample_step
            w, h = benchmark.macro_sizes[idx]
            new_x = float(max(w / 2, min(canvas_w - w / 2, old_pos[0] + step_x)))
            new_y = float(max(h / 2, min(canvas_h - h / 2, old_pos[1] + step_y)))

            # Apply, measure, roll back. Have to do it inline (not via
            # _apply_position_move) because that function couples
            # accept/reject decisions to a temperature; here we want to
            # measure delta regardless and then always revert.
            est.update_macro_pos(idx, torch.tensor([new_x, new_y]))
            if est.overlap_count == 0:
                # Only count moves that wouldn't be immediately rejected
                # by the hard-overlap constraint, since those are the
                # ones T actually arbitrates.
                new_cost = est.get_costs()["proxy_cost"]
                delta = new_cost - old_cost
                if delta > 0:
                    positive_deltas.append(delta)
            # Roll back this single move with another update_macro_pos.
            est.update_macro_pos(idx, old_pos)

        # If the design happens to have no positive deltas in the sample
        # (already at local min, or estimator perfectly flat), keep the
        # default ladder.
        if len(positive_deltas) < 5:
            if self.verbose:
                print(f"[PT-SA] calibration: only {len(positive_deltas)} positive deltas, "
                      f"keeping default T range [{self._t_min_init:.4g}, {self._t_max_init:.4g}]")
            self._restore_chain(scratch_chain, snapshot)
            return

        positive_deltas.sort()
        # 95th percentile → T_max (hot chain accepts most uphill moves).
        # 10th percentile → T_min (cold chain rejects nearly all uphill).
        t_max = positive_deltas[int(0.95 * len(positive_deltas))]
        t_min = max(positive_deltas[int(0.10 * len(positive_deltas))], t_max * 1e-3)
        if t_min >= t_max:
            t_min = t_max * 1e-3

        # Rebuild geometric ladder.
        if self.num_chains == 1:
            self.temperatures = [t_max]
        else:
            ratio = (t_min / t_max) ** (1.0 / (self.num_chains - 1))
            self.temperatures = [t_max * (ratio ** i) for i in range(self.num_chains)]

        if self.verbose:
            print(
                f"[PT-SA] calibration: {len(positive_deltas)} positive deltas, "
                f"median={positive_deltas[len(positive_deltas)//2]:.4g}, "
                f"T_max={t_max:.4g}, T_min={t_min:.4g}"
            )

        # Make sure scratch chain is back to exactly its starting state.
        self._restore_chain(scratch_chain, snapshot)

    @staticmethod
    def _snapshot_chain(chain: ChainState) -> dict:
        """Capture the parts of a chain that calibration would disturb."""
        est = chain.estimator
        return {
            "placement": est.placement.clone(),
            "all_owner_pos": est.all_owner_pos.clone(),
            "density_grid": est.density_grid.clone(),
            "congestion_grid": est.congestion_grid.clone(),
            "net_hpwls": est.net_hpwls.clone(),
            "total_weighted_hpwl": est.total_weighted_hpwl,
            "total_overlap_area": est.total_overlap_area,
            "overlap_count": est.overlap_count,
            "current_cost": chain.current_cost,
        }

    @staticmethod
    def _restore_chain(chain: ChainState, snap: dict):
        est = chain.estimator
        est.placement = snap["placement"]
        est.all_owner_pos = snap["all_owner_pos"]
        est.density_grid = snap["density_grid"]
        est.congestion_grid = snap["congestion_grid"]
        est.net_hpwls = snap["net_hpwls"]
        est.total_weighted_hpwl = snap["total_weighted_hpwl"]
        est.total_overlap_area = snap["total_overlap_area"]
        est.overlap_count = snap["overlap_count"]
        chain.current_cost = snap["current_cost"]

    # -------------------------------------------------------------- internals

    def _init_chains(
        self,
        primary_placement: torch.Tensor,
        benchmark: Benchmark,
        extra_starts: Optional[Sequence[torch.Tensor]],
    ) -> List[ChainState]:
        """
        Build R independent ChainState objects.

        The first `1 + len(extra_starts)` chains get distinct starting
        placements (primary + extras). The rest start from the primary.
        That way PT swaps mix basins as well as temperatures.
        """
        chains: List[ChainState] = []
        starts = [primary_placement]
        if extra_starts:
            starts.extend(extra_starts)

        for ci in range(self.num_chains):
            start = starts[ci] if ci < len(starts) else primary_placement
            # Each chain owns its own CostEstimator with its own positions,
            # density/congestion grids, and HPWL cache. They never share
            # state; PT swap is the only cross-chain interaction.
            est = self._build_estimator_from_placement(start, benchmark)

            cost0 = est.get_costs()["proxy_cost"]
            chains.append(
                ChainState(
                    estimator=est,
                    temperature=self.temperatures[ci],
                    step_frac=0.02,  # placeholder; refine() overwrites after calibration
                    current_cost=cost0,
                    best_cost=cost0,
                    best_placement=est.placement.clone(),
                )
            )
        return chains

    @staticmethod
    def _build_estimator_from_placement(
        placement: torch.Tensor, benchmark: Benchmark
    ) -> CostEstimator:
        """
        Build a CostEstimator initialized to `placement` rather than
        benchmark.macro_positions.
        """
        # CostEstimator reads benchmark.macro_positions at construction.
        # We temporarily swap it, build, then put the original back.
        original = benchmark.macro_positions
        try:
            benchmark.macro_positions = placement.clone()
            est = CostEstimator(benchmark)
        finally:
            benchmark.macro_positions = original
        return est

    # ---- single-chain step ---------------------------------------------

    def _step_chain(
        self,
        chain: ChainState,
        benchmark: Benchmark,
        movable_hard: List[int],
        rng: random.Random,
    ):
        """
        Propose one move on this chain. Snapshot the state, try it,
        accept-or-reject by Metropolis.
        """
        chain.moves_proposed += 1
        move_type = self._pick_move(rng)

        if move_type == 0:  # translate
            ok = self._try_translate(chain, benchmark, movable_hard, rng)
        elif move_type == 1:  # pair-swap
            ok = self._try_swap(chain, benchmark, movable_hard, rng)
        else:                # block
            ok = self._try_block_move(chain, benchmark, movable_hard, rng)

        return ok

    def _pick_move(self, rng: random.Random) -> int:
        r = rng.random()
        for i, c in enumerate(self._move_cdf):
            if r < c:
                return i
        return len(self._move_cdf) - 1

    # ---- move implementations ----------------------------------------

    def _try_translate(self, chain, benchmark, movable_hard, rng) -> bool:
        """Gaussian step around current position."""
        idx = rng.choice(movable_hard)
        est = chain.estimator
        old_pos = est.placement[idx].clone()

        # Step size set per-chain at calibration. See refine() for the formula.
        sf = chain.step_frac
        step_x = rng.gauss(0, 1) * benchmark.canvas_width * sf
        step_y = rng.gauss(0, 1) * benchmark.canvas_height * sf

        w, h = benchmark.macro_sizes[idx]
        new_x = float(max(w / 2, min(benchmark.canvas_width - w / 2, old_pos[0] + step_x)))
        new_y = float(max(h / 2, min(benchmark.canvas_height - h / 2, old_pos[1] + step_y)))
        new_pos = torch.tensor([new_x, new_y])

        return self._apply_position_move(
            chain, [(idx, old_pos, new_pos)], rng
        )

    def _try_swap(self, chain, benchmark, movable_hard, rng) -> bool:
        """Swap centers of two movable hard macros."""
        if len(movable_hard) < 2:
            return False
        i, j = rng.sample(movable_hard, 2)
        est = chain.estimator
        old_i = est.placement[i].clone()
        old_j = est.placement[j].clone()

        # Check the swap doesn't drop either macro out of canvas (would
        # happen if they have very different sizes and one was near an
        # edge). We just bounds-check the proposed new centers.
        wi, hi = benchmark.macro_sizes[i]
        wj, hj = benchmark.macro_sizes[j]
        if (old_j[0] - wi / 2 < 0 or old_j[0] + wi / 2 > benchmark.canvas_width or
            old_j[1] - hi / 2 < 0 or old_j[1] + hi / 2 > benchmark.canvas_height or
            old_i[0] - wj / 2 < 0 or old_i[0] + wj / 2 > benchmark.canvas_width or
            old_i[1] - hj / 2 < 0 or old_i[1] + hj / 2 > benchmark.canvas_height):
            chain.rejected_oob += 1
            return False

        return self._apply_position_move(
            chain, [(i, old_i, old_j.clone()), (j, old_j, old_i.clone())], rng
        )

    def _try_block_move(self, chain, benchmark, movable_hard, rng) -> bool:
        """
        Translate a small cluster of nearby macros together.

        Pick a seed macro, find its K-1 nearest movable hard neighbors,
        and translate them all by the same Gaussian step.
        """
        if len(movable_hard) < 2:
            return False
        seed_idx = rng.choice(movable_hard)
        est = chain.estimator
        seed_pos = est.placement[seed_idx]

        # K nearest movable hard macros (incl seed) by L2 distance.
        movable_t = torch.tensor(movable_hard, dtype=torch.long)
        positions = est.placement[movable_t]
        d2 = ((positions - seed_pos) ** 2).sum(dim=1)
        k = min(self.block_k, len(movable_hard))
        _, nearest = torch.topk(d2, k, largest=False)
        block_indices = movable_t[nearest].tolist()

        # Common displacement; block moves use the same per-chain step
        # as single-macro translate.
        sf = chain.step_frac
        dx = rng.gauss(0, 1) * benchmark.canvas_width * sf
        dy = rng.gauss(0, 1) * benchmark.canvas_height * sf

        moves = []
        for idx in block_indices:
            old_pos = est.placement[idx].clone()
            w, h = benchmark.macro_sizes[idx]
            new_x = float(old_pos[0]) + dx
            new_y = float(old_pos[1]) + dy
            # If any macro in the block would leave canvas, reject the
            # whole move — partial blocks lose the "translate as one"
            # property we wanted.
            if (new_x - w / 2 < 0 or new_x + w / 2 > benchmark.canvas_width or
                new_y - h / 2 < 0 or new_y + h / 2 > benchmark.canvas_height):
                chain.rejected_oob += 1
                return False
            moves.append((idx, old_pos, torch.tensor([new_x, new_y])))

        return self._apply_position_move(chain, moves, rng)

    # ---- shared position-move acceptance path -----------------------

    def _apply_position_move(self, chain, moves, rng) -> bool:
        """
        Apply a list of (idx, old_pos, new_pos) updates atomically.

        Steps:
          1. Snapshot the things update_macro_pos changes (placement,
             density grid, congestion grid, hpwls, overlap count).
          2. Apply each move.
          3. Hard-reject if final overlap_count > 0 (we started legal,
             can't go illegal).
          4. Metropolis-test the cost delta. Accept or roll back.
        """
        est = chain.estimator

        # Snapshot. Density / congestion grids are R*C tensors so this
        # is a few KB — cheap. net_hpwls is per-net but we only need to
        # snapshot the affected ones.
        affected_nets = set()
        for idx, _, _ in moves:
            affected_nets.update(est.macro_to_nets[idx])
        old_net_hpwls = {n: est.net_hpwls[n].item() for n in affected_nets}
        old_total_hpwl = est.total_weighted_hpwl
        old_density = est.density_grid.clone()
        old_congestion = est.congestion_grid.clone()
        old_overlap_area = est.total_overlap_area
        old_overlap_count = est.overlap_count
        old_placement = {idx: est.placement[idx].clone() for idx, _, _ in moves}
        old_all_owner_pos = {idx: est.all_owner_pos[idx].clone() for idx, _, _ in moves}
        old_cost = chain.current_cost

        # Apply moves.
        for idx, _, new_pos in moves:
            est.update_macro_pos(idx, new_pos)

        # Hard reject if overlaps appeared.
        if est.overlap_count > 0:
            # Rollback.
            for idx, _, _ in moves:
                est.placement[idx] = old_placement[idx]
                est.all_owner_pos[idx] = old_all_owner_pos[idx]
            est.density_grid = old_density
            est.congestion_grid = old_congestion
            est.total_overlap_area = old_overlap_area
            est.overlap_count = old_overlap_count
            for n in affected_nets:
                est.net_hpwls[n] = torch.tensor(old_net_hpwls[n])
            est.total_weighted_hpwl = old_total_hpwl
            chain.rejected_overlap += 1
            return False

        new_cost = est.get_costs()["proxy_cost"]
        delta = new_cost - old_cost

        if self._metropolis_accept(delta, chain.temperature, rng):
            chain.current_cost = new_cost
            chain.moves_accepted += 1
            if new_cost < chain.best_cost:
                chain.best_cost = new_cost
                chain.best_placement = est.placement.clone()
            return True
        else:
            # Rollback to pre-move state.
            for idx, _, _ in moves:
                est.placement[idx] = old_placement[idx]
                est.all_owner_pos[idx] = old_all_owner_pos[idx]
            est.density_grid = old_density
            est.congestion_grid = old_congestion
            est.total_overlap_area = old_overlap_area
            est.overlap_count = old_overlap_count
            for n in affected_nets:
                est.net_hpwls[n] = torch.tensor(old_net_hpwls[n])
            est.total_weighted_hpwl = old_total_hpwl
            chain.rejected_metropolis += 1
            return False

    @staticmethod
    def _metropolis_accept(delta: float, temperature: float, rng: random.Random) -> bool:
        if delta <= 0:
            return True
        if temperature <= 0:
            return False
        # Clamp to avoid overflow in exp().
        ratio = -delta / temperature
        if ratio < -50:
            return False
        return rng.random() < math.exp(ratio)

    # ---- parallel tempering swap -----------------------------------

    def _pt_swap_step(self, chains: List[ChainState], rng: random.Random):
        """
        Attempt swaps between every adjacent pair of chains (sorted by
        temperature). Swap acceptance: min(1, exp((1/T_i - 1/T_j)(E_i - E_j)))
        where i,j are adjacent and E is the chain's current cost.
        """
        # Chains are already in T-ascending order from _init_chains
        # (temperatures[0]=t_max, temperatures[1]=lower, ...).
        # Adjacent pairs: (0,1), (1,2), ..., (R-2, R-1).
        for i in range(len(chains) - 1):
            a = chains[i]
            b = chains[i + 1]
            beta_diff = (1.0 / b.temperature) - (1.0 / a.temperature)
            energy_diff = a.current_cost - b.current_cost
            log_p = beta_diff * energy_diff
            if log_p >= 0 or rng.random() < math.exp(max(-50, log_p)):
                # Swap: every piece of mutable state. Note that we swap
                # the ChainState's *contents* in place to keep the
                # temperature ladder intact. After swapping, chain a
                # still has temperature T_a, but now holds the
                # configuration that was at T_b.
                self._exchange_states(a, b)

    @staticmethod
    def _exchange_states(a: ChainState, b: ChainState):
        """
        Swap configuration between two chains at different temperatures.

        We swap the estimator (which carries positions, grids, HPWLs) and
        the best-so-far. Temperatures stay attached to their chain — after
        a swap, chain a still has temperature T_a, but now holds the
        configuration that was at T_b.
        """
        a.estimator, b.estimator = b.estimator, a.estimator
        a.current_cost, b.current_cost = b.current_cost, a.current_cost
        a.best_cost, b.best_cost = b.best_cost, a.best_cost
        a.best_placement, b.best_placement = b.best_placement, a.best_placement


# ─── module-level smoke test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from macro_place.loader import load_benchmark_from_dir

    bench_dir = (
        Path(__file__).resolve().parents[1]
        / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    )
    if not bench_dir.exists():
        print(f"Skipping smoke test: {bench_dir} not present")
        sys.exit(0)

    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    sa = ParallelTemperingSA(
        num_chains=2,
        max_iterations=500,
        time_budget_s=30,
        verbose=True,
    )
    refined = sa.refine(benchmark.macro_positions.clone(), benchmark, plc=plc)
    print("Smoke test done; refined shape:", refined.shape)
