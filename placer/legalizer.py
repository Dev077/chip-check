"""
Legalizer for macro placement.

Designed to be called after an analytical placer (e.g. DreamPlacer) produces
a continuous, low-cost placement that may contain hard-macro overlaps.

Goal: produce a strictly legal placement (zero hard-macro overlaps, all
macros in-canvas, fixed macros untouched) while staying as close as possible
to the input positions, so the wirelength/density/congestion gains made by
the analytical placer are preserved.

Strategy: two phases.

  Phase 1 — Iterative force-directed spreading (vectorized).
      For every overlapping pair (i, j) of movable hard macros, compute the
      smaller of (x-overlap, y-overlap) and push the two macros apart along
      that axis. Pushes are accumulated as a displacement field, scaled by a
      damping factor, applied, then clamped to canvas bounds. Fixed macros
      are treated as immovable obstacles (they push other macros but never
      move themselves). This is cheap (O(N^2) pair test per iter, all
      vectorized) and tends to fix the vast majority of conflicts with very
      small per-macro displacement, preserving the analytical placer's cost.

  Phase 2 — Greedy nearest-legal-slot fallback.
      Whatever Phase 1 could not resolve (typically a handful of stubborn
      conflicts in dense regions) is fixed deterministically: for each macro
      still in conflict, search outward from its current position on a
      coarse grid for the nearest empty slot and snap it there. This phase
      guarantees a legal output even if Phase 1 stalls.

Usage:
    from macro_place.legalizer import Legalizer
    placer = DreamPlacer(...)
    raw = placer.place(benchmark)
    legal = Legalizer().legalize(raw, benchmark)
"""

from typing import Tuple

import torch

from macro_place.benchmark import Benchmark


class Legalizer:
    """
    Minimum-perturbation legalizer for hard-macro placements.

    Run as a post-processing step on the output of any continuous-domain
    placer. Only hard, movable macros are repositioned; fixed macros and
    soft macros are left exactly where the input placement put them.
    """

    def __init__(
        self,
        max_spread_iters: int = 500,
        damping: float = 0.5,
        convergence_eps: float = 1e-4,
        max_fallback_radius_frac: float = 0.5,
        fallback_grid_step_frac: float = 0.01,
        verbose: bool = True,
    ):
        # Phase-1 iteration cap. 500 is more than enough for IBM-scale designs;
        # most benchmarks converge in <50.
        self.max_spread_iters = max_spread_iters
        # Fraction of the per-pair penetration to apply per step. 0.5 = each
        # macro absorbs half the push. Values <1 add stability; values >=1
        # can oscillate when many pairs disagree.
        self.damping = damping
        # If the global penetration sum drops below this for several iters,
        # Phase 1 declares convergence and hands off (legal or not) to Phase 2.
        self.convergence_eps = convergence_eps
        # Phase 2 spiral search caps at this fraction of the canvas diagonal.
        self.max_fallback_radius_frac = max_fallback_radius_frac
        # Grid step for the spiral search, as a fraction of canvas dimension.
        self.fallback_grid_step_frac = fallback_grid_step_frac
        self.verbose = verbose

    # ------------------------------------------------------------------ public

    def legalize(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Return a legal placement closest to *placement*."""
        legal = placement.clone().detach().float()
        num_hard = benchmark.num_hard_macros

        if num_hard <= 1:
            return legal  # Nothing to legalize.

        sizes = benchmark.macro_sizes[:num_hard].float()
        fixed = benchmark.macro_fixed[:num_hard].bool()
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        # Clamp everything into bounds up front so Phase 1 starts from a
        # canvas-legal state. The analytical placer should already respect
        # this, but enforcing it here makes the legalizer safe to call on
        # any input.
        legal[:num_hard] = self._clamp_to_canvas(legal[:num_hard], sizes, canvas_w, canvas_h)

        initial_count, initial_area = self._overlap_summary(legal[:num_hard], sizes)
        if initial_count == 0:
            if self.verbose:
                print(f"[Legalizer] Input already legal (0 overlap pairs). No-op.")
            return legal

        if self.verbose:
            print(
                f"[Legalizer] Input has {initial_count} overlap pairs, "
                f"total area {initial_area:.2f} um^2. Starting spread."
            )

        # ── Phase 1: vectorized spreading ────────────────────────────────
        legal[:num_hard], phase1_count = self._spread(
            legal[:num_hard], sizes, fixed, canvas_w, canvas_h
        )

        # ── Phase 2: deterministic fallback if any pairs remain ─────────
        if phase1_count > 0:
            if self.verbose:
                print(
                    f"[Legalizer] {phase1_count} pairs survived spreading. "
                    f"Running greedy nearest-slot fallback."
                )
            legal[:num_hard] = self._greedy_repack(
                legal[:num_hard], sizes, fixed, canvas_w, canvas_h
            )

        # Final sanity check.
        final_count, final_area = self._overlap_summary(legal[:num_hard], sizes)
        if self.verbose:
            status = "LEGAL" if final_count == 0 else f"{final_count} overlaps remain"
            print(
                f"[Legalizer] Done. {status}. "
                f"Total displacement vs input: "
                f"{self._mean_displacement(placement[:num_hard], legal[:num_hard]):.2f} um (mean)"
            )

        # Reassert fixed-macro positions exactly — guards against any
        # float drift introduced by clamping or update arithmetic.
        legal[:num_hard][fixed] = benchmark.macro_positions[:num_hard][fixed].float()

        return legal

    # ----------------------------------------------------------------- Phase 1

    def _spread(
        self,
        pos: torch.Tensor,
        sizes: torch.Tensor,
        fixed: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
    ) -> Tuple[torch.Tensor, int]:
        """
        Iteratively push overlapping macros apart along the cheapest axis.

        Returns the updated positions and the remaining pair count.
        """
        n = pos.shape[0]
        movable = ~fixed

        # Precompute pairwise minimum separations (depends only on sizes).
        # min_sep_x[i, j] = (w_i + w_j) / 2, same for y.
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        min_sep_x = half_w.unsqueeze(0) + half_w.unsqueeze(1)  # [n, n]
        min_sep_y = half_h.unsqueeze(0) + half_h.unsqueeze(1)  # [n, n]

        stall_counter = 0
        last_total_penetration = float("inf")
        remaining = -1

        for it in range(self.max_spread_iters):
            # Pairwise center-to-center signed deltas (j relative to i).
            dx = pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)  # [n, n]
            dy = pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)

            # Penetration along each axis (clamped at 0 — no penetration if separated).
            pen_x = (min_sep_x - dx.abs()).clamp(min=0.0)
            pen_y = (min_sep_y - dy.abs()).clamp(min=0.0)

            # A pair overlaps only if BOTH axes penetrate.
            overlap_mask = (pen_x > 0) & (pen_y > 0)
            overlap_mask.fill_diagonal_(False)

            total_penetration = (pen_x * pen_y * overlap_mask.float()).sum().item()
            n_pairs = int(overlap_mask.sum().item() // 2)

            if n_pairs == 0:
                remaining = 0
                if self.verbose and (it + 1) % 25 != 0:
                    print(f"[Legalizer]   spread converged at iter {it} (no overlaps).")
                return pos, 0

            # Push along the *cheaper* axis per pair: pushing on the axis
            # with the smaller penetration moves macros less. This is the
            # key trick for minimum-displacement legalization.
            push_on_x = pen_x <= pen_y  # [n, n] bool
            push_on_x &= overlap_mask
            push_on_y = (~push_on_x) & overlap_mask

            # Direction sign: if dx > 0, j is to the right of i, so i moves
            # left (-) and j moves right (+). We compute the displacement
            # *applied to i* from each neighbour j; the j side is just the
            # negation, handled by the symmetric sum below.
            sign_x = torch.sign(dx)
            sign_y = torch.sign(dy)
            # If centers exactly coincide, sign is 0 — pick an arbitrary
            # direction (right/up) to break the tie and let the next iter
            # propagate.
            sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
            sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)

            # Per-pair half-pushes assigned to macro i (j's side is symmetric).
            # i needs to move AWAY from j, i.e. in -sign(j - i) direction.
            push_ix = -sign_x * pen_x * push_on_x.float() * 0.5
            push_iy = -sign_y * pen_y * push_on_y.float() * 0.5

            # Sum contributions over all conflicting neighbours.
            # push_ix[i, j] is the push on i from neighbour j → sum over j (dim=1).
            dx_total = push_ix.sum(dim=1) * self.damping
            dy_total = push_iy.sum(dim=1) * self.damping

            # Freeze fixed macros — they push others but never move.
            dx_total = torch.where(movable, dx_total, torch.zeros_like(dx_total))
            dy_total = torch.where(movable, dy_total, torch.zeros_like(dy_total))

            # When a movable macro conflicts with a fixed one, the fixed
            # side contributes 0 (because of the line above for itself),
            # so the movable macro only absorbs *its* half of the push.
            # Double its share so the relative displacement matches the
            # penetration.
            fixed_neighbour_x = (push_on_x & fixed.unsqueeze(0)).any(dim=1)
            fixed_neighbour_y = (push_on_y & fixed.unsqueeze(0)).any(dim=1)
            # Only boost where the macro itself is movable.
            boost_x = movable & fixed_neighbour_x
            boost_y = movable & fixed_neighbour_y
            dx_total = torch.where(boost_x, dx_total * 2.0, dx_total)
            dy_total = torch.where(boost_y, dy_total * 2.0, dy_total)

            pos = pos.clone()
            pos[:, 0] += dx_total
            pos[:, 1] += dy_total
            pos = self._clamp_to_canvas(pos, sizes, canvas_w, canvas_h)

            # Convergence check: if total penetration stops decreasing for
            # 5 iters, give up and hand off to Phase 2.
            improvement = last_total_penetration - total_penetration
            if abs(improvement) < self.convergence_eps:
                stall_counter += 1
            else:
                stall_counter = 0
            last_total_penetration = total_penetration

            if self.verbose and (it + 1) % 25 == 0:
                print(
                    f"[Legalizer]   iter {it + 1:4d} | pairs={n_pairs:4d} | "
                    f"pen_area={total_penetration:.2f}"
                )

            if stall_counter >= 5:
                remaining = n_pairs
                if self.verbose:
                    print(f"[Legalizer]   spread stalled at iter {it} ({n_pairs} pairs left).")
                return pos, n_pairs

        # Hit iteration cap.
        _, remaining_count = self._overlap_summary(pos, sizes)
        return pos, remaining_count

    # ----------------------------------------------------------------- Phase 2

    def _greedy_repack(
        self,
        pos: torch.Tensor,
        sizes: torch.Tensor,
        fixed: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
    ) -> torch.Tensor:
        """
        For each macro still in conflict, snap it to the nearest empty slot
        via outward spiral search. Deterministic and guaranteed legal *if*
        such a slot exists — i.e. as long as the total macro area fits in
        the canvas with some breathing room.

        Process macros in descending area order: larger macros are harder to
        place, so giving them first pick of free space avoids wasted work.
        """
        n = pos.shape[0]
        pos = pos.clone()

        # Identify conflicting movable macros.
        conflicting = self._conflict_mask(pos, sizes)
        conflicting = conflicting & (~fixed)
        conflict_indices = torch.where(conflicting)[0].tolist()

        # Sort by area, largest first.
        areas = (sizes[:, 0] * sizes[:, 1]).tolist()
        conflict_indices.sort(key=lambda i: -areas[i])

        step_x = canvas_w * self.fallback_grid_step_frac
        step_y = canvas_h * self.fallback_grid_step_frac
        max_radius = (canvas_w**2 + canvas_h**2) ** 0.5 * self.max_fallback_radius_frac

        for idx in conflict_indices:
            new_xy = self._find_nearest_legal(
                idx, pos, sizes, fixed, canvas_w, canvas_h, step_x, step_y, max_radius
            )
            if new_xy is not None:
                pos[idx] = new_xy
            elif self.verbose:
                print(
                    f"[Legalizer]   WARN: no legal slot found within radius for macro {idx}."
                )

        return pos

    def _find_nearest_legal(
        self,
        idx: int,
        pos: torch.Tensor,
        sizes: torch.Tensor,
        fixed: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
        step_x: float,
        step_y: float,
        max_radius: float,
    ):
        """
        Outward spiral search for the nearest position where macro idx does
        not overlap any other hard macro. Returns a [2] tensor or None.
        """
        cur = pos[idx].clone()
        w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
        half_w, half_h = w * 0.5, h * 0.5
        x_lo, x_hi = half_w, canvas_w - half_w
        y_lo, y_hi = half_h, canvas_h - half_h

        # Other macros' centers and sizes.
        others_mask = torch.ones(pos.shape[0], dtype=torch.bool)
        others_mask[idx] = False
        others_pos = pos[others_mask]
        others_sizes = sizes[others_mask]
        others_min_sep_x = (others_sizes[:, 0] + w) * 0.5
        others_min_sep_y = (others_sizes[:, 1] + h) * 0.5

        def is_legal(x: float, y: float) -> bool:
            if x < x_lo or x > x_hi or y < y_lo or y > y_hi:
                return False
            dx = (others_pos[:, 0] - x).abs()
            dy = (others_pos[:, 1] - y).abs()
            return not bool(((dx < others_min_sep_x) & (dy < others_min_sep_y)).any())

        # Try the current spot first (often the spread phase got it close).
        if is_legal(cur[0].item(), cur[1].item()):
            return cur

        # Spiral outward in rings of growing radius.
        radius = max(step_x, step_y)
        while radius <= max_radius:
            # Sample ~16 candidate points per ring; denser rings as radius
            # grows would be overkill since step size already bounds error.
            n_samples = max(8, int(2 * 3.14159 * radius / max(step_x, step_y)))
            n_samples = min(n_samples, 64)  # cap for speed
            for k in range(n_samples):
                angle = 2 * 3.14159 * k / n_samples
                cand_x = cur[0].item() + radius * torch.cos(torch.tensor(angle)).item()
                cand_y = cur[1].item() + radius * torch.sin(torch.tensor(angle)).item()
                # Clamp to bounds before testing.
                cand_x = max(x_lo, min(x_hi, cand_x))
                cand_y = max(y_lo, min(y_hi, cand_y))
                if is_legal(cand_x, cand_y):
                    return torch.tensor([cand_x, cand_y], dtype=pos.dtype)
            radius += max(step_x, step_y)

        return None

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _clamp_to_canvas(
        pos: torch.Tensor, sizes: torch.Tensor, canvas_w: float, canvas_h: float
    ) -> torch.Tensor:
        """Clamp macro centers so the full macro stays inside the canvas."""
        half = sizes * 0.5
        pos = pos.clone()
        pos[:, 0] = pos[:, 0].clamp(half[:, 0], canvas_w - half[:, 0])
        pos[:, 1] = pos[:, 1].clamp(half[:, 1], canvas_h - half[:, 1])
        return pos

    @staticmethod
    def _overlap_summary(pos: torch.Tensor, sizes: torch.Tensor) -> Tuple[int, float]:
        """Return (overlap_pair_count, total_overlap_area)."""
        dx = (pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)).abs()
        dy = (pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)).abs()
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        min_sep_x = half_w.unsqueeze(0) + half_w.unsqueeze(1)
        min_sep_y = half_h.unsqueeze(0) + half_h.unsqueeze(1)
        ox = (min_sep_x - dx).clamp(min=0.0)
        oy = (min_sep_y - dy).clamp(min=0.0)
        mask = (ox > 0) & (oy > 0)
        mask.fill_diagonal_(False)
        count = int(mask.sum().item() // 2)
        area = float((ox * oy * mask.float()).sum().item() / 2.0)
        return count, area

    @staticmethod
    def _conflict_mask(pos: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        """Per-macro bool mask: True if the macro overlaps at least one other."""
        dx = (pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)).abs()
        dy = (pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)).abs()
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        min_sep_x = half_w.unsqueeze(0) + half_w.unsqueeze(1)
        min_sep_y = half_h.unsqueeze(0) + half_h.unsqueeze(1)
        mask = ((min_sep_x - dx) > 0) & ((min_sep_y - dy) > 0)
        mask.fill_diagonal_(False)
        return mask.any(dim=1)

    @staticmethod
    def _mean_displacement(before: torch.Tensor, after: torch.Tensor) -> float:
        diff = after - before
        return float((diff[:, 0] ** 2 + diff[:, 1] ** 2).sqrt().mean().item())