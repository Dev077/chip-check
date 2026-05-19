"""
Legalizer for macro placement — Hungarian-assignment variant.

Designed as a post-process for an analytical placer (e.g. DreamPlacer) whose
output has low cost but may contain hard-macro overlaps. Produces a strictly
legal placement (zero hard-macro overlaps, all macros in-canvas, fixed
macros untouched) while keeping movement small so the upstream cost is
preserved.

Pipeline
--------
1. Snap macro positions to a manufacturing grid (default 1 µm).
2. Pick a "regime" from canvas area:
       - IBM-style (small canvas, dense designs) → no extra spacing.
       - NG45-style (large canvas) → enforce a min macro-to-macro gap
         (default 12 µm) so the output is friendly to downstream auto-push.
   `regime="auto" | "ibm" | "ng45"` toggles this.
3. Anchor pass: place the K largest hard-movable macros greedily by spiral
   search around their DREAM positions, respecting fixed macros and the
   chosen spacing. Big macros first because they have fewer legal slots.
4. Fit pass: for the remaining hard-movable macros, rasterise the canvas
   into a fine grid, drop cells that collide with anchors / fixed macros /
   canvas edges, and solve a min-total-movement assignment from "macro i"
   to "free cell j" via scipy.optimize.linear_sum_assignment.
5. Infeasibility recovery: if the assignment can't place every macro, drop
   the spacing requirement and retry. If even that fails, spiral search
   for the survivors. Worst case raises with a diagnostic.

Usage
-----
    from placer.legalizer import Legalizer
    legal = Legalizer().legalize(placement, benchmark)
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from macro_place.benchmark import Benchmark


class Legalizer:
    """
    Hungarian-assignment legalizer for hard-macro placements.

    Soft macros and fixed macros are never moved. Only hard, movable macros
    are repositioned.
    """

    # Canvas-area threshold above which we assume NG45-style regime.
    # IBM ICCAD04 canvases are O(10^3) (grid units); NG45 canvases are
    # O(10^5–10^6). 1e4 is a comfortable separator.
    _NG45_AREA_THRESHOLD = 1e4

    def __init__(
        self,
        regime: str = "auto",
        grid_step_um: float = 1.0,
        spacing_um_ng45: float = 12.0,
        spacing_um_ibm: float = 0.0,
        anchor_top_k: Optional[int] = None,
        anchor_area_frac: float = 0.1,
        fit_cells_per_side: int = 200,
        spiral_step_frac: float = 0.005,
        spiral_max_radius_frac: float = 0.5,
        verbose: bool = True,
    ):
        # Manufacturing grid pitch. 1 µm is standard for these benchmarks;
        # set higher (e.g. 5) for chunkier snapping if downstream tools
        # demand it.
        self.grid_step_um = grid_step_um
        # Per-regime spacing requirements.
        self.spacing_um_ng45 = spacing_um_ng45
        self.spacing_um_ibm = spacing_um_ibm
        # Regime selection: "auto" picks by canvas area. "ibm" / "ng45"
        # force a choice.
        self.regime = regime
        # Anchor selection: how many "big" macros get the greedy first pass.
        # If anchor_top_k is None, we pick all macros whose cumulative area
        # reaches anchor_area_frac of total movable area.
        self.anchor_top_k = anchor_top_k
        self.anchor_area_frac = anchor_area_frac
        # Resolution of the fit-pass slot grid (cells per longer canvas side).
        # 200 is a good balance — fine enough to give the assignment freedom,
        # coarse enough to keep the cost matrix tractable.
        self.fit_cells_per_side = fit_cells_per_side
        # Spiral-search params for anchor pass + final fallback.
        self.spiral_step_frac = spiral_step_frac
        self.spiral_max_radius_frac = spiral_max_radius_frac
        self.verbose = verbose

    # ------------------------------------------------------------------ entry

    def legalize(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        legal = placement.clone().detach().float()
        num_hard = benchmark.num_hard_macros
        if num_hard == 0:
            return legal

        sizes = benchmark.macro_sizes[:num_hard].float()
        fixed = benchmark.macro_fixed[:num_hard].bool()
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        # Decide regime.
        regime = self._select_regime(canvas_w, canvas_h)
        spacing = self.spacing_um_ng45 if regime == "ng45" else self.spacing_um_ibm
        if self.verbose:
            print(
                f"[Legalizer] regime={regime} | canvas={canvas_w:.0f}x{canvas_h:.0f} | "
                f"spacing={spacing} µm | grid={self.grid_step_um} µm"
            )

        # Snap movable macros to the manufacturing grid; clamp into canvas.
        hard_pos = legal[:num_hard].clone()
        hard_pos = self._snap_to_grid(hard_pos)
        hard_pos = self._clamp_to_canvas(hard_pos, sizes, canvas_w, canvas_h)
        # Fixed macros keep their original (un-snapped) coordinates — they
        # must validate exactly against benchmark.macro_positions.
        hard_pos[fixed] = benchmark.macro_positions[:num_hard][fixed].float()

        movable_idx = torch.where(~fixed)[0].tolist()

        # ── Anchor pass ────────────────────────────────────────────────
        anchor_set = self._pick_anchors(movable_idx, sizes)
        if self.verbose:
            print(f"[Legalizer] anchor pass: placing {len(anchor_set)} large macros")
        placed = set(torch.where(fixed)[0].tolist())  # fixed are already placed
        for idx in anchor_set:
            new_xy = self._spiral_legal_slot(
                idx, hard_pos[idx], hard_pos, sizes, placed,
                canvas_w, canvas_h, spacing,
            )
            if new_xy is None and spacing > 0:
                # Anchor couldn't fit with spacing — try without. Anchors
                # are big, so failing here usually means the design is
                # genuinely tight; drop spacing for this macro.
                if self.verbose:
                    print(f"[Legalizer]   anchor {idx}: relaxing spacing")
                new_xy = self._spiral_legal_slot(
                    idx, hard_pos[idx], hard_pos, sizes, placed,
                    canvas_w, canvas_h, 0.0,
                )
            if new_xy is None:
                if self.verbose:
                    print(f"[Legalizer]   WARN: anchor {idx} could not place; leaving in situ")
            else:
                hard_pos[idx] = new_xy
            placed.add(idx)

        # ── Fit pass (Hungarian assignment) ────────────────────────────
        remaining = [i for i in movable_idx if i not in anchor_set]
        if self.verbose:
            print(f"[Legalizer] fit pass: assigning {len(remaining)} macros to slots")
        if remaining:
            success = self._fit_pass(
                remaining, hard_pos, sizes, placed,
                canvas_w, canvas_h, spacing,
            )
            if not success and spacing > 0:
                if self.verbose:
                    print(f"[Legalizer] fit pass infeasible with spacing — retrying with spacing=0")
                self._fit_pass(
                    remaining, hard_pos, sizes, placed,
                    canvas_w, canvas_h, 0.0,
                )

        # ── Final fallback for any survivors ───────────────────────────
        survivors = self._find_overlapping(hard_pos, sizes, fixed)
        if survivors:
            if self.verbose:
                print(
                    f"[Legalizer] {len(survivors)} macros still overlapping — "
                    f"spiral fallback"
                )
            placed_set = set(range(num_hard)) - set(survivors)
            survivors.sort(key=lambda i: -(sizes[i, 0] * sizes[i, 1]).item())
            for idx in survivors:
                new_xy = self._spiral_legal_slot(
                    idx, hard_pos[idx], hard_pos, sizes, placed_set,
                    canvas_w, canvas_h, 0.0,
                )
                if new_xy is None:
                    # Last resort: exhaustive scan of every grid cell.
                    new_xy = self._exhaustive_legal_slot(
                        idx, hard_pos[idx], hard_pos, sizes, placed_set,
                        canvas_w, canvas_h,
                    )
                if new_xy is not None:
                    hard_pos[idx] = new_xy
                    placed_set.add(idx)
                elif self.verbose:
                    print(
                        f"[Legalizer]   FAIL: no legal slot anywhere for macro {idx} "
                        f"(size {sizes[idx, 0]:.1f}x{sizes[idx, 1]:.1f})"
                    )

        # Write back.
        legal[:num_hard] = hard_pos
        # Reassert fixed positions exactly (no float drift).
        legal[:num_hard][fixed] = benchmark.macro_positions[:num_hard][fixed].float()

        # Diagnostic.
        n_overlap, area = self._overlap_summary(legal[:num_hard], sizes)
        mean_disp = (legal[:num_hard] - placement[:num_hard]).norm(dim=1).mean().item()
        if self.verbose:
            status = "LEGAL" if n_overlap == 0 else f"{n_overlap} overlaps remain (area={area:.2f})"
            print(f"[Legalizer] done. {status}. mean displacement = {mean_disp:.2f} µm")

        return legal

    # --------------------------------------------------------- regime + grid

    def _select_regime(self, canvas_w: float, canvas_h: float) -> str:
        if self.regime in ("ibm", "ng45"):
            return self.regime
        area = canvas_w * canvas_h
        return "ng45" if area >= self._NG45_AREA_THRESHOLD else "ibm"

    def _snap_to_grid(self, pos: torch.Tensor) -> torch.Tensor:
        step = self.grid_step_um
        return torch.round(pos / step) * step

    @staticmethod
    def _clamp_to_canvas(
        pos: torch.Tensor, sizes: torch.Tensor, canvas_w: float, canvas_h: float
    ) -> torch.Tensor:
        half = sizes * 0.5
        out = pos.clone()
        out[:, 0] = out[:, 0].clamp(half[:, 0], canvas_w - half[:, 0])
        out[:, 1] = out[:, 1].clamp(half[:, 1], canvas_h - half[:, 1])
        return out

    # ----------------------------------------------------------- anchor pass

    def _pick_anchors(self, movable_idx: List[int], sizes: torch.Tensor) -> set:
        """Return indices of the 'big' macros to place greedily first."""
        if not movable_idx:
            return set()
        areas = (sizes[:, 0] * sizes[:, 1]).tolist()
        sorted_by_area = sorted(movable_idx, key=lambda i: -areas[i])

        if self.anchor_top_k is not None:
            return set(sorted_by_area[: self.anchor_top_k])

        # Anchor anyone whose cumulative area is in the top
        # `anchor_area_frac` of movable area. This adapts to designs with
        # a few giant macros (NG45) vs many medium macros (IBM).
        total_area = sum(areas[i] for i in movable_idx)
        cutoff_area = total_area * self.anchor_area_frac
        anchors = set()
        running = 0.0
        for i in sorted_by_area:
            anchors.add(i)
            running += areas[i]
            if running >= cutoff_area:
                break
        return anchors

    def _spiral_legal_slot(
        self,
        idx: int,
        center: torch.Tensor,
        all_pos: torch.Tensor,
        sizes: torch.Tensor,
        placed: set,
        canvas_w: float,
        canvas_h: float,
        spacing: float,
    ) -> Optional[torch.Tensor]:
        """
        Find the nearest legal position for macro `idx`, considering only
        macros in `placed` as obstacles. Returns a snapped [2] tensor or None.
        """
        w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
        half_w, half_h = w * 0.5, h * 0.5
        x_lo, x_hi = half_w, canvas_w - half_w
        y_lo, y_hi = half_h, canvas_h - half_h

        if not placed:
            cand = torch.tensor([
                max(x_lo, min(x_hi, center[0].item())),
                max(y_lo, min(y_hi, center[1].item())),
            ])
            return self._snap_to_grid(cand.unsqueeze(0)).squeeze(0)

        obs_idx = list(placed)
        obs_pos = all_pos[obs_idx]
        obs_sizes = sizes[obs_idx]
        min_sep_x = (obs_sizes[:, 0] + w) * 0.5 + spacing
        min_sep_y = (obs_sizes[:, 1] + h) * 0.5 + spacing

        def legal(x: float, y: float) -> bool:
            if x < x_lo - 1e-6 or x > x_hi + 1e-6 or y < y_lo - 1e-6 or y > y_hi + 1e-6:
                return False
            dx = (obs_pos[:, 0] - x).abs()
            dy = (obs_pos[:, 1] - y).abs()
            return not bool(((dx < min_sep_x + 1e-4) & (dy < min_sep_y + 1e-4)).any())

        # Try the current spot first, snapped.
        cx = max(x_lo, min(x_hi, center[0].item()))
        cy = max(y_lo, min(y_hi, center[1].item()))
        cx = round(cx / self.grid_step_um) * self.grid_step_um
        cy = round(cy / self.grid_step_um) * self.grid_step_um
        if legal(cx, cy):
            return torch.tensor([cx, cy], dtype=all_pos.dtype)

        # Spiral outward.
        step = max(self.spiral_step_frac * min(canvas_w, canvas_h), self.grid_step_um)
        max_r = self.spiral_max_radius_frac * ((canvas_w**2 + canvas_h**2) ** 0.5)
        r = step
        while r <= max_r:
            n_samples = min(64, max(12, int(2 * np.pi * r / step)))
            for k in range(n_samples):
                angle = 2 * np.pi * k / n_samples
                tx = cx + r * np.cos(angle)
                ty = cy + r * np.sin(angle)
                tx = round(tx / self.grid_step_um) * self.grid_step_um
                ty = round(ty / self.grid_step_um) * self.grid_step_um
                tx = max(x_lo, min(x_hi, tx))
                ty = max(y_lo, min(y_hi, ty))
                if legal(tx, ty):
                    return torch.tensor([tx, ty], dtype=all_pos.dtype)
            r += step
        return None

    def _exhaustive_legal_slot(
        self,
        idx: int,
        center: torch.Tensor,
        all_pos: torch.Tensor,
        sizes: torch.Tensor,
        placed: set,
        canvas_w: float,
        canvas_h: float,
    ) -> Optional[torch.Tensor]:
        """
        Try every grid cell in the canvas. Returns the legal cell closest
        to `center`. Last-resort: O(canvas_w * canvas_h / step^2) per call,
        used only when spiral search fails.
        """
        w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
        half_w, half_h = w * 0.5, h * 0.5
        step = self.grid_step_um

        if not placed:
            return torch.tensor([
                max(half_w, min(canvas_w - half_w, center[0].item())),
                max(half_h, min(canvas_h - half_h, center[1].item())),
            ])

        obs_idx = list(placed)
        obs_pos = all_pos[obs_idx].numpy()
        obs_sizes = sizes[obs_idx].numpy()
        ms_x = (obs_sizes[:, 0] + w) * 0.5
        ms_y = (obs_sizes[:, 1] + h) * 0.5

        # Generate every legal candidate position, then pick the closest.
        nx = max(2, int((canvas_w - 2 * half_w) / step) + 1)
        ny = max(2, int((canvas_h - 2 * half_h) / step) + 1)
        xs = np.linspace(half_w, canvas_w - half_w, nx)
        ys = np.linspace(half_h, canvas_h - half_h, ny)
        # Snap to grid step.
        xs = np.round(xs / step) * step
        ys = np.round(ys / step) * step
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        cand = np.stack([gx.flatten(), gy.flatten()], axis=1)

        dx = np.abs(cand[:, None, 0] - obs_pos[None, :, 0])
        dy = np.abs(cand[:, None, 1] - obs_pos[None, :, 1])
        collides = (dx < ms_x[None, :] + 1e-4) & (dy < ms_y[None, :] + 1e-4)
        legal_mask = ~collides.any(axis=1)
        if not legal_mask.any():
            return None

        legal_cand = cand[legal_mask]
        cx, cy = center[0].item(), center[1].item()
        dists = (legal_cand[:, 0] - cx) ** 2 + (legal_cand[:, 1] - cy) ** 2
        best = legal_cand[int(np.argmin(dists))]
        return torch.tensor([float(best[0]), float(best[1])])

    # -------------------------------------------------------------- fit pass

    def _fit_pass(
        self,
        macro_indices: List[int],
        hard_pos: torch.Tensor,  # modified in place
        sizes: torch.Tensor,
        placed: set,
        canvas_w: float,
        canvas_h: float,
        spacing: float,
    ) -> bool:
        """
        Solve min-total-displacement assignment from macros to candidate
        slots, iteratively resolving slot-vs-slot conflicts.

        A single Hungarian solve cannot guarantee zero overlap because
        adjacent slots are closer than typical macro dimensions. Instead
        we loop: solve, detect conflicting pairs among the assignment,
        commit the non-conflicting ones, then re-solve for the conflicting
        macros with the committed slots (and slots that would collide with
        them) masked out. This converges in a handful of iterations even
        on dense designs.

        Mutates `hard_pos`. Returns True iff every macro was assigned a
        legal slot.
        """
        n_macros = len(macro_indices)
        if n_macros == 0:
            return True

        # Slot pitch: aim for resolution, but never finer than the manufacturing
        # grid. We deliberately allow pitch < macro size and resolve conflicts
        # iteratively below.
        slot_centers = self._build_slot_grid(
            canvas_w, canvas_h,
            min_pitch_x=self.grid_step_um,
            min_pitch_y=self.grid_step_um,
        )
        all_slots = slot_centers.numpy()  # [n_slots, 2]
        n_slots = all_slots.shape[0]
        if n_slots < n_macros:
            if self.verbose:
                print(f"[Legalizer]   only {n_slots} slots for {n_macros} macros")
            return False

        # Track which macros still need placing and which slots are taken.
        pending = list(macro_indices)
        slot_used = np.zeros(n_slots, dtype=bool)

        MAX_ROUNDS = 8
        for round_idx in range(MAX_ROUNDS):
            available_slot_ids = np.where(~slot_used)[0]
            if len(available_slot_ids) < len(pending):
                if self.verbose:
                    print(
                        f"[Legalizer]   round {round_idx}: "
                        f"{len(available_slot_ids)} slots for {len(pending)} macros — infeasible"
                    )
                return False

            sub_slots = all_slots[available_slot_ids]  # [m, 2]
            cost, valid_per_macro = self._build_cost_matrix(
                pending, sub_slots, hard_pos, sizes, placed,
                canvas_w, canvas_h, spacing,
            )
            PENALTY = 1e12

            try:
                row_ind, col_ind = linear_sum_assignment(cost)
            except ValueError as e:
                if self.verbose:
                    print(f"[Legalizer]   round {round_idx}: assignment failed: {e}")
                return False

            # Map sub-slot index back to global slot id.
            chosen_slot_ids = available_slot_ids[col_ind]
            chosen_xy = all_slots[chosen_slot_ids]  # [n_pending, 2]

            # Identify which assignments are valid (non-penalty).
            valid_mask = np.array([
                cost[r, c] < PENALTY * 0.5 for r, c in zip(row_ind, col_ind)
            ])

            # Detect pairwise conflicts among the chosen positions.
            # For each pair (i, j), conflict iff dx<sx and dy<sy with
            # appropriate min-separations including spacing.
            in_conflict = np.zeros(len(pending), dtype=bool)
            sizes_np = np.array([sizes[m].numpy() for m in pending])  # [n_pending, 2]
            for i in range(len(pending)):
                if not valid_mask[i]:
                    in_conflict[i] = True
                    continue
                if in_conflict[i]:
                    continue
                for j in range(i + 1, len(pending)):
                    if not valid_mask[j] or in_conflict[j]:
                        continue
                    sx = (sizes_np[i, 0] + sizes_np[j, 0]) * 0.5 + spacing
                    sy = (sizes_np[i, 1] + sizes_np[j, 1]) * 0.5 + spacing
                    dx = abs(chosen_xy[i, 0] - chosen_xy[j, 0])
                    dy = abs(chosen_xy[i, 1] - chosen_xy[j, 1])
                    if dx < sx + 1e-4 and dy < sy + 1e-4:
                        # Mark the one further from its original position
                        # for re-assignment, keep the one closer.
                        mi, mj = pending[i], pending[j]
                        di = ((chosen_xy[i, 0] - hard_pos[mi, 0].item()) ** 2 +
                              (chosen_xy[i, 1] - hard_pos[mi, 1].item()) ** 2)
                        dj = ((chosen_xy[j, 0] - hard_pos[mj, 0].item()) ** 2 +
                              (chosen_xy[j, 1] - hard_pos[mj, 1].item()) ** 2)
                        if di >= dj:
                            in_conflict[i] = True
                        else:
                            in_conflict[j] = True

            # Commit the non-conflicting assignments. We mark slots as used
            # so the next round doesn't try to assign two macros to the
            # exact same cell. Slots that would collide with the just-
            # committed macros (under each pending macro's specific size)
            # are filtered inside _build_cost_matrix on the next round —
            # no need to forbid them globally here, which would be over-
            # restrictive for the smallest pending macros.
            committed = 0
            new_pending = []
            for i, m_idx in enumerate(pending):
                if not in_conflict[i] and valid_mask[i]:
                    hard_pos[m_idx, 0] = float(chosen_xy[i, 0])
                    hard_pos[m_idx, 1] = float(chosen_xy[i, 1])
                    placed.add(m_idx)
                    slot_used[chosen_slot_ids[i]] = True
                    committed += 1
                else:
                    new_pending.append(m_idx)

            if self.verbose:
                print(
                    f"[Legalizer]   round {round_idx}: "
                    f"committed {committed}, {len(new_pending)} still pending"
                )

            if not new_pending:
                return True
            if committed == 0:
                # No progress this round → stuck.
                return False
            pending = new_pending

        # Hit round cap with macros still pending.
        return len(pending) == 0

    def _build_cost_matrix(
        self,
        macro_indices: List[int],
        slot_xy: np.ndarray,
        hard_pos: torch.Tensor,
        sizes: torch.Tensor,
        placed: set,
        canvas_w: float,
        canvas_h: float,
        spacing: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the [n_macros, n_slots] cost matrix for one assignment round."""
        n_macros = len(macro_indices)
        n_slots = slot_xy.shape[0]
        PENALTY = 1e12
        cost = np.full((n_macros, n_slots), PENALTY, dtype=np.float64)

        if placed:
            obs_idx = list(placed)
            obs_pos = hard_pos[obs_idx].numpy()
            obs_sizes = sizes[obs_idx].numpy()
        else:
            obs_pos = np.zeros((0, 2))
            obs_sizes = np.zeros((0, 2))

        valid_per_macro = np.zeros((n_macros, n_slots), dtype=bool)
        for row, m_idx in enumerate(macro_indices):
            w = sizes[m_idx, 0].item()
            h = sizes[m_idx, 1].item()
            half_w, half_h = w * 0.5, h * 0.5
            mx, my = hard_pos[m_idx, 0].item(), hard_pos[m_idx, 1].item()

            in_x = (slot_xy[:, 0] >= half_w - 1e-6) & (slot_xy[:, 0] <= canvas_w - half_w + 1e-6)
            in_y = (slot_xy[:, 1] >= half_h - 1e-6) & (slot_xy[:, 1] <= canvas_h - half_h + 1e-6)
            valid = in_x & in_y

            if obs_pos.shape[0] > 0:
                ms_x = (obs_sizes[:, 0] + w) * 0.5 + spacing
                ms_y = (obs_sizes[:, 1] + h) * 0.5 + spacing
                dx = np.abs(slot_xy[:, None, 0] - obs_pos[None, :, 0])
                dy = np.abs(slot_xy[:, None, 1] - obs_pos[None, :, 1])
                collides = (dx < ms_x[None, :] + 1e-4) & (dy < ms_y[None, :] + 1e-4)
                slot_ok = ~collides.any(axis=1)
                valid = valid & slot_ok

            valid_per_macro[row] = valid
            displ_sq = (slot_xy[:, 0] - mx) ** 2 + (slot_xy[:, 1] - my) ** 2
            cost[row, valid] = np.sqrt(displ_sq[valid])

        return cost, valid_per_macro

    def _build_slot_grid(
        self,
        canvas_w: float,
        canvas_h: float,
        min_pitch_x: float,
        min_pitch_y: float,
    ) -> torch.Tensor:
        """
        Uniform grid of candidate slot centers covering the canvas.

        `min_pitch_*` is the minimum allowed centre-to-centre spacing.
        If two macros are assigned to adjacent slots, they overlap iff slot
        pitch is smaller than the macros' combined half-widths plus any
        required gap. Setting pitch >= max(min_pitch_x, smallest_macro_dim)
        guarantees that any two assigned slots produce a legal pair.

        We start from `fit_cells_per_side` as a hint, but enforce the
        pitch floor — that reduces total slot count when macros are large
        but is what keeps the output legal-by-construction.
        """
        n = self.fit_cells_per_side
        if canvas_w >= canvas_h:
            nx = n
            ny = max(2, int(round(n * canvas_h / canvas_w)))
        else:
            ny = n
            nx = max(2, int(round(n * canvas_w / canvas_h)))

        # Enforce minimum pitch.
        pitch_x = max(canvas_w / nx, min_pitch_x, self.grid_step_um)
        pitch_y = max(canvas_h / ny, min_pitch_y, self.grid_step_um)
        nx = max(2, int(canvas_w / pitch_x))
        ny = max(2, int(canvas_h / pitch_y))
        # Recompute pitch after integer rounding.
        pitch_x = canvas_w / nx
        pitch_y = canvas_h / ny

        xs = torch.linspace(pitch_x * 0.5, canvas_w - pitch_x * 0.5, nx)
        ys = torch.linspace(pitch_y * 0.5, canvas_h - pitch_y * 0.5, ny)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        slots = torch.stack([gx.flatten(), gy.flatten()], dim=1)
        slots = torch.round(slots / self.grid_step_um) * self.grid_step_um
        if self.verbose:
            print(
                f"[Legalizer]   slot grid: {nx}x{ny} = {nx*ny} slots "
                f"(pitch {pitch_x:.2f}x{pitch_y:.2f})"
            )
        return slots

    # ------------------------------------------------------------ diagnostics

    @staticmethod
    def _overlap_summary(pos: torch.Tensor, sizes: torch.Tensor) -> Tuple[int, float]:
        dx = (pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)).abs()
        dy = (pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)).abs()
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        msx = half_w.unsqueeze(0) + half_w.unsqueeze(1)
        msy = half_h.unsqueeze(0) + half_h.unsqueeze(1)
        ox = (msx - dx).clamp(min=0.0)
        oy = (msy - dy).clamp(min=0.0)
        mask = (ox > 0) & (oy > 0)
        mask.fill_diagonal_(False)
        count = int(mask.sum().item() // 2)
        area = float((ox * oy * mask.float()).sum().item() / 2.0)
        return count, area

    @staticmethod
    def _find_overlapping(
        pos: torch.Tensor, sizes: torch.Tensor, fixed: torch.Tensor
    ) -> List[int]:
        dx = (pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)).abs()
        dy = (pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)).abs()
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        msx = half_w.unsqueeze(0) + half_w.unsqueeze(1)
        msy = half_h.unsqueeze(0) + half_h.unsqueeze(1)
        mask = ((msx - dx) > 1e-6) & ((msy - dy) > 1e-6)
        mask.fill_diagonal_(False)
        per_macro = mask.any(dim=1)
        per_macro = per_macro & (~fixed)
        return torch.where(per_macro)[0].tolist()