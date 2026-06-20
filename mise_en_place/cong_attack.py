"""
Phase 2.5 — Direct congestion attack (true grid).

Bit-exact congestion attack using FastEvaluator: identify hot cells from the
smoothed congestion grid, find macros whose incident-net bboxes touch them,
and grid-sweep each macro to drop the proxy.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from .evaluator import FastEvaluator
from .lk import _slide_legal


# ────────────────────────────────────────────────────────────────────────────
# Phase 2.5 — Direct congestion-attack (true grid)
# ────────────────────────────────────────────────────────────────────────────


def _smoothed_congestion_grid(ev: FastEvaluator) -> np.ndarray:
    """Compute the FastEvaluator's smoothed combined V+H congestion grid.

    This is the SAME math that produces the top-5% mean in `_congestion_cost`,
    exposed for the direct-attack phase that wants to know WHICH cells are hot.
    """
    v = ev.v_pin_cong / ev.grid_v_routes
    h = ev.h_pin_cong / ev.grid_h_routes
    vm = ev.v_macro_cong / ev.grid_v_routes
    hm = ev.h_macro_cong / ev.grid_h_routes
    v_s = ev._smooth(v, axis=0)
    h_s = ev._smooth(h, axis=1)
    return (v_s + vm) + (h_s + hm)


def direct_congestion_attack(
    ev: FastEvaluator,
    n_passes: int = 3,
    time_budget_s: float = 60.0,
    sweep_steps: int = 5,
    sweep_radius_frac: float = 0.06,
    n_top_cells: int = 24,
    verbose: bool = True,
):
    """Bit-exact congestion attack using FastEvaluator.

    Each pass:
      1. Compute the current smoothed congestion grid.
      2. Identify the top-`n_top_cells` hottest cells (these dominate the
         top-5% mean that defines congestion cost).
      3. For each hot cell, find every hard macro whose incident-net bbox
         touches the cell.  Score macros by sum of touching-cell heat.
      4. Process macros in priority order: 5×5 grid-sweep around current
         position, accept best move that strictly improves the proxy.
    Each move is evaluated against the bit-exact proxy.
    """
    cur_cost = ev.proxy_cost()["proxy_cost"]
    start_cost = cur_cost
    best_cost = cur_cost
    best_pos = ev.positions.copy()
    t0 = time.time()
    accepted_total = 0
    for pass_idx in range(n_passes):
        if time.time() - t0 > time_budget_s:
            break
        cong = _smoothed_congestion_grid(ev)
        flat = cong.ravel()
        n = flat.size
        k_top = min(n_top_cells, max(1, int(n * 0.05)))
        top_idx = np.argpartition(-flat, k_top - 1)[:k_top]
        # Hot cells as (row, col) with their heat
        hot_cells = [((int(idx) // ev.grid_col), (int(idx) % ev.grid_col), float(flat[idx])) for idx in top_idx]

        # Find candidate hard macros via net bboxes
        macro_priority: Dict[int, float] = {}
        for net_idx in range(ev.n_nets):
            ymin_cell, xmin_cell = ev._grid_cell(ev._net_xmin[net_idx], ev._net_ymin[net_idx])
            ymax_cell, xmax_cell = ev._grid_cell(ev._net_xmax[net_idx], ev._net_ymax[net_idx])
            net_score = 0.0
            for (r, c, h) in hot_cells:
                if ymin_cell <= r <= ymax_cell and xmin_cell <= c <= xmax_cell:
                    net_score += h
            if net_score <= 0:
                continue
            for owner in ev.net_owner[net_idx]:
                owner = int(owner)
                if owner < ev.n_hard and ev.movable[owner]:
                    macro_priority[owner] = macro_priority.get(owner, 0.0) + net_score
        if not macro_priority:
            break
        macros_sorted = sorted(macro_priority.items(), key=lambda x: -x[1])

        pass_accepted = 0
        for m, _score in macros_sorted:
            if time.time() - t0 > time_budget_s:
                break
            is_hard = m < ev.n_hard
            ox, oy = ev.positions[m]
            rx = sweep_radius_frac * ev.cw
            ry = sweep_radius_frac * ev.ch
            best_local = cur_cost
            best_xy = (ox, oy)
            for dx in np.linspace(-rx, rx, sweep_steps):
                for dy in np.linspace(-ry, ry, sweep_steps):
                    nx = max(ev.half[m, 0], min(ev.cw - ev.half[m, 0], ox + dx))
                    ny = max(ev.half[m, 1], min(ev.ch - ev.half[m, 1], oy + dy))
                    if is_hard and not _slide_legal(ev, m, nx, ny):
                        continue
                    ev.move_macro(m, nx, ny, is_hard=is_hard)
                    c_new = ev.proxy_cost()["proxy_cost"]
                    if c_new < best_local:
                        best_local = c_new
                        best_xy = (nx, ny)
                    # Restore for next candidate
                    ev.move_macro(m, ox, oy, is_hard=is_hard)
            if best_local < cur_cost:
                ev.move_macro(m, best_xy[0], best_xy[1], is_hard=is_hard)
                cur_cost = best_local
                pass_accepted += 1
                accepted_total += 1
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_pos = ev.positions.copy()
        if verbose:
            print(f"  [CONG-ATTACK] pass {pass_idx+1}/{n_passes}  hot cells={len(hot_cells)}  candidates={len(macros_sorted)}  accepted={pass_accepted}  cur={cur_cost:.4f}", flush=True)
        if pass_accepted == 0:
            break  # no improvements found, stop early
    if not np.array_equal(ev.positions, best_pos):
        ev.restore(best_pos)
    return {"proxy_cost": best_cost, "improvement": start_cost - best_cost, "accepted": accepted_total}
