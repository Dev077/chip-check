"""
Phase 3 — LAHC polish (centroid-biased soft + hard swap/slide).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np

from .cong_attack import _smoothed_congestion_grid
from .evaluator import FastEvaluator
from .lk import _slide_legal, _swap_legal


# ────────────────────────────────────────────────────────────────────────────
# Phase 3 — LAHC polish (centroid-biased soft + hard swap/slide)
# ────────────────────────────────────────────────────────────────────────────


def _soft_centroid_target(ev: FastEvaluator, soft_global_idx: int):
    nets = ev._owner_to_nets.get(soft_global_idx, ())
    if not nets:
        return None
    total_w = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for n in nets:
        owners = ev.net_owner[n]
        if owners.size < 2:
            continue
        xs = ev._pin_x(owners, ev.net_offx[n])
        ys = ev._pin_y(owners, ev.net_offy[n])
        own_pos = np.where(owners == soft_global_idx)[0]
        if own_pos.size == 0:
            continue
        i = int(own_pos[0])
        k = owners.size
        px = (xs.sum() - xs[i]) / (k - 1)
        py = (ys.sum() - ys[i]) / (k - 1)
        w = ev._net_weight[n] / (k - 1)
        sum_x += w * px
        sum_y += w * py
        total_w += w
    if total_w <= 0:
        return None
    return float(sum_x / total_w), float(sum_y / total_w)


def lahc_polish(
    ev: FastEvaluator,
    list_len: int = 100,
    time_budget_s: float = 600.0,
    move_radius_frac: float = 0.06,
    soft_move_radius_frac: float = 0.03,
    soft_centroid_prob: float = 0.50,
    swap_prob: float = 0.30,
    soft_prob: float = 0.40,
    decongest_prob: float = 0.0,         # disabled: didn't outperform random LAHC + LK
    n_swap_neighbors: int = 12,
    n_decongest_top_cells: int = 16,
    decongest_refresh_every: int = 100,  # recompute hot-cell list every N iters
    seed: int = 0,
    verbose: bool = True,
):
    rng = np.random.default_rng(seed)
    cur_cost = ev.proxy_cost()["proxy_cost"]
    best_cost = cur_cost
    best_pos = ev.positions.copy()
    history = [cur_cost] * list_len
    t0 = time.time()
    last_log = t0
    it = 0
    # Hot-cell cache for decongest proposals
    hot_cells: List[Tuple[int, int, float]] = []   # (row, col, heat)
    hot_macros: List[int] = []                       # macros contributing to hot cells
    last_hot_refresh = -1
    while time.time() - t0 < time_budget_s:
        if verbose and time.time() - last_log > 20.0:
            print(f"  [LAHC] t={time.time()-t0:.0f}s it={it} cur={cur_cost:.4f} best={best_cost:.4f}", flush=True)
            last_log = time.time()
        # Periodically refresh the hot-cell list and the macros contributing to them
        if it - last_hot_refresh >= decongest_refresh_every:
            cong = _smoothed_congestion_grid(ev)
            flat = cong.ravel()
            n_top = min(n_decongest_top_cells, max(1, int(flat.size * 0.05)))
            top_idx = np.argpartition(-flat, n_top - 1)[:n_top]
            hot_cells = [((int(idx) // ev.grid_col), (int(idx) % ev.grid_col), float(flat[idx])) for idx in top_idx]
            hot_macro_scores: Dict[int, float] = {}
            for net_idx in range(ev.n_nets):
                ymin_c, xmin_c = ev._grid_cell(ev._net_xmin[net_idx], ev._net_ymin[net_idx])
                ymax_c, xmax_c = ev._grid_cell(ev._net_xmax[net_idx], ev._net_ymax[net_idx])
                stress = 0.0
                for (r_h, c_h, h_val) in hot_cells:
                    if ymin_c <= r_h <= ymax_c and xmin_c <= c_h <= xmax_c:
                        stress += h_val
                if stress > 0:
                    for o in ev.net_owner[net_idx]:
                        o = int(o)
                        if o < ev.n_hard and ev.movable[o]:
                            hot_macro_scores[o] = hot_macro_scores.get(o, 0.0) + stress
            if hot_macro_scores:
                # Top-50 hottest hard macros
                hot_macros = [m for m, _ in sorted(hot_macro_scores.items(), key=lambda x: -x[1])[:50]]
            else:
                hot_macros = []
            last_hot_refresh = it
        r = rng.random()
        do_swap = r < swap_prob
        do_soft = (r >= swap_prob) and (r < swap_prob + soft_prob) and (ev.n_soft > 0)
        do_decongest = (
            (r >= swap_prob + soft_prob)
            and (r < swap_prob + soft_prob + decongest_prob)
            and (len(hot_macros) > 0)
        )
        if do_swap:
            i = int(rng.integers(0, ev.n_hard))
            if not ev.movable[i]:
                it += 1
                continue
            d = np.linalg.norm(ev.positions[:ev.n_hard] - ev.positions[i], axis=1)
            d[i] = np.inf
            cands = np.argsort(d)[:n_swap_neighbors]
            j = int(cands[int(rng.integers(0, cands.size))])
            if not ev.movable[j] or not _swap_legal(ev, i, j):
                it += 1
                continue
            ev.swap_macros(i, j)
            cand = ev.proxy_cost()["proxy_cost"]
            idx_h = it % list_len
            if cand < cur_cost or cand < history[idx_h]:
                cur_cost = cand
                history[idx_h] = cand
                if cand < best_cost:
                    best_cost = cand
                    best_pos = ev.positions.copy()
            else:
                ev.swap_macros(i, j)
        elif do_soft:
            i_soft = int(rng.integers(0, ev.n_soft))
            i = ev.n_hard + i_soft
            if not ev.movable[i]:
                it += 1
                continue
            ox, oy = ev.positions[i]
            use_cent = rng.random() < soft_centroid_prob
            if use_cent:
                tgt = _soft_centroid_target(ev, i)
                if tgt is None:
                    it += 1
                    continue
                tx, ty = tgt
                f = float(rng.uniform(0.05, 0.5))
                nx = ox + f * (tx - ox)
                ny = oy + f * (ty - oy)
            else:
                rx = soft_move_radius_frac * ev.cw
                ry = soft_move_radius_frac * ev.ch
                nx = ox + float(rng.uniform(-rx, rx))
                ny = oy + float(rng.uniform(-ry, ry))
            nx = max(ev.half[i, 0], min(ev.cw - ev.half[i, 0], nx))
            ny = max(ev.half[i, 1], min(ev.ch - ev.half[i, 1], ny))
            ev.move_macro(i, nx, ny, is_hard=False)
            cand = ev.proxy_cost()["proxy_cost"]
            idx_h = it % list_len
            if cand < cur_cost or cand < history[idx_h]:
                cur_cost = cand
                history[idx_h] = cand
                if cand < best_cost:
                    best_cost = cand
                    best_pos = ev.positions.copy()
            else:
                ev.move_macro(i, ox, oy, is_hard=False)
        elif do_decongest:
            # Pick a hard macro contributing to a hot cell; propose moving it
            # AWAY from the hot cell centroid (with small random jitter so LAHC
            # can explore around the bias direction).
            i = int(rng.choice(hot_macros))
            if not ev.movable[i]:
                it += 1
                continue
            # Hot centroid (heat-weighted)
            heat_sum = sum(h for _, _, h in hot_cells)
            if heat_sum <= 0:
                it += 1
                continue
            hot_cx = sum((c + 0.5) * ev.gw * h for _, c, h in hot_cells) / heat_sum
            hot_cy = sum((r + 0.5) * ev.gh * h for r, _, h in hot_cells) / heat_sum
            ox, oy = ev.positions[i]
            # Direction AWAY from hot centroid
            dx_dir = ox - hot_cx
            dy_dir = oy - hot_cy
            norm = math.sqrt(dx_dir * dx_dir + dy_dir * dy_dir) + 1e-9
            dx_dir /= norm
            dy_dir /= norm
            step = move_radius_frac * 0.5 * (ev.cw + ev.ch) * float(rng.uniform(0.3, 1.0))
            # Add some lateral noise so we don't always move along the same line
            jitter_x = float(rng.uniform(-0.3, 0.3)) * step
            jitter_y = float(rng.uniform(-0.3, 0.3)) * step
            nx = ox + dx_dir * step + jitter_x
            ny = oy + dy_dir * step + jitter_y
            nx = max(ev.half[i, 0], min(ev.cw - ev.half[i, 0], nx))
            ny = max(ev.half[i, 1], min(ev.ch - ev.half[i, 1], ny))
            if not _slide_legal(ev, i, nx, ny):
                it += 1
                continue
            ev.move_macro(i, nx, ny, is_hard=True)
            cand = ev.proxy_cost()["proxy_cost"]
            idx_h = it % list_len
            if cand < cur_cost or cand < history[idx_h]:
                cur_cost = cand
                history[idx_h] = cand
                if cand < best_cost:
                    best_cost = cand
                    best_pos = ev.positions.copy()
            else:
                ev.move_macro(i, ox, oy, is_hard=True)
        else:
            i = int(rng.integers(0, ev.n_hard))
            if not ev.movable[i]:
                it += 1
                continue
            rx = move_radius_frac * ev.cw
            ry = move_radius_frac * ev.ch
            ox, oy = ev.positions[i]
            nx = max(ev.half[i, 0], min(ev.cw - ev.half[i, 0], ox + float(rng.uniform(-rx, rx))))
            ny = max(ev.half[i, 1], min(ev.ch - ev.half[i, 1], oy + float(rng.uniform(-ry, ry))))
            if not _slide_legal(ev, i, nx, ny):
                it += 1
                continue
            ev.move_macro(i, nx, ny, is_hard=True)
            cand = ev.proxy_cost()["proxy_cost"]
            idx_h = it % list_len
            if cand < cur_cost or cand < history[idx_h]:
                cur_cost = cand
                history[idx_h] = cand
                if cand < best_cost:
                    best_cost = cand
                    best_pos = ev.positions.copy()
            else:
                ev.move_macro(i, ox, oy, is_hard=True)
        it += 1
    if not np.array_equal(ev.positions, best_pos):
        ev.restore(best_pos)
    return {"proxy_cost": best_cost, "iters": it}
