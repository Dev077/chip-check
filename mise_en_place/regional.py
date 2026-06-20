"""
Phase 4 — Hierarchical Regional Polish (CPU).

Partition the canvas into R×R regions, visit regions in descending congestion
order, run a focused mini-LAHC restricted to each region's movable macros.
Progressive grids (3 → 5 → 7) refine from coarse clusters to tight pockets.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from .cong_attack import _smoothed_congestion_grid
from .evaluator import FastEvaluator
from .lahc import _soft_centroid_target
from .lk import _slide_legal, _swap_legal


# ────────────────────────────────────────────────────────────────────────────
# Phase 4 — Hierarchical Regional Polish
# ────────────────────────────────────────────────────────────────────────────


def _region_heat(ev: FastEvaluator, cong: np.ndarray, R: int) -> np.ndarray:
    """Aggregate the fine smoothed-congestion grid into an R×R region heat map."""
    rw = ev.cw / R
    rh = ev.ch / R
    cell_ci = np.clip(((np.arange(ev.grid_col) + 0.5) * ev.gw / rw).astype(int), 0, R - 1)
    cell_ri = np.clip(((np.arange(ev.grid_row) + 0.5) * ev.gh / rh).astype(int), 0, R - 1)
    heat = np.zeros((R, R), dtype=np.float64)
    for r in range(ev.grid_row):
        ri = int(cell_ri[r])
        for c in range(ev.grid_col):
            heat[ri, int(cell_ci[c])] += cong[r, c]
    return heat


def _macros_in_region(
    ev: FastEvaluator,
    x0: float, y0: float, x1: float, y1: float,
) -> Tuple[List[int], List[int]]:
    """Return (hard_idx, soft_idx) of movable macros whose centers lie in [x0,x1)×[y0,y1)."""
    hard, soft = [], []
    pos = ev.positions
    for m in range(ev.n_macros):
        if not ev.movable[m]:
            continue
        px, py = pos[m]
        if x0 <= px < x1 and y0 <= py < y1:
            if m < ev.n_hard:
                hard.append(m)
            else:
                soft.append(m)
    return hard, soft


def local_lahc(
    ev: FastEvaluator,
    hard_idx: List[int],
    soft_idx: List[int],
    list_len: int,
    time_budget_s: float,
    move_radius_frac: float,
    soft_move_radius_frac: float,
    soft_centroid_prob: float,
    swap_prob: float,
    soft_prob: float,
    n_swap_neighbors: int,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Mini-LAHC restricted to a given set of macro indices (frozen exterior).

    Move proposals only pick from `hard_idx` ∪ `soft_idx`; swap partners come
    from `hard_idx` only.  Acceptance uses the *global* proxy_cost — guarantees
    no globally regressive move is accepted, so the frozen-exterior contract
    is enforced without explicit per-region cost.
    """
    n_h = len(hard_idx)
    n_s = len(soft_idx)
    if n_h == 0 and n_s == 0:
        return 0, 0
    cur_cost = ev.proxy_cost()["proxy_cost"]
    history = [cur_cost] * list_len
    t0 = time.time()
    it = 0
    accepted = 0
    hard_arr = np.array(hard_idx, dtype=np.int64) if n_h else None
    while time.time() - t0 < time_budget_s:
        r = rng.random()
        do_swap = (r < swap_prob) and (n_h >= 2)
        do_soft = (not do_swap) and (r < swap_prob + soft_prob) and (n_s > 0)
        if do_swap:
            i = int(hard_arr[int(rng.integers(0, n_h))])
            d = np.linalg.norm(ev.positions[hard_arr] - ev.positions[i], axis=1)
            # mask self
            mask = hard_arr != i
            cand_arr = hard_arr[mask]
            d_arr = d[mask]
            if cand_arr.size == 0:
                it += 1
                continue
            k = min(n_swap_neighbors, cand_arr.size)
            nbrs = cand_arr[np.argsort(d_arr)[:k]]
            j = int(nbrs[int(rng.integers(0, nbrs.size))])
            if not _swap_legal(ev, i, j):
                it += 1
                continue
            ev.swap_macros(i, j)
            cand = ev.proxy_cost()["proxy_cost"]
            idx_h = it % list_len
            if cand < cur_cost or cand < history[idx_h]:
                cur_cost = cand
                history[idx_h] = cand
                accepted += 1
            else:
                ev.swap_macros(i, j)
        elif do_soft:
            i = int(soft_idx[int(rng.integers(0, n_s))])
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
                accepted += 1
            else:
                ev.move_macro(i, ox, oy, is_hard=False)
        elif n_h > 0:
            i = int(hard_arr[int(rng.integers(0, n_h))])
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
                accepted += 1
            else:
                ev.move_macro(i, ox, oy, is_hard=True)
        it += 1
    return it, accepted


def regional_polish(
    ev: FastEvaluator,
    region_grids: Tuple[int, ...] = (3, 5, 7),
    time_budget_s: float = 300.0,
    list_len: int = 60,
    move_radius_frac: float = 0.12,
    soft_move_radius_frac: float = 0.06,
    soft_centroid_prob: float = 0.50,
    swap_prob: float = 0.30,
    soft_prob: float = 0.40,
    n_swap_neighbors: int = 8,
    min_macros_per_region: int = 3,
    seed: int = 0,
    verbose: bool = True,
):
    """Hierarchical region-based polish.

    For each grid size R in `region_grids`, partition the canvas into an R×R
    set of regions, then visit regions in descending congestion order.  Each
    region runs a focused mini-LAHC restricted to the movable macros whose
    centers fall in that region — exterior macros stay frozen.  The
    per-region subproblem has many fewer DOFs, so the mini-LAHC can afford a
    larger move radius and reach configurations a global LAHC pass would
    almost never sample.

    Progressive grids (3 → 5 → 7) refine from coarse multi-region clusters
    to tight local pockets.
    """
    rng = np.random.default_rng(seed)
    cur_cost = ev.proxy_cost()["proxy_cost"]
    best_cost = cur_cost
    best_pos = ev.positions.copy()
    t0 = time.time()
    total_iters = 0
    total_accepted = 0
    for sweep_idx, R in enumerate(region_grids):
        if time.time() - t0 >= time_budget_s:
            break
        rw = ev.cw / R
        rh = ev.ch / R
        cong = _smoothed_congestion_grid(ev)
        heat = _region_heat(ev, cong, R)
        order = sorted(
            [(int(ri), int(ci)) for ri in range(R) for ci in range(R)],
            key=lambda x: -heat[x[0], x[1]],
        )
        sweeps_left = len(region_grids) - sweep_idx
        time_left = time_budget_s - (time.time() - t0)
        sweep_budget = time_left / sweeps_left
        per_region_budget = max(2.0, sweep_budget / (R * R))
        regions_done = 0
        for (ri, ci) in order:
            if time.time() - t0 >= time_budget_s:
                break
            x0, x1 = ci * rw, (ci + 1) * rw
            y0, y1 = ri * rh, (ri + 1) * rh
            hard_in, soft_in = _macros_in_region(ev, x0, y0, x1, y1)
            if len(hard_in) + len(soft_in) < min_macros_per_region:
                continue
            it, acc = local_lahc(
                ev, hard_in, soft_in,
                list_len=list_len,
                time_budget_s=per_region_budget,
                move_radius_frac=move_radius_frac,
                soft_move_radius_frac=soft_move_radius_frac,
                soft_centroid_prob=soft_centroid_prob,
                swap_prob=swap_prob,
                soft_prob=soft_prob,
                n_swap_neighbors=n_swap_neighbors,
                rng=rng,
            )
            total_iters += it
            total_accepted += acc
            new_c = ev.proxy_cost()["proxy_cost"]
            if new_c < best_cost:
                best_cost = new_c
                best_pos = ev.positions.copy()
            cur_cost = new_c
            regions_done += 1
        if verbose:
            print(
                f"  [REGIONAL] sweep {sweep_idx+1}/{len(region_grids)} R={R}  "
                f"regions={regions_done}/{R*R}  iters={total_iters}  "
                f"accepted={total_accepted}  cur={cur_cost:.4f}  best={best_cost:.4f}",
                flush=True,
            )
    if not np.array_equal(ev.positions, best_pos):
        ev.restore(best_pos)
    return {"proxy_cost": best_cost, "iters": total_iters, "accepted": total_accepted}
