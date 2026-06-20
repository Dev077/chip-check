"""
Phase 2 — Lin-Kernighan k-opt swaps + grid sweeps.

Also exposes the legality helpers (`_swap_legal`, `_slide_legal`,
`_slide_candidates`) used by the subgradient, congestion-attack, LAHC, and
regional phases.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .evaluator import FastEvaluator


# ────────────────────────────────────────────────────────────────────────────
# Phase 2 — Lin-Kernighan k-opt swaps + grid sweeps
# ────────────────────────────────────────────────────────────────────────────


def _macro_priority(ev: FastEvaluator) -> List[int]:
    cong = ev.v_pin_cong + ev.h_pin_cong
    score = np.zeros(ev.n_hard, dtype=np.float64)
    for m in range(ev.n_hard):
        if not ev.movable[m]:
            score[m] = -np.inf
            continue
        nets = ev._owner_to_nets.get(m, ())
        s = 0.0
        for n in nets:
            r1, c1 = ev._grid_cell(ev._net_xmin[n], ev._net_ymin[n])
            r2, c2 = ev._grid_cell(ev._net_xmax[n], ev._net_ymax[n])
            s += float(cong[r1:r2 + 1, c1:c2 + 1].sum())
        score[m] = s
    return list(np.argsort(-score))


def _swap_legal(ev: FastEvaluator, i: int, j: int) -> bool:
    pi = ev.positions[i].copy()
    pj = ev.positions[j].copy()
    hi = ev.half[i]
    hj = ev.half[j]
    if pj[0] - hi[0] < 0 or pj[0] + hi[0] > ev.cw:
        return False
    if pj[1] - hi[1] < 0 or pj[1] + hi[1] > ev.ch:
        return False
    if pi[0] - hj[0] < 0 or pi[0] + hj[0] > ev.cw:
        return False
    if pi[1] - hj[1] < 0 or pi[1] + hj[1] > ev.ch:
        return False
    for k in range(ev.n_hard):
        if k == i or k == j:
            continue
        pk = ev.positions[k]
        sk = ev.sizes[k]
        ox = (ev.sizes[i, 0] + sk[0]) / 2 - abs(pj[0] - pk[0])
        oy = (ev.sizes[i, 1] + sk[1]) / 2 - abs(pj[1] - pk[1])
        if ox > 0 and oy > 0:
            return False
        ox = (ev.sizes[j, 0] + sk[0]) / 2 - abs(pi[0] - pk[0])
        oy = (ev.sizes[j, 1] + sk[1]) / 2 - abs(pi[1] - pk[1])
        if ox > 0 and oy > 0:
            return False
    return True


def _slide_legal(ev: FastEvaluator, i: int, nx: float, ny: float) -> bool:
    half = ev.half[i]
    if nx - half[0] < 0 or nx + half[0] > ev.cw:
        return False
    if ny - half[1] < 0 or ny + half[1] > ev.ch:
        return False
    for k in range(ev.n_hard):
        if k == i:
            continue
        pk = ev.positions[k]
        sk = ev.sizes[k]
        ox = (ev.sizes[i, 0] + sk[0]) / 2 - abs(nx - pk[0])
        oy = (ev.sizes[i, 1] + sk[1]) / 2 - abs(ny - pk[1])
        if ox > 0 and oy > 0:
            return False
    return True


def _slide_candidates(ev: FastEvaluator, i: int, n_steps: int = 5, radius_frac: float = 0.08):
    half = ev.half[i]
    cx, cy = ev.positions[i]
    rx = radius_frac * ev.cw
    ry = radius_frac * ev.ch
    out = []
    for dx in np.linspace(-rx, rx, n_steps):
        for dy in np.linspace(-ry, ry, n_steps):
            if dx == 0 and dy == 0:
                continue
            nx = max(half[0], min(ev.cw - half[0], cx + dx))
            ny = max(half[1], min(ev.ch - half[1], cy + dy))
            out.append((nx, ny))
    return out


def lk_swap_pass(
    ev: FastEvaluator,
    macros: List[int],
    chain_depth: int = 4,
    n_neighbors_per_macro: int = 24,
    log_every: Optional[int] = None,
):
    cur_cost = ev.proxy_cost()["proxy_cost"]
    accepted = 0
    for step, i in enumerate(macros):
        if not ev.movable[i]:
            continue
        d = np.linalg.norm(ev.positions[:ev.n_hard] - ev.positions[i], axis=1)
        d[i] = np.inf
        nbrs = np.argsort(d)[:n_neighbors_per_macro]
        best_gain = 0.0
        best_move = None
        for j in nbrs:
            j = int(j)
            if not ev.movable[j] or not _swap_legal(ev, i, j):
                continue
            ev.swap_macros(i, j)
            c = ev.proxy_cost()["proxy_cost"]
            ev.swap_macros(i, j)  # incremental undo
            gain = cur_cost - c
            if gain > best_gain:
                best_gain = gain
                best_move = ("swap", j, c)
        for (nx, ny) in _slide_candidates(ev, i, n_steps=5, radius_frac=0.08):
            if not _slide_legal(ev, i, nx, ny):
                continue
            ox, oy = ev.positions[i]
            ev.move_macro(i, nx, ny, is_hard=True)
            c = ev.proxy_cost()["proxy_cost"]
            ev.move_macro(i, ox, oy, is_hard=True)
            gain = cur_cost - c
            if gain > best_gain:
                best_gain = gain
                best_move = ("slide", (nx, ny), c)
        if best_move is None:
            continue
        if best_move[0] == "swap":
            _, j, new_c = best_move
            ev.swap_macros(i, j)
            cur_cost = new_c
            accepted += 1
            cur_node = j
            for _ in range(chain_depth - 1):
                d2 = np.linalg.norm(ev.positions[:ev.n_hard] - ev.positions[cur_node], axis=1)
                d2[cur_node] = np.inf
                nb = np.argsort(d2)[:n_neighbors_per_macro]
                lbest_gain = 0.0
                lbest = None
                for k in nb:
                    k = int(k)
                    if k == cur_node or not ev.movable[k] or not _swap_legal(ev, cur_node, k):
                        continue
                    ev.swap_macros(cur_node, k)
                    c = ev.proxy_cost()["proxy_cost"]
                    ev.swap_macros(cur_node, k)
                    g = cur_cost - c
                    if g > lbest_gain:
                        lbest_gain = g
                        lbest = (k, c)
                if lbest is None:
                    break
                k, c = lbest
                ev.swap_macros(cur_node, k)
                cur_cost = c
                accepted += 1
                cur_node = k
        else:
            _, (nx, ny), new_c = best_move
            ev.move_macro(i, nx, ny, is_hard=True)
            cur_cost = new_c
            accepted += 1
        if log_every and (step + 1) % log_every == 0:
            print(f"  [LK] step {step+1}/{len(macros)} cost={cur_cost:.4f} (accepted {accepted})", flush=True)
    return cur_cost, accepted
