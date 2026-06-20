"""
Phase α₂ — Stochastic true-cost subgradient.

Adam-style stochastic gradient descent on the EXACT proxy cost via finite
differences on the bit-exact FastEvaluator.
"""

from __future__ import annotations

import time

import numpy as np

from .evaluator import FastEvaluator
from .lk import _slide_legal


# ────────────────────────────────────────────────────────────────────────────
# Phase α₂ — Stochastic true-cost subgradient
# ────────────────────────────────────────────────────────────────────────────


def true_cost_subgradient(
    ev: FastEvaluator,
    time_budget_s: float = 60.0,
    eps_frac: float = 0.01,
    lr_frac: float = 0.005,
    momentum: float = 0.9,
    seed: int = 0,
    verbose: bool = True,
):
    """Adam-style stochastic gradient descent on the EXACT proxy cost.

    Innovation 2.  We compute the per-macro gradient numerically using
    finite differences on the bit-exact FastEvaluator (no surrogate gap):

        ∂proxy/∂x_i ≈ ( proxy(x_i + ε) - proxy(x_i - ε) ) / (2 ε)

    For each randomly selected macro we do 4 incremental evaluations
    (±ε in x, ±ε in y); each is ~0.5 ms.  So one stochastic update per
    macro is ~3 ms.  Updates are clamped to keep hard macros non-overlapping
    and inside the canvas.

    This phase polishes Phase α₁'s output by descending the actual cost,
    not a smoothed surrogate — closing the surrogate-truth gap that
    typically limits analytical GP convergence.
    """
    rng = np.random.default_rng(seed)
    eps_x = eps_frac * ev.cw * 0.1   # small perturbation: 0.1% canvas
    eps_y = eps_frac * ev.ch * 0.1
    lr_x = lr_frac * ev.cw           # step size: 0.5% canvas per update
    lr_y = lr_frac * ev.ch
    cur_cost = ev.proxy_cost()["proxy_cost"]
    best_cost = cur_cost
    best_pos = ev.positions.copy()
    # Per-macro momentum buffers
    mom = np.zeros_like(ev.positions)
    t0 = time.time()
    last_log = t0
    accepted = 0
    n_iters = 0
    while time.time() - t0 < time_budget_s:
        i = int(rng.integers(0, ev.n_macros))
        if not ev.movable[i]:
            n_iters += 1
            continue
        is_hard = i < ev.n_hard
        ox, oy = ev.positions[i]
        # Estimate ∂/∂x via central difference
        ev.move_macro(i, ox + eps_x, oy, is_hard=is_hard)
        c_xp = ev.proxy_cost()["proxy_cost"]
        ev.move_macro(i, ox - eps_x, oy, is_hard=is_hard)
        c_xm = ev.proxy_cost()["proxy_cost"]
        gx = (c_xp - c_xm) / (2.0 * eps_x)
        # ∂/∂y
        ev.move_macro(i, ox, oy + eps_y, is_hard=is_hard)
        c_yp = ev.proxy_cost()["proxy_cost"]
        ev.move_macro(i, ox, oy - eps_y, is_hard=is_hard)
        c_ym = ev.proxy_cost()["proxy_cost"]
        gy = (c_yp - c_ym) / (2.0 * eps_y)
        # Restore current
        ev.move_macro(i, ox, oy, is_hard=is_hard)
        # Momentum update + step
        mom[i, 0] = momentum * mom[i, 0] - lr_x * gx
        mom[i, 1] = momentum * mom[i, 1] - lr_y * gy
        # Clamp step magnitude to a single grid cell at most (avoid huge jumps)
        max_step = 1.5 * ev.gw
        sx = float(np.clip(mom[i, 0], -max_step, max_step))
        sy = float(np.clip(mom[i, 1], -max_step, max_step))
        nx = ox + sx
        ny = oy + sy
        nx = max(ev.half[i, 0], min(ev.cw - ev.half[i, 0], nx))
        ny = max(ev.half[i, 1], min(ev.ch - ev.half[i, 1], ny))
        # For hard macros, only commit if no overlap with neighbors
        if is_hard and not _slide_legal(ev, i, nx, ny):
            n_iters += 1
            continue
        ev.move_macro(i, nx, ny, is_hard=is_hard)
        new_cost = ev.proxy_cost()["proxy_cost"]
        if new_cost < cur_cost:
            cur_cost = new_cost
            accepted += 1
            if new_cost < best_cost:
                best_cost = new_cost
                best_pos = ev.positions.copy()
        else:
            # Revert (subgradient sometimes overshoots; treat as a hill-climbing oracle)
            ev.move_macro(i, ox, oy, is_hard=is_hard)
            mom[i] *= 0.5  # damp the momentum after rejection
        n_iters += 1
        if verbose and time.time() - last_log > 20.0:
            print(f"  [α₂] t={time.time()-t0:.0f}s it={n_iters} accepted={accepted} cur={cur_cost:.4f} best={best_cost:.4f}", flush=True)
            last_log = time.time()
    if not np.array_equal(ev.positions, best_pos):
        ev.restore(best_pos)
    return {"proxy_cost": best_cost, "iters": n_iters, "accepted": accepted}
