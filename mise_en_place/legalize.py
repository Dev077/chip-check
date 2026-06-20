"""
Phase 0 — Legalization (LKPlacer variant).

Greedy push-apart legalizer with spiral fallback for stragglers.
"""

from __future__ import annotations

import math

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Phase 0 — Legalization
# ────────────────────────────────────────────────────────────────────────────


def _overlap_pair(p1, s1, p2, s2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    ox = (s1[0] + s2[0]) / 2 - dx
    oy = (s1[1] + s2[1]) / 2 - dy
    if ox > 0 and oy > 0:
        return ox, oy
    return 0.0, 0.0


def _has_overlap(i, pos, sizes):
    n = pos.shape[0]
    for j in range(n):
        if i == j:
            continue
        ox, oy = _overlap_pair(pos[i], sizes[i], pos[j], sizes[j])
        if ox > 0 and oy > 0:
            return True
    return False


def _spiral_search(i, pos, sizes, movable, cw, ch, gap):
    base = pos[i].copy()
    half = sizes[i] / 2
    step = min(cw, ch) * 0.01
    for r in range(1, 200):
        for ang in range(0, 360, 15):
            t = math.radians(ang)
            cand = base + np.array([math.cos(t), math.sin(t)]) * (step * r)
            cand[0] = max(half[0], min(cw - half[0], cand[0]))
            cand[1] = max(half[1], min(ch - half[1], cand[1]))
            ok = True
            for j in range(pos.shape[0]):
                if i == j:
                    continue
                ox, oy = _overlap_pair(cand, sizes[i], pos[j], sizes[j])
                if ox > 0 and oy > 0:
                    ok = False
                    break
            if ok:
                return cand
    return base


def _legalize(
    positions: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.02,
    max_passes: int = 80,
) -> np.ndarray:
    """Greedy push-apart legalizer with spiral fallback for stragglers."""
    n = positions.shape[0]
    pos = positions.copy()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    for _ in range(max_passes):
        any_ov = False
        order = np.argsort(pos[:, 0])
        for ii in range(n):
            i = order[ii]
            for jj in range(ii + 1, n):
                j = order[jj]
                if pos[j, 0] - pos[i, 0] > (sizes[i, 0] + sizes[j, 0]) / 2 + gap:
                    break
                ox, oy = _overlap_pair(pos[i], sizes[i], pos[j], sizes[j])
                if ox <= 0 or oy <= 0:
                    continue
                any_ov = True
                if ox < oy:
                    push = ox + gap
                    sign = 1.0 if pos[j, 0] >= pos[i, 0] else -1.0
                    if movable[i] and movable[j]:
                        pos[i, 0] -= sign * push * 0.5
                        pos[j, 0] += sign * push * 0.5
                    elif movable[j]:
                        pos[j, 0] += sign * push
                    elif movable[i]:
                        pos[i, 0] -= sign * push
                else:
                    push = oy + gap
                    sign = 1.0 if pos[j, 1] >= pos[i, 1] else -1.0
                    if movable[i] and movable[j]:
                        pos[i, 1] -= sign * push * 0.5
                        pos[j, 1] += sign * push * 0.5
                    elif movable[j]:
                        pos[j, 1] += sign * push
                    elif movable[i]:
                        pos[i, 1] -= sign * push
        np.clip(pos[:, 0], half_w, canvas_w - half_w, out=pos[:, 0])
        np.clip(pos[:, 1], half_h, canvas_h - half_h, out=pos[:, 1])
        if not any_ov:
            break
    for i in range(n):
        if not movable[i]:
            continue
        if _has_overlap(i, pos, sizes):
            pos[i] = _spiral_search(i, pos, sizes, movable, canvas_w, canvas_h, gap)
    return pos
