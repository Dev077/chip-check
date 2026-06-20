"""
Phase 1 — FastEvaluator (bit-exact NumPy mirror of PlacementCost).

Implements get_cost / get_density_cost / get_congestion_cost with incremental
update support.  Validated bit-exact against PlacementCost on ibm01 (and others);
a single move_macro() call is ~2 ms (vs ~4000 ms for the oracle).
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from macro_place.benchmark import Benchmark


# ────────────────────────────────────────────────────────────────────────────
# Phase 1 — FastEvaluator (bit-exact mirror of PlacementCost)
# ────────────────────────────────────────────────────────────────────────────


class FastEvaluator:
    """NumPy reimplementation of PlacementCost.get_cost / get_density_cost /
    get_congestion_cost with incremental update support.

    Validated bit-exact against PlacementCost on ibm01 (and others); a single
    move_macro() call is ~2 ms (vs ~4000 ms for the oracle).
    """

    def __init__(self, benchmark: Benchmark, plc):
        self.benchmark = benchmark
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.grid_col = int(benchmark.grid_cols)
        self.grid_row = int(benchmark.grid_rows)
        self.gw = self.cw / self.grid_col
        self.gh = self.ch / self.grid_row
        self.grid_area = self.gw * self.gh
        self.h_per_um = float(benchmark.hroutes_per_micron)
        self.v_per_um = float(benchmark.vroutes_per_micron)
        self.grid_v_routes = self.gw * self.v_per_um
        self.grid_h_routes = self.gh * self.h_per_um
        # Routing allocation + smoothing range come from PlacementCost.
        self.h_alloc = 0.0
        self.v_alloc = 0.0
        self.smooth_range = 2
        if plc is not None:
            try:
                self.h_alloc, self.v_alloc = plc.get_macro_routing_allocation()
            except Exception:
                self.h_alloc = getattr(plc, "hrouting_alloc", 0.0)
                self.v_alloc = getattr(plc, "vrouting_alloc", 0.0)
            try:
                self.smooth_range = int(plc.get_congestion_smooth_range())
            except Exception:
                self.smooth_range = int(getattr(plc, "smooth_range", 2))
        self.n_hard = benchmark.num_hard_macros
        self.n_macros = benchmark.num_macros
        self.n_soft = self.n_macros - self.n_hard
        self.n_nets = int(benchmark.num_nets)
        self.n_ports = int(benchmark.port_positions.shape[0])
        # WL normalization uses plc.net_cnt (counts every driver pin, not just nets with sinks)
        self.wl_norm_n_nets = int(getattr(plc, "net_cnt", self.n_nets)) if plc is not None else self.n_nets
        if self.wl_norm_n_nets <= 0:
            self.wl_norm_n_nets = max(self.n_nets, 1)
        # State arrays
        self.positions = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)
        self.sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
        self.half = self.sizes / 2.0
        self.port_pos = benchmark.port_positions.detach().cpu().numpy().astype(np.float64) if self.n_ports else np.zeros((0, 2))
        self.movable = benchmark.get_movable_mask().detach().cpu().numpy().astype(bool)
        # Per-net tables
        self._build_net_pin_tables(benchmark)
        self._net_xmin = np.zeros(self.n_nets, dtype=np.float64)
        self._net_ymin = np.zeros(self.n_nets, dtype=np.float64)
        self._net_xmax = np.zeros(self.n_nets, dtype=np.float64)
        self._net_ymax = np.zeros(self.n_nets, dtype=np.float64)
        self._net_weight = np.ones(self.n_nets, dtype=np.float64)
        if plc is not None:
            self._fetch_net_weights(plc)
        self._owner_to_nets: Dict[int, List[int]] = {}
        for n in range(self.n_nets):
            for o in self.net_owner[n]:
                self._owner_to_nets.setdefault(int(o), []).append(n)
        # Grids
        self.density_grid = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.h_pin_cong = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.v_pin_cong = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.h_macro_cong = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.v_macro_cong = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self._init_caches()

    def _build_net_pin_tables(self, benchmark: Benchmark):
        pin_offsets = benchmark.macro_pin_offsets
        npn = benchmark.net_pin_nodes
        self.net_owner: List[np.ndarray] = []
        self.net_offx: List[np.ndarray] = []
        self.net_offy: List[np.ndarray] = []
        if not npn:
            for n in range(self.n_nets):
                nodes = benchmark.net_nodes[n].cpu().numpy().astype(np.int64) if benchmark.net_nodes else np.zeros(0, dtype=np.int64)
                self.net_owner.append(nodes)
                self.net_offx.append(np.zeros(nodes.shape[0]))
                self.net_offy.append(np.zeros(nodes.shape[0]))
            return
        for n in range(self.n_nets):
            pn = npn[n].cpu().numpy().astype(np.int64)
            if pn.size == 0:
                self.net_owner.append(np.zeros(0, dtype=np.int64))
                self.net_offx.append(np.zeros(0))
                self.net_offy.append(np.zeros(0))
                continue
            owners = pn[:, 0]
            slots = pn[:, 1]
            offx = np.zeros(owners.shape[0])
            offy = np.zeros(owners.shape[0])
            for k in range(owners.shape[0]):
                o, s = int(owners[k]), int(slots[k])
                if o < self.n_hard and pin_offsets and o < len(pin_offsets):
                    po = pin_offsets[o]
                    if po is not None and po.shape[0] > s:
                        offx[k] = float(po[s, 0])
                        offy[k] = float(po[s, 1])
            self.net_owner.append(owners)
            self.net_offx.append(offx)
            self.net_offy.append(offy)

    def _fetch_net_weights(self, plc):
        try:
            driver_names = list(plc.nets.keys())
            for n in range(min(self.n_nets, len(driver_names))):
                pi = plc.mod_name_to_indices[driver_names[n]]
                self._net_weight[n] = float(plc.modules_w_pins[pi].get_weight())
        except Exception:
            pass

    def _pin_x(self, owners, offx):
        out = np.empty(owners.shape[0], dtype=np.float64)
        m = owners < self.n_macros
        out[m] = self.positions[owners[m], 0] + offx[m]
        if (~m).any():
            p_idx = owners[~m] - self.n_macros
            out[~m] = self.port_pos[p_idx, 0] + offx[~m]
        return out

    def _pin_y(self, owners, offy):
        out = np.empty(owners.shape[0], dtype=np.float64)
        m = owners < self.n_macros
        out[m] = self.positions[owners[m], 1] + offy[m]
        if (~m).any():
            p_idx = owners[~m] - self.n_macros
            out[~m] = self.port_pos[p_idx, 1] + offy[~m]
        return out

    def _net_bbox(self, n):
        owners = self.net_owner[n]
        if owners.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        xs = self._pin_x(owners, self.net_offx[n])
        ys = self._pin_y(owners, self.net_offy[n])
        return xs.min(), ys.min(), xs.max(), ys.max()

    def _grid_cell(self, x, y):
        c = int(math.floor(x / self.gw))
        r = int(math.floor(y / self.gh))
        return max(0, min(self.grid_row - 1, r)), max(0, min(self.grid_col - 1, c))

    def _add_macro_density(self, macro_idx, sign=+1):
        x, y = self.positions[macro_idx]
        w, h = self.sizes[macro_idx]
        x_min, x_max = x - w / 2, x + w / 2
        y_min, y_max = y - h / 2, y + h / 2
        ur_r, ur_c = self._grid_cell(x_max, y_max)
        bl_r, bl_c = self._grid_cell(x_min, y_min)
        for r in range(bl_r, ur_r + 1):
            gy0 = r * self.gh
            gy1 = (r + 1) * self.gh
            dy = min(y_max, gy1) - max(y_min, gy0)
            if dy <= 0:
                continue
            for c in range(bl_c, ur_c + 1):
                gx0 = c * self.gw
                gx1 = (c + 1) * self.gw
                dx = min(x_max, gx1) - max(x_min, gx0)
                if dx <= 0:
                    continue
                self.density_grid[r, c] += sign * dx * dy

    def _add_macro_route(self, macro_idx, sign=+1):
        x, y = self.positions[macro_idx]
        w, h = self.sizes[macro_idx]
        x_min, x_max = x - w / 2, x + w / 2
        y_min, y_max = y - h / 2, y + h / 2
        ur_r, ur_c = self._grid_cell(x_max, y_max)
        bl_r, bl_c = self._grid_cell(x_min, y_min)
        partial_v = False
        partial_h = False
        eps = 1e-5
        for r in range(bl_r, ur_r + 1):
            gy0 = r * self.gh
            gy1 = (r + 1) * self.gh
            dy = min(y_max, gy1) - max(y_min, gy0)
            if dy <= 0:
                continue
            for c in range(bl_c, ur_c + 1):
                gx0 = c * self.gw
                gx1 = (c + 1) * self.gw
                dx = min(x_max, gx1) - max(x_min, gx0)
                if dx <= 0:
                    continue
                self.v_macro_cong[r, c] += sign * dx * self.v_alloc
                self.h_macro_cong[r, c] += sign * dy * self.h_alloc
                if ur_r != bl_r and (r == bl_r or r == ur_r) and abs(dy - self.gh) > eps:
                    partial_v = True
                if ur_c != bl_c and (c == bl_c or c == ur_c) and abs(dx - self.gw) > eps:
                    partial_h = True
        if partial_v:
            r = ur_r
            for c in range(bl_c, ur_c + 1):
                gx0, gx1 = c * self.gw, (c + 1) * self.gw
                dx = min(x_max, gx1) - max(x_min, gx0)
                if dx > 0:
                    self.v_macro_cong[r, c] -= sign * dx * self.v_alloc
        if partial_h:
            c = ur_c
            for r in range(bl_r, ur_r + 1):
                gy0, gy1 = r * self.gh, (r + 1) * self.gh
                dy = min(y_max, gy1) - max(y_min, gy0)
                if dy > 0:
                    self.h_macro_cong[r, c] -= sign * dy * self.h_alloc

    def _route_pin_cong(self, net_idx, sign=+1):
        owners = self.net_owner[net_idx]
        if owners.size == 0:
            return
        xs = self._pin_x(owners, self.net_offx[net_idx])
        ys = self._pin_y(owners, self.net_offy[net_idx])
        cells = []
        cells_set = set()
        for i in range(owners.shape[0]):
            r, c = self._grid_cell(xs[i], ys[i])
            cells.append((r, c))
            cells_set.add((r, c))
        if len(cells_set) <= 1:
            return
        src = cells[0]
        w = self._net_weight[net_idx]
        if len(cells_set) == 2:
            self._two_pin(src, list(cells_set), w, sign)
        elif len(cells_set) == 3:
            self._three_pin(list(cells_set), w, sign)
        else:
            for n in cells_set:
                if n == src:
                    continue
                self._two_pin(src, [src, n], w, sign)

    def _two_pin(self, src, two, w, sign):
        sink = two[1] if two[0] == src else two[0]
        r_min, r_max = min(src[0], sink[0]), max(src[0], sink[0])
        c_min, c_max = min(src[1], sink[1]), max(src[1], sink[1])
        if c_max > c_min:
            self.h_pin_cong[src[0], c_min:c_max] += sign * w
        if r_max > r_min:
            self.v_pin_cong[r_min:r_max, sink[1]] += sign * w

    def _three_pin(self, cells, w, sign):
        cs = sorted(cells, key=lambda x: (x[1], x[0]))
        (y1, x1), (y2, x2), (y3, x3) = cs
        if x1 < x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            self._l(cs, w, sign)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            if x2 > x1:
                self.h_pin_cong[y1, x1:x2] += sign * w
            r_lo, r_hi = y1, max(y2, y3)
            if r_hi > r_lo:
                self.v_pin_cong[r_lo:r_hi, x2] += sign * w
        elif y2 == y3:
            if x2 > x1:
                self.h_pin_cong[y1, x1:x2] += sign * w
            if x3 > x2:
                self.h_pin_cong[y2, x2:x3] += sign * w
            r_lo, r_hi = min(y1, y2), max(y1, y2)
            if r_hi > r_lo:
                self.v_pin_cong[r_lo:r_hi, x2] += sign * w
        else:
            self._t(cs, w, sign)

    def _l(self, cs, w, sign):
        (y1, x1), (y2, x2), (y3, x3) = cs
        if x2 > x1:
            self.h_pin_cong[y1, x1:x2] += sign * w
        if x3 > x2:
            self.h_pin_cong[y2, x2:x3] += sign * w
        r_lo, r_hi = min(y1, y2), max(y1, y2)
        if r_hi > r_lo:
            self.v_pin_cong[r_lo:r_hi, x2] += sign * w
        r_lo, r_hi = min(y2, y3), max(y2, y3)
        if r_hi > r_lo:
            self.v_pin_cong[r_lo:r_hi, x3] += sign * w

    def _t(self, cs, w, sign):
        cs2 = sorted(cs)
        (y1, x1), (y2, x2), (y3, x3) = cs2
        xmin = min(x1, x2, x3)
        xmax = max(x1, x2, x3)
        if xmax > xmin:
            self.h_pin_cong[y2, xmin:xmax] += sign * w
        r_lo, r_hi = min(y1, y2), max(y1, y2)
        if r_hi > r_lo:
            self.v_pin_cong[r_lo:r_hi, x1] += sign * w
        r_lo, r_hi = min(y2, y3), max(y2, y3)
        if r_hi > r_lo:
            self.v_pin_cong[r_lo:r_hi, x3] += sign * w

    def _init_caches(self):
        self.density_grid[...] = 0
        self.h_pin_cong[...] = 0
        self.v_pin_cong[...] = 0
        self.h_macro_cong[...] = 0
        self.v_macro_cong[...] = 0
        for m in range(self.n_macros):
            self._add_macro_density(m, +1)
        for m in range(self.n_hard):
            self._add_macro_route(m, +1)
        for n in range(self.n_nets):
            x0, y0, x1, y1 = self._net_bbox(n)
            self._net_xmin[n] = x0
            self._net_ymin[n] = y0
            self._net_xmax[n] = x1
            self._net_ymax[n] = y1
            self._route_pin_cong(n, +1)

    def _density_cost(self):
        gc = (self.density_grid / self.grid_area).ravel()
        nz = gc[gc > 0]
        if nz.size == 0:
            return 0.0
        N = gc.size
        if N < 10:
            return 0.5 * float(nz.mean())
        cnt = math.floor(N * 0.1)
        if cnt == 0:
            return 0.5 * float(nz.max())
        sd = np.sort(nz)[::-1]
        take = min(cnt, sd.size)
        return 0.5 * float(sd[:take].sum() / cnt)

    def _smooth(self, grid, axis):
        sr = self.smooth_range
        R, C = grid.shape
        if axis == 0:
            cols = np.arange(C)
            lp = np.maximum(0, cols - sr)
            rp = np.minimum(C - 1, cols + sr)
            cnt = (rp - lp + 1).astype(np.float64)
            scaled = grid / cnt[np.newaxis, :]
            pad = np.pad(scaled, ((0, 0), (sr, sr)), mode="constant")
            cs = np.cumsum(pad, axis=1)
            cs0 = cs[:, 2 * sr:]
            cs1 = np.concatenate([np.zeros((R, 1)), cs[:, :C - 1 + 2 * sr]], axis=1)[:, :C]
            return cs0[:, :C] - cs1
        else:
            rows = np.arange(R)
            lp = np.maximum(0, rows - sr)
            up = np.minimum(R - 1, rows + sr)
            cnt = (up - lp + 1).astype(np.float64)
            scaled = grid / cnt[:, np.newaxis]
            pad = np.pad(scaled, ((sr, sr), (0, 0)), mode="constant")
            cs = np.cumsum(pad, axis=0)
            cs0 = cs[2 * sr:, :]
            cs1 = np.concatenate([np.zeros((1, C)), cs[:R - 1 + 2 * sr, :]], axis=0)[:R, :]
            return cs0[:R, :] - cs1

    def _congestion_cost(self):
        v = self.v_pin_cong / self.grid_v_routes
        h = self.h_pin_cong / self.grid_h_routes
        vm = self.v_macro_cong / self.grid_v_routes
        hm = self.h_macro_cong / self.grid_h_routes
        v_s = self._smooth(v, axis=0)
        h_s = self._smooth(h, axis=1)
        combined = np.concatenate([(v_s + vm).ravel(), (h_s + hm).ravel()])
        xs = np.sort(combined)[::-1]
        cnt = math.floor(xs.size * 0.05)
        if cnt == 0:
            return float(xs.max()) if xs.size else 0.0
        return float(xs[:cnt].mean())

    def _wirelength_cost(self):
        hpwl = (self._net_xmax - self._net_xmin) + (self._net_ymax - self._net_ymin)
        return float(np.sum(hpwl * self._net_weight)) / ((self.cw + self.ch) * self.wl_norm_n_nets)

    def proxy_cost(self):
        wl = self._wirelength_cost()
        d = self._density_cost()
        c = self._congestion_cost()
        return {
            "proxy_cost": wl + 0.5 * d + 0.5 * c,
            "wirelength_cost": wl,
            "density_cost": d,
            "congestion_cost": c,
        }

    def move_macro(self, macro_idx, new_x, new_y, is_hard=True):
        if is_hard:
            self._add_macro_route(macro_idx, -1)
        self._add_macro_density(macro_idx, -1)
        nets = self._owner_to_nets.get(macro_idx, ())
        for n in nets:
            self._route_pin_cong(n, -1)
        self.positions[macro_idx, 0] = new_x
        self.positions[macro_idx, 1] = new_y
        self._add_macro_density(macro_idx, +1)
        if is_hard:
            self._add_macro_route(macro_idx, +1)
        for n in nets:
            x0, y0, x1, y1 = self._net_bbox(n)
            self._net_xmin[n] = x0
            self._net_ymin[n] = y0
            self._net_xmax[n] = x1
            self._net_ymax[n] = y1
            self._route_pin_cong(n, +1)

    def swap_macros(self, i, j):
        xi, yi = self.positions[i]
        xj, yj = self.positions[j]
        self.move_macro(i, xj, yj, is_hard=(i < self.n_hard))
        self.move_macro(j, xi, yi, is_hard=(j < self.n_hard))

    def snapshot(self):
        return self.positions.copy()

    def restore(self, positions):
        if np.array_equal(positions, self.positions):
            return
        self.positions[:] = positions
        self._init_caches()
