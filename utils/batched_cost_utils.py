"""
BatchedCostEstimator: GPU-parallel cost estimator for R SA chains.

Tracks R placements simultaneously and exposes a propose/apply API so that
each SA iteration can score moves on all R chains in a single batched
forward pass.

API in one paragraph
--------------------
Construct with `(benchmark, num_chains=R, device='cuda')`. All state has a
leading batch dim R. Each SA iteration:

    deltas = est.propose_moves(chain_idx, macro_idx, new_pos)
    # deltas is a dict with 'proxy_cost', 'overlap_count_after', etc.,
    # each shape [R]. The estimator is NOT mutated yet.
    accept = metropolis(deltas, temperatures)   # [R] bool
    est.apply_moves(chain_idx, macro_idx, new_pos, accept_mask=accept)
    # Only accepted chains have state changed; others untouched.

For parallel tempering swaps:

    est.swap_chains(i, j)   # exchange all R-indexed state between i and j

Math fidelity vs. serial CostEstimator
--------------------------------------
Algorithm-identical, not bit-exact. Tests in test_batched_cost_utils.py
verify that for the same sequence of moves, the batched version matches
the serial version to ~1e-4 absolute on proxy cost (float32 batched-op
rounding). That tolerance is comfortably below the 1e-3 cost differences
that SA's Metropolis criterion can resolve.

Architecture notes
------------------
- Density grid: [R, rows, cols].
- Congestion grid: [R, rows, cols].
- Per-net HPWL: [R, num_nets]. Each chain has its own net wirelengths.
- Overlap counts: [R] integers.

Grid updates use a padded-rectangle scheme: every macro's footprint fits
in a max_k × max_k bounding box of grid cells (max_k computed from the
largest macro size). We compute per-chain inter_w[R, max_k] and
inter_h[R, max_k] vectors, outer product to [R, max_k, max_k], then
scatter_add into the grid at per-chain row/col offsets. Padding waste is
~2x worst case but kernel launch overhead drops R-fold, so the net
trade is firmly in favour of batching for R≥8.

HPWL uses a padded [num_macros, max_pins_per_macro_in_nets] structure
to vectorize the "for each net touching this macro, recompute" loop.
"""

from typing import Dict, List, Optional, Tuple

import torch

from macro_place.benchmark import Benchmark


def _make_macro_to_nets_padded(
    benchmark: Benchmark, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a padded [num_macros, max_nets] tensor where row i lists the
    net indices that macro i participates in, padded with -1.

    Returns (padded_nets, nets_per_macro).
    """
    macro_to_nets: List[List[int]] = [[] for _ in range(benchmark.num_macros)]
    for net_idx, net_pins in enumerate(benchmark.net_pin_nodes):
        for owner_idx in torch.unique(net_pins[:, 0]):
            o = int(owner_idx.item())
            if o < benchmark.num_macros:
                macro_to_nets[o].append(net_idx)

    nets_per_macro = torch.tensor(
        [len(L) for L in macro_to_nets], dtype=torch.long, device=device
    )
    max_nets = int(nets_per_macro.max().item()) if benchmark.num_macros else 0
    if max_nets == 0:
        padded = torch.empty(benchmark.num_macros, 0, dtype=torch.long, device=device)
    else:
        padded = torch.full(
            (benchmark.num_macros, max_nets), -1, dtype=torch.long, device=device
        )
        for i, L in enumerate(macro_to_nets):
            if L:
                padded[i, : len(L)] = torch.tensor(L, dtype=torch.long, device=device)

    return padded, nets_per_macro


def _make_net_pin_table(
    benchmark: Benchmark, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded [num_nets, max_pins] tensors describing each net's pins.

    Returns:
      owner_idx_padded:  [num_nets, max_pins] long, -1 padding
      pin_slot_padded:   [num_nets, max_pins] long, -1 padding
      pin_mask:          [num_nets, max_pins] bool, True where real pin
      pin_offsets_static: [num_nets, max_pins, 2] float — the per-pin
        offset within its owner macro. Soft macros and ports have zero
        offset. Hard macros' pin offsets are looked up from
        benchmark.macro_pin_offsets.
    """
    num_nets = benchmark.num_nets
    pins_per_net = [len(p) for p in benchmark.net_pin_nodes]
    max_pins = max(pins_per_net) if pins_per_net else 0

    owner_idx_padded = torch.full(
        (num_nets, max_pins), -1, dtype=torch.long, device=device
    )
    pin_slot_padded = torch.full(
        (num_nets, max_pins), -1, dtype=torch.long, device=device
    )
    pin_mask = torch.zeros((num_nets, max_pins), dtype=torch.bool, device=device)
    pin_offsets_static = torch.zeros(
        (num_nets, max_pins, 2), dtype=torch.float32, device=device
    )

    num_hard = benchmark.num_hard_macros
    for n, pins in enumerate(benchmark.net_pin_nodes):
        k = pins.shape[0]
        owners = pins[:, 0].to(device=device, dtype=torch.long)
        slots = pins[:, 1].to(device=device, dtype=torch.long)
        owner_idx_padded[n, :k] = owners
        pin_slot_padded[n, :k] = slots
        pin_mask[n, :k] = True
        # Hard macros have explicit per-pin offsets; everything else is 0.
        for i in range(k):
            o = int(owners[i].item())
            p = int(slots[i].item())
            if o < num_hard:
                pin_offsets_static[n, i] = benchmark.macro_pin_offsets[o][p].to(device)

    return owner_idx_padded, pin_slot_padded, pin_mask, pin_offsets_static


def _macro_grid_rect(
    pos: torch.Tensor,            # [R, 2]
    size: torch.Tensor,           # [2]
    cell_w: float, cell_h: float,
    rows: int, cols: int,
):
    """
    Given a macro's position per chain and its fixed size, return
    everything needed to scatter its footprint into the density grid.

    Returns:
      r_start, r_end : [R] long, clamped to [0, rows-1]
      c_start, c_end : [R] long, clamped to [0, cols-1]
      m_lx, m_ux, m_by, m_ty : [R] float — the macro's bounding box.
    """
    w = size[0]
    h = size[1]
    x = pos[:, 0]
    y = pos[:, 1]
    m_lx = x - w / 2
    m_ux = x + w / 2
    m_by = y - h / 2
    m_ty = y + h / 2

    c_start = torch.clamp((m_lx / cell_w).floor().long(), 0, cols - 1)
    c_end = torch.clamp((m_ux / cell_w).floor().long(), 0, cols - 1)
    r_start = torch.clamp((m_by / cell_h).floor().long(), 0, rows - 1)
    r_end = torch.clamp((m_ty / cell_h).floor().long(), 0, rows - 1)

    return r_start, r_end, c_start, c_end, m_lx, m_ux, m_by, m_ty


def _density_contribution(
    pos: torch.Tensor,            # [R, 2]
    size: torch.Tensor,           # [2]
    cell_w: float, cell_h: float,
    rows: int, cols: int,
    max_k_r: int, max_k_c: int,
):
    """
    Compute the padded contribution of one macro (across R chains) to the
    density grid.

    Returns:
      contrib : [R, max_k_r, max_k_c] — the per-cell overlap area within
        the macro's footprint, padded with zeros outside.
      r_start : [R] long — starting row of each chain's rectangle.
      c_start : [R] long — starting column of each chain's rectangle.

    The caller can scatter `contrib` into the grid at offsets
    `(r_start, c_start)` to add (or subtract) this macro's footprint.
    """
    r_start, r_end, c_start, c_end, m_lx, m_ux, m_by, m_ty = _macro_grid_rect(
        pos, size, cell_w, cell_h, rows, cols
    )
    R = pos.shape[0]
    device = pos.device

    # For each chain, we want inter_w[r, k] = overlap of column (c_start[r] + k)
    # with the macro's x-extent. Vectorize by building a [max_k_c] column-offset
    # vector and a [R, max_k_c] cell-left tensor.
    k_c = torch.arange(max_k_c, device=device, dtype=torch.long)  # [max_k_c]
    col_idx = c_start.unsqueeze(1) + k_c.unsqueeze(0)              # [R, max_k_c]
    cell_left = col_idx.float() * cell_w
    cell_right = cell_left + cell_w
    inter_w = torch.clamp(
        torch.minimum(cell_right, m_ux.unsqueeze(1)) -
        torch.maximum(cell_left, m_lx.unsqueeze(1)),
        min=0.0,
    )
    # Zero out columns beyond the macro's actual rectangle (k > c_end - c_start).
    col_valid = col_idx <= c_end.unsqueeze(1)
    inter_w = inter_w * col_valid

    k_r = torch.arange(max_k_r, device=device, dtype=torch.long)
    row_idx = r_start.unsqueeze(1) + k_r.unsqueeze(0)
    cell_bot = row_idx.float() * cell_h
    cell_top = cell_bot + cell_h
    inter_h = torch.clamp(
        torch.minimum(cell_top, m_ty.unsqueeze(1)) -
        torch.maximum(cell_bot, m_by.unsqueeze(1)),
        min=0.0,
    )
    row_valid = row_idx <= r_end.unsqueeze(1)
    inter_h = inter_h * row_valid

    # Outer product per chain: [R, max_k_r, max_k_c]
    contrib = inter_h.unsqueeze(2) * inter_w.unsqueeze(1)
    return contrib, r_start, c_start


def _scatter_rect_into_grid(
    grid: torch.Tensor,         # [R, rows, cols]
    contrib: torch.Tensor,      # [R, max_k_r, max_k_c]
    r_start: torch.Tensor,      # [R] long
    c_start: torch.Tensor,      # [R] long
    sign: float,
):
    """
    Add `sign * contrib[r]` into `grid[r, r_start[r]:r_start[r]+kr,
    c_start[r]:c_start[r]+kc]` for each chain r.

    Implementation: build absolute row/col index tensors and use
    `index_put_` with `accumulate=True`. This avoids the variable-shape
    slicing that would otherwise break batching.
    """
    R, rows, cols = grid.shape
    max_k_r, max_k_c = contrib.shape[1], contrib.shape[2]
    device = grid.device

    chain_idx = torch.arange(R, device=device).view(R, 1, 1).expand(R, max_k_r, max_k_c)
    k_r = torch.arange(max_k_r, device=device).view(1, max_k_r, 1)
    k_c = torch.arange(max_k_c, device=device).view(1, 1, max_k_c)

    r_abs = (r_start.view(R, 1, 1) + k_r).expand(R, max_k_r, max_k_c)
    c_abs = (c_start.view(R, 1, 1) + k_c).expand(R, max_k_r, max_k_c)

    # Clamp out-of-bounds cells (shouldn't happen, but defensive); we
    # also zero those entries in contrib to avoid corrupting the grid
    # at clamped positions.
    in_bounds = (r_abs >= 0) & (r_abs < rows) & (c_abs >= 0) & (c_abs < cols)
    r_clamped = r_abs.clamp(0, rows - 1)
    c_clamped = c_abs.clamp(0, cols - 1)

    safe_contrib = contrib * in_bounds
    grid.index_put_(
        (chain_idx, r_clamped, c_clamped),
        sign * safe_contrib,
        accumulate=True,
    )


# ─── BatchedCostEstimator ──────────────────────────────────────────────────


class BatchedCostEstimator:
    """
    R-chain batched cost estimator. See module docstring.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        num_chains: int,
        initial_placement: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ):
        if num_chains < 1:
            raise ValueError("num_chains must be >= 1")
        self.benchmark = benchmark
        self.R = int(num_chains)
        self.weights = weights if weights else {
            "wirelength": 1.0, "density": 0.5, "congestion": 0.5,
        }
        self.device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Per-chain placement: all R chains start from the same initial
        # placement (the SA will diverge them via moves).
        if initial_placement is None:
            initial_placement = benchmark.macro_positions
        if initial_placement.shape != (benchmark.num_macros, 2):
            raise ValueError(
                f"initial_placement shape {tuple(initial_placement.shape)} "
                f"!= expected ({benchmark.num_macros}, 2)"
            )
        # [R, num_macros, 2]
        self.placement = initial_placement.to(self.device).unsqueeze(0).expand(
            self.R, -1, -1
        ).contiguous()

        # Grid geometry.
        self.rows = benchmark.grid_rows
        self.cols = benchmark.grid_cols
        self.canvas_w = float(benchmark.canvas_width)
        self.canvas_h = float(benchmark.canvas_height)
        self.cell_w = self.canvas_w / self.cols
        self.cell_h = self.canvas_h / self.rows
        self.cell_area = self.cell_w * self.cell_h

        # HPWL normalization.
        total_connections = float(benchmark.net_weights.sum().item())
        self.hpwl_norm = (self.canvas_w + self.canvas_h) * total_connections

        # Macro sizes etc. — move to device once.
        self.macro_sizes = benchmark.macro_sizes.to(self.device)         # [num_macros, 2]
        self.net_weights = benchmark.net_weights.to(self.device)         # [num_nets]
        self.port_positions = benchmark.port_positions.to(self.device)   # [num_ports, 2]

        # Cached: max rectangle in grid cells that any macro footprint
        # can occupy. Used to size the contribution tensors.
        max_w = float(self.macro_sizes[:, 0].max().item())
        max_h = float(self.macro_sizes[:, 1].max().item())
        # +2 = one cell of padding on each side for footprints straddling
        # cell boundaries. Cheap to be generous.
        self.max_k_c = int(max_w // self.cell_w) + 2
        self.max_k_r = int(max_h // self.cell_h) + 2

        # Padded macro→nets structure.
        self._macro_nets_padded, self._nets_per_macro = _make_macro_to_nets_padded(
            benchmark, self.device,
        )
        self._max_nets_per_macro = self._macro_nets_padded.shape[1]

        # Padded net-pin table.
        (
            self._net_owner_padded,
            self._net_slot_padded,
            self._net_pin_mask,
            self._net_pin_offsets_static,
        ) = _make_net_pin_table(benchmark, self.device)
        self._max_pins_per_net = self._net_owner_padded.shape[1]
        self.num_nets = benchmark.num_nets
        self.num_hard = benchmark.num_hard_macros
        self.num_macros = benchmark.num_macros
        self.num_ports = benchmark.port_positions.shape[0]

        # Allocate per-chain grids and HPWLs.
        self.density_grid = torch.zeros(
            (self.R, self.rows, self.cols), device=self.device
        )
        self.congestion_grid = torch.zeros(
            (self.R, self.rows, self.cols), device=self.device
        )
        self.net_hpwls = torch.zeros((self.R, self.num_nets), device=self.device)
        self.total_weighted_hpwl = torch.zeros(self.R, device=self.device)
        # Overlap state per chain.
        self.total_overlap_area = torch.zeros(self.R, device=self.device)
        self.overlap_count = torch.zeros(self.R, dtype=torch.long, device=self.device)

        # Precompute the "all owner positions" tensor: [R, num_macros + num_ports, 2].
        # Ports are static, macros come from self.placement.
        self._build_all_owner_pos()

        # Initial full recompute of all per-chain costs.
        self._full_recompute()

    # ---- initial full recompute ----------------------------------------

    def _build_all_owner_pos(self):
        """
        Materialize all_owner_pos = [R, num_macros + num_ports, 2] from
        placement + port positions. Ports are broadcast identically across
        all R chains.
        """
        ports_bcast = self.port_positions.unsqueeze(0).expand(self.R, -1, -1)
        self.all_owner_pos = torch.cat([self.placement, ports_bcast], dim=1)

    def _full_recompute(self):
        """
        Build density grid, congestion grid, all net HPWLs, and overlap
        counts from scratch for all R chains. Called once at init.
        """
        # 1. Density grid: add each macro's contribution.
        self.density_grid.zero_()
        for m in range(self.num_macros):
            contrib, r_s, c_s = _density_contribution(
                self.placement[:, m, :],
                self.macro_sizes[m],
                self.cell_w, self.cell_h, self.rows, self.cols,
                self.max_k_r, self.max_k_c,
            )
            _scatter_rect_into_grid(
                self.density_grid, contrib, r_s, c_s, sign=+1.0
            )

        # 2. Congestion grid: per-net contribution (which depends on the
        # bounding box of each net's pins, then a similar separable
        # outer product into the affected cells).
        self.congestion_grid.zero_()
        self._add_congestion_for_all_nets(sign=+1.0)

        # 3. HPWL per net per chain.
        self._recompute_all_hpwls()

        # 4. Overlap: pairwise across hard macros, vectorized per chain.
        self._recompute_overlaps_from_scratch()

    # ---- HPWL ------------------------------------------------------------

    def _recompute_all_hpwls(self):
        """
        Compute self.net_hpwls : [R, num_nets] and the weighted sum.
        Vectorized over (R, num_nets, max_pins).
        """
        # owner_idx[n, p] → position [R, 2]
        # all_owner_pos shape [R, num_macros + num_ports, 2]
        owner = self._net_owner_padded  # [N, P]
        mask = self._net_pin_mask        # [N, P]
        # Replace -1 with 0 so gather doesn't crash; we'll mask after.
        owner_safe = owner.clamp(min=0)
        # Gather: [R, N, P, 2]
        base = self.all_owner_pos[:, owner_safe, :]              # [R, N, P, 2]
        offs = self._net_pin_offsets_static.unsqueeze(0)         # [1, N, P, 2]
        pin_abs = base + offs                                    # [R, N, P, 2]

        # Mask invalid pins by setting them to +inf for the min computation
        # and -inf for max. This way they don't affect the bbox.
        mask_b = mask.unsqueeze(0).unsqueeze(-1)                 # [1, N, P, 1]
        pin_for_min = pin_abs.masked_fill(~mask_b, float("inf"))
        pin_for_max = pin_abs.masked_fill(~mask_b, float("-inf"))

        x_min = pin_for_min[..., 0].min(dim=2).values            # [R, N]
        x_max = pin_for_max[..., 0].max(dim=2).values            # [R, N]
        y_min = pin_for_min[..., 1].min(dim=2).values            # [R, N]
        y_max = pin_for_max[..., 1].max(dim=2).values            # [R, N]

        self.net_hpwls = (x_max - x_min) + (y_max - y_min)       # [R, N]
        self.total_weighted_hpwl = (
            self.net_hpwls * self.net_weights.unsqueeze(0)
        ).sum(dim=1)                                             # [R]

    def _recompute_hpwls_for_nets(self, net_idx: torch.Tensor) -> torch.Tensor:
        """
        Recompute HPWL on a subset of nets (e.g. nets touching a moved
        macro). `net_idx` is a [R, K] long tensor where each row lists
        the K affected nets for that chain (padded with -1).

        Returns the new HPWLs as [R, K]. Caller is responsible for
        writing them into self.net_hpwls.
        """
        R, K = net_idx.shape
        # Replace padding (-1) with 0 so gather works; mask away later.
        valid = net_idx >= 0                                     # [R, K]
        net_idx_safe = net_idx.clamp(min=0)                      # [R, K]

        # Pin owner/slot tables for the selected nets.
        # owner table: [num_nets, P] → gather along net axis with
        # net_idx_safe gives [R, K, P].
        owner = self._net_owner_padded[net_idx_safe]             # [R, K, P]
        mask  = self._net_pin_mask[net_idx_safe]                 # [R, K, P]
        offs  = self._net_pin_offsets_static[net_idx_safe]       # [R, K, P, 2]
        owner_safe = owner.clamp(min=0)                          # [R, K, P]

        # Gather positions from each chain's own all_owner_pos.
        # all_owner_pos: [R, M+ports, 2]. We want, for each (r, k, p):
        # all_owner_pos[r, owner_safe[r,k,p], :].
        R_idx = torch.arange(R, device=self.device).view(R, 1, 1).expand_as(owner_safe)
        base = self.all_owner_pos[R_idx, owner_safe, :]          # [R, K, P, 2]

        pin_abs = base + offs                                    # [R, K, P, 2]
        mask_b = mask.unsqueeze(-1)                              # [R, K, P, 1]
        pin_for_min = pin_abs.masked_fill(~mask_b, float("inf"))
        pin_for_max = pin_abs.masked_fill(~mask_b, float("-inf"))

        x_min = pin_for_min[..., 0].min(dim=2).values            # [R, K]
        x_max = pin_for_max[..., 0].max(dim=2).values
        y_min = pin_for_min[..., 1].min(dim=2).values
        y_max = pin_for_max[..., 1].max(dim=2).values

        hpwl = (x_max - x_min) + (y_max - y_min)                 # [R, K]
        # Mask out padding net slots → set them to 0 so they don't matter.
        hpwl = hpwl * valid
        return hpwl

    # ---- Congestion ------------------------------------------------------

    def _add_congestion_for_all_nets(self, sign: float):
        """Iterate nets and add (or subtract) their congestion contribution."""
        for n in range(self.num_nets):
            self._add_congestion_for_net(n, sign)

    def _add_congestion_for_net(self, net_idx: int, sign: float):
        """
        Same outer-product rectangle approach as density, but the scaling
        factor depends on the net's bounding box. Computed per chain.
        """
        owner = self._net_owner_padded[net_idx]                  # [P]
        mask  = self._net_pin_mask[net_idx]                      # [P]
        if not mask.any():
            return
        owner_safe = owner.clamp(min=0)
        # [R, P, 2]
        base = self.all_owner_pos[:, owner_safe, :]
        # Note: congestion ignores pin offsets in the original code (uses
        # macro centers / port positions directly). Keep that behavior.
        mask_b = mask.unsqueeze(0).unsqueeze(-1)
        pin_for_min = base.masked_fill(~mask_b, float("inf"))
        pin_for_max = base.masked_fill(~mask_b, float("-inf"))
        lx = pin_for_min[..., 0].min(dim=1).values               # [R]
        ux = pin_for_max[..., 0].max(dim=1).values
        by = pin_for_min[..., 1].min(dim=1).values
        ty = pin_for_max[..., 1].max(dim=1).values
        w = ux - lx
        h = ty - by
        # If both w and h are 0 the net is a single-point net — skip.
        degenerate = (w == 0) & (h == 0)
        # Net demand density (positive scalar per chain). Avoid /0.
        demand_density = (w + h) / (w * h + 1e-6)
        # Mask out degenerate chains: set their contribution to 0.
        demand_density = torch.where(degenerate, torch.zeros_like(demand_density), demand_density)

        avg_cap = (
            self.cell_w * float(self.benchmark.vroutes_per_micron) +
            self.cell_h * float(self.benchmark.hroutes_per_micron)
        ) / 2.0
        scale = sign * demand_density / (avg_cap + 1e-6)         # [R]

        # Compute the bounding rectangle in grid cells, per chain.
        c_start = torch.clamp((lx / self.cell_w).floor().long(), 0, self.cols - 1)
        c_end   = torch.clamp((ux / self.cell_w).floor().long(), 0, self.cols - 1)
        r_start = torch.clamp((by / self.cell_h).floor().long(), 0, self.rows - 1)
        r_end   = torch.clamp((ty / self.cell_h).floor().long(), 0, self.rows - 1)

        # Worst-case rectangle size: spans the whole grid. We don't
        # pre-cap this like we do for macros because nets can have wide
        # bounding boxes. So use the full grid as the upper bound.
        max_k_c = self.cols
        max_k_r = self.rows
        device = self.device

        k_c = torch.arange(max_k_c, device=device, dtype=torch.long)
        col_idx = c_start.unsqueeze(1) + k_c.unsqueeze(0)        # [R, max_k_c]
        cell_left = col_idx.float() * self.cell_w
        cell_right = cell_left + self.cell_w
        inter_w = torch.clamp(
            torch.minimum(cell_right, ux.unsqueeze(1)) -
            torch.maximum(cell_left, lx.unsqueeze(1)),
            min=0.0,
        )
        col_valid = col_idx <= c_end.unsqueeze(1)
        inter_w = inter_w * col_valid

        k_r = torch.arange(max_k_r, device=device, dtype=torch.long)
        row_idx = r_start.unsqueeze(1) + k_r.unsqueeze(0)
        cell_bot = row_idx.float() * self.cell_h
        cell_top = cell_bot + self.cell_h
        inter_h = torch.clamp(
            torch.minimum(cell_top, ty.unsqueeze(1)) -
            torch.maximum(cell_bot, by.unsqueeze(1)),
            min=0.0,
        )
        row_valid = row_idx <= r_end.unsqueeze(1)
        inter_h = inter_h * row_valid

        # contrib[r, ki, kj] = scale[r] * inter_h[r, ki] * inter_w[r, kj]
        contrib = (
            scale.view(self.R, 1, 1)
            * inter_h.unsqueeze(2)
            * inter_w.unsqueeze(1)
        )
        _scatter_rect_into_grid(
            self.congestion_grid, contrib, r_start, c_start, sign=1.0,
            # We already baked sign into contrib via scale, so the
            # scatter just adds.
        )

    # ---- Overlap ---------------------------------------------------------

    def _recompute_overlaps_from_scratch(self):
        """
        Compute total_overlap_area [R] and overlap_count [R] across all
        hard-macro pairs, vectorized.
        """
        if self.num_hard < 2:
            self.total_overlap_area.zero_()
            self.overlap_count.zero_()
            return
        # [R, H, 2], [H, 2]
        hp = self.placement[:, : self.num_hard, :]
        hs = self.macro_sizes[: self.num_hard]
        # Pairwise (i, j) distance: [R, H, H]
        dx = (hp[:, :, 0].unsqueeze(2) - hp[:, :, 0].unsqueeze(1)).abs()
        dy = (hp[:, :, 1].unsqueeze(2) - hp[:, :, 1].unsqueeze(1)).abs()
        # Min separation in each axis: [H, H]
        sep_x = (hs[:, 0].unsqueeze(1) + hs[:, 0].unsqueeze(0)) / 2.0
        sep_y = (hs[:, 1].unsqueeze(1) + hs[:, 1].unsqueeze(0)) / 2.0
        # Overlap in each axis: clamp(sep - dist, min=0). Self-overlap is
        # zero because dx[i,i] = 0 and sep[i,i] = w[i] (so overlap_x = w[i])
        # which we then zero out via the diagonal mask.
        ox = torch.clamp(sep_x - dx, min=0.0)
        oy = torch.clamp(sep_y - dy, min=0.0)
        area = ox * oy                                            # [R, H, H]
        # Zero out the diagonal (self) and lower triangle (double-count).
        H = self.num_hard
        eye = torch.eye(H, dtype=torch.bool, device=self.device)
        upper = torch.triu(torch.ones((H, H), dtype=torch.bool, device=self.device), diagonal=1)
        area = area * upper.unsqueeze(0)
        self.total_overlap_area = area.sum(dim=(1, 2))
        self.overlap_count = (area > 0).sum(dim=(1, 2))

    # ---- Cost reporting --------------------------------------------------

    def get_costs(self) -> Dict[str, torch.Tensor]:
        """
        Return per-chain cost components.

        Each value is a [R] tensor.
        """
        wl_cost = self.total_weighted_hpwl / self.hpwl_norm if self.hpwl_norm != 0 \
            else torch.zeros_like(self.total_weighted_hpwl)

        # Top-10% density mean per chain. density_grid: [R, rows, cols].
        density_map = self.density_grid / self.cell_area
        flat_d = density_map.view(self.R, -1)
        k_d = max(1, int(self.rows * self.cols * 0.1))
        top_d_vals, _ = torch.topk(flat_d, k_d, dim=1)
        den_cost = 0.5 * top_d_vals.mean(dim=1)

        # Top-5% congestion.
        flat_c = self.congestion_grid.view(self.R, -1)
        k_c = max(1, int(self.rows * self.cols * 0.05))
        top_c_vals, _ = torch.topk(flat_c, k_c, dim=1)
        cong_cost = top_c_vals.mean(dim=1)

        proxy = (
            self.weights["wirelength"] * wl_cost
            + self.weights["density"] * den_cost
            + self.weights["congestion"] * cong_cost
        )

        return {
            "proxy_cost": proxy,
            "wirelength_cost": wl_cost,
            "density_cost": den_cost,
            "congestion_cost": cong_cost,
            "overlap_count": self.overlap_count.clone(),
            "total_overlap_area": self.total_overlap_area.clone(),
        }

    # ---- Move proposal + apply (combined) ─────────────────────────────────

    def step_moves(
        self,
        macro_idx: torch.Tensor,           # [R] long, which macro each chain moves
        new_pos: torch.Tensor,             # [R, 2] float, the proposed center
        temperatures: torch.Tensor,        # [R] float
        rng_values: torch.Tensor,          # [R] float in [0,1) for Metropolis
        forbid_overlap: bool = True,       # hard-reject moves that create overlap
    ) -> Dict[str, torch.Tensor]:
        """
        Per-chain: propose one move, decide accept/reject, commit if accepted.

        All R chains' decisions happen in a single batched pass. The
        decision rule is:
          1. Compute the new cost as if the move were applied.
          2. If forbid_overlap and the move would create overlap > 0, reject.
          3. Else Metropolis: accept with probability min(1, exp(-Δ/T)).

        Returns a dict of [R] tensors describing what happened:
          accepted      : bool — chain's state was changed
          delta_cost    : float — cost change (positive = uphill)
          new_cost      : float — proxy cost after the move (whether
                          accepted or not — accepted chains have this as
                          their new state's cost; rejected chains
                          would have had this cost)
          new_overlap_count : long — overlap count under hypothetical
                          new state
        """
        R = self.R
        assert macro_idx.shape == (R,), f"macro_idx must be [{R}], got {tuple(macro_idx.shape)}"
        assert new_pos.shape == (R, 2)
        assert temperatures.shape == (R,)
        assert rng_values.shape == (R,)

        device = self.device

        # Snapshot the per-chain quantities we'll need to undo on reject.
        # We snapshot the SUMS, not the full grids — the grids we mutate
        # in place and then conditionally reverse on rejected chains.
        old_pos = self.placement[torch.arange(R, device=device), macro_idx, :].clone()  # [R, 2]
        # Sizes of the moved macros (per chain) — gathered.
        sizes = self.macro_sizes[macro_idx]                                              # [R, 2]
        # Net list affected by this move (per chain) — padded.
        # macro_to_nets is [num_macros, max_nets]; gather along axis 0.
        affected_nets = self._macro_nets_padded[macro_idx]                               # [R, K]
        K_max = affected_nets.shape[1]

        # 1) Apply optimistically to density grid: subtract old footprint,
        # add new footprint. Snapshot is the delta itself — we know how
        # to reverse.
        contrib_old, r_s_old, c_s_old = self._density_contribution_per_chain(
            old_pos, sizes
        )
        contrib_new, r_s_new, c_s_new = self._density_contribution_per_chain(
            new_pos, sizes
        )
        # Subtract old.
        self._scatter_per_chain_delta(self.density_grid, contrib_old, r_s_old, c_s_old, sign=-1.0)
        # Add new.
        self._scatter_per_chain_delta(self.density_grid, contrib_new, r_s_new, c_s_new, sign=+1.0)

        # 2) Update placement and all_owner_pos optimistically. Snapshot
        # the per-chain old position so we can roll back.
        chain_arange = torch.arange(R, device=device)
        self.placement[chain_arange, macro_idx, :] = new_pos
        self.all_owner_pos[chain_arange, macro_idx, :] = new_pos

        # 3) Update congestion grid for nets affected by this macro
        # (subtract old contribution, add new). The "old" contribution
        # is the one we'd compute using positions before our placement
        # update — but we just overwrote placement. Trick: revert
        # placement temporarily, compute old contrib, revert back, then
        # add new contrib. To avoid the toggling, we compute both sets
        # of net bounding boxes from the saved positions explicitly.
        #
        # For now, do the safe thing: save old positions, run subtract,
        # then reapply new positions, run add. This costs us extra
        # writes but is correct.
        # Restore old positions, subtract old congestion contribution.
        self.placement[chain_arange, macro_idx, :] = old_pos
        self.all_owner_pos[chain_arange, macro_idx, :] = old_pos
        self._update_congestion_for_chain_nets(affected_nets, sign=-1.0)
        # Re-apply new positions, add new contribution.
        self.placement[chain_arange, macro_idx, :] = new_pos
        self.all_owner_pos[chain_arange, macro_idx, :] = new_pos
        self._update_congestion_for_chain_nets(affected_nets, sign=+1.0)

        # 4) Update HPWL for affected nets.
        old_hpwls = self._gather_net_hpwls(affected_nets)                                # [R, K]
        new_hpwls_raw = self._recompute_hpwls_for_nets(affected_nets)                    # [R, K]
        # Mask off invalid (padded) net slots: their old/new are both 0.
        valid_net = affected_nets >= 0                                                   # [R, K]
        old_hpwls = old_hpwls * valid_net
        new_hpwls = new_hpwls_raw * valid_net
        # Apply delta to total_weighted_hpwl.
        weights_for_affected = torch.where(
            valid_net,
            self.net_weights[affected_nets.clamp(min=0)],
            torch.zeros_like(old_hpwls),
        )
        delta_weighted_hpwl = ((new_hpwls - old_hpwls) * weights_for_affected).sum(dim=1)  # [R]
        self.total_weighted_hpwl += delta_weighted_hpwl
        # Write new per-net HPWLs (only valid slots).
        self._scatter_net_hpwls(affected_nets, new_hpwls, valid_net)

        # 5) Recompute overlap state for the moved macro. Only the
        # moved macro's row/column in the pairwise table changes.
        new_overlap_count_total, new_overlap_area_total = self._overlap_after_single_move(
            macro_idx, old_pos, new_pos
        )
        # Snapshot old overlap state for rollback.
        old_overlap_count = self.overlap_count.clone()
        old_overlap_area = self.total_overlap_area.clone()
        self.overlap_count = new_overlap_count_total
        self.total_overlap_area = new_overlap_area_total

        # 6) Get the new per-chain cost.
        costs = self.get_costs()
        new_cost = costs["proxy_cost"]                                                   # [R]
        # We need old cost — but we already overwrote state. The "old"
        # cost equals new_cost - delta_cost; we compute delta_cost via
        # the inverse symbolic path. Easiest correct way: track the
        # pre-move cost externally. To do that we save the cost from
        # the previous step in self._last_cost. On first call we have
        # to compute it from scratch.
        if not hasattr(self, "_last_cost") or self._last_cost is None:
            # First time: we just changed the state, so to get pre-move
            # cost we have to recompute under the old state. Cheapest:
            # actually compute the old cost up-front (before mutating).
            # For now, since we've already mutated, fall back to
            # approximate: delta_cost = new_cost - "what cost would be"
            # under undone state. We undo, get_cost, redo. Costly but
            # correct.
            #
            # NOTE: callers should set est._last_cost after init via
            # get_costs() to avoid this branch. The SA does that.
            self._undo_move_state(
                macro_idx, old_pos, new_pos, sizes,
                contrib_old, r_s_old, c_s_old,
                contrib_new, r_s_new, c_s_new,
                affected_nets, old_hpwls, new_hpwls, valid_net,
                weights_for_affected, delta_weighted_hpwl,
                old_overlap_count, old_overlap_area,
            )
            old_cost = self.get_costs()["proxy_cost"].clone()
            # Redo the move:
            self._redo_move_state(
                macro_idx, old_pos, new_pos, sizes,
                contrib_old, r_s_old, c_s_old,
                contrib_new, r_s_new, c_s_new,
                affected_nets, old_hpwls, new_hpwls, valid_net,
                weights_for_affected, delta_weighted_hpwl,
                new_overlap_count_total, new_overlap_area_total,
            )
        else:
            old_cost = self._last_cost

        delta_cost = new_cost - old_cost                                                 # [R]

        # 7) Decide accept per chain.
        new_oc = self.overlap_count                                                      # [R]
        if forbid_overlap:
            overlap_forbid = new_oc > 0
        else:
            overlap_forbid = torch.zeros(R, dtype=torch.bool, device=device)

        # Metropolis: accept if delta<=0 OR rng < exp(-delta/T). T<=0
        # rejects all uphill.
        downhill = delta_cost <= 0
        # Safe exp: clamp -delta/T to a reasonable range.
        ratio = -delta_cost / temperatures.clamp(min=1e-12)
        ratio = ratio.clamp(min=-50.0, max=0.0)  # never accept with >100% prob; never overflow
        exp_p = torch.exp(ratio)
        metropolis = (rng_values < exp_p) | downhill

        accepted = metropolis & (~overlap_forbid)                                        # [R] bool

        # 8) For rejected chains, roll back all our optimistic changes.
        #    Use accepted mask to selectively undo.
        if (~accepted).any():
            self._partial_undo(
                ~accepted,
                macro_idx, old_pos, new_pos, sizes,
                contrib_old, r_s_old, c_s_old,
                contrib_new, r_s_new, c_s_new,
                affected_nets, old_hpwls, new_hpwls, valid_net,
                weights_for_affected, delta_weighted_hpwl,
                old_overlap_count, old_overlap_area,
            )

        # 9) Update _last_cost cache for accepted chains; rejected chains
        # keep their old cost.
        if hasattr(self, "_last_cost") and self._last_cost is not None:
            self._last_cost = torch.where(accepted, new_cost, old_cost)
        else:
            # If we got here without _last_cost set, set it now from the
            # final per-chain state.
            self._last_cost = torch.where(accepted, new_cost, old_cost)

        return {
            "accepted": accepted,
            "delta_cost": delta_cost,
            "new_cost": self._last_cost.clone(),
            "new_overlap_count": new_oc.clone(),
        }

    # ---- helpers for step_moves ─────────────────────────────────────────

    def _density_contribution_per_chain(
        self, pos: torch.Tensor, sizes: torch.Tensor,
    ):
        """
        Per-chain version of _density_contribution: takes pos [R, 2] and
        SIZES [R, 2] (one per chain — different chains may move
        different macros with different sizes).
        """
        R = pos.shape[0]
        device = pos.device

        m_lx = pos[:, 0] - sizes[:, 0] / 2
        m_ux = pos[:, 0] + sizes[:, 0] / 2
        m_by = pos[:, 1] - sizes[:, 1] / 2
        m_ty = pos[:, 1] + sizes[:, 1] / 2

        c_start = torch.clamp((m_lx / self.cell_w).floor().long(), 0, self.cols - 1)
        c_end = torch.clamp((m_ux / self.cell_w).floor().long(), 0, self.cols - 1)
        r_start = torch.clamp((m_by / self.cell_h).floor().long(), 0, self.rows - 1)
        r_end = torch.clamp((m_ty / self.cell_h).floor().long(), 0, self.rows - 1)

        k_c = torch.arange(self.max_k_c, device=device, dtype=torch.long)
        col_idx = c_start.unsqueeze(1) + k_c.unsqueeze(0)
        cell_left = col_idx.float() * self.cell_w
        cell_right = cell_left + self.cell_w
        inter_w = torch.clamp(
            torch.minimum(cell_right, m_ux.unsqueeze(1)) -
            torch.maximum(cell_left, m_lx.unsqueeze(1)),
            min=0.0,
        )
        inter_w = inter_w * (col_idx <= c_end.unsqueeze(1))

        k_r = torch.arange(self.max_k_r, device=device, dtype=torch.long)
        row_idx = r_start.unsqueeze(1) + k_r.unsqueeze(0)
        cell_bot = row_idx.float() * self.cell_h
        cell_top = cell_bot + self.cell_h
        inter_h = torch.clamp(
            torch.minimum(cell_top, m_ty.unsqueeze(1)) -
            torch.maximum(cell_bot, m_by.unsqueeze(1)),
            min=0.0,
        )
        inter_h = inter_h * (row_idx <= r_end.unsqueeze(1))

        contrib = inter_h.unsqueeze(2) * inter_w.unsqueeze(1)            # [R, max_k_r, max_k_c]
        return contrib, r_start, c_start

    def _scatter_per_chain_delta(self, grid, contrib, r_start, c_start, sign):
        """Same as _scatter_rect_into_grid but inlined to keep this readable."""
        _scatter_rect_into_grid(grid, contrib, r_start, c_start, sign=sign)

    def _update_congestion_for_chain_nets(
        self, affected_nets: torch.Tensor, sign: float,
    ):
        """
        For each chain r, iterate the affected_nets[r] list and update
        the congestion grid by `sign * contribution`.

        Implementation note: this version loops over the (small) padded
        net axis K with a Python loop, applying one batched per-net
        update per iteration. This is fine because K (max nets per
        macro) is small (~dozens). We could do all K simultaneously
        as a [R, K]-batched update but the indexing math gets gnarly
        with congestion's per-net scaling factor — defer.
        """
        K = affected_nets.shape[1]
        if K == 0:
            return
        for k in range(K):
            net_per_chain = affected_nets[:, k]                          # [R]
            self._add_congestion_for_chain_nets_one_slot(net_per_chain, sign)

    def _add_congestion_for_chain_nets_one_slot(
        self, net_per_chain: torch.Tensor, sign: float,
    ):
        """
        For each chain r, apply the congestion contribution of net
        net_per_chain[r] (with `sign`). Chains where net_per_chain[r]
        is -1 (padding) are no-op'd.
        """
        R = self.R
        device = self.device
        valid = net_per_chain >= 0
        if not valid.any():
            return
        # Net pin tables: gather along axis 0 with net_per_chain.
        safe = net_per_chain.clamp(min=0)
        owner = self._net_owner_padded[safe]                             # [R, P]
        pin_mask = self._net_pin_mask[safe]                              # [R, P]
        owner_safe = owner.clamp(min=0)
        R_idx = torch.arange(R, device=device).view(R, 1).expand_as(owner_safe)
        base = self.all_owner_pos[R_idx, owner_safe, :]                  # [R, P, 2]
        # Per-net bbox per chain (ignoring pin offsets, like original).
        mask_b = pin_mask.unsqueeze(-1)
        pin_for_min = base.masked_fill(~mask_b, float("inf"))
        pin_for_max = base.masked_fill(~mask_b, float("-inf"))
        lx = pin_for_min[..., 0].min(dim=1).values
        ux = pin_for_max[..., 0].max(dim=1).values
        by = pin_for_min[..., 1].min(dim=1).values
        ty = pin_for_max[..., 1].max(dim=1).values
        w = ux - lx
        h = ty - by
        degenerate = (w == 0) & (h == 0)
        demand_density = (w + h) / (w * h + 1e-6)
        demand_density = torch.where(degenerate | (~valid), torch.zeros_like(demand_density), demand_density)

        avg_cap = (
            self.cell_w * float(self.benchmark.vroutes_per_micron) +
            self.cell_h * float(self.benchmark.hroutes_per_micron)
        ) / 2.0
        scale = sign * demand_density / (avg_cap + 1e-6)                 # [R]

        c_start = torch.clamp((lx / self.cell_w).floor().long(), 0, self.cols - 1)
        c_end = torch.clamp((ux / self.cell_w).floor().long(), 0, self.cols - 1)
        r_start = torch.clamp((by / self.cell_h).floor().long(), 0, self.rows - 1)
        r_end = torch.clamp((ty / self.cell_h).floor().long(), 0, self.rows - 1)

        # Use the full grid as max rectangle, like in _add_congestion_for_net.
        # In practice nets' bboxes are small; we can size this to a
        # reasonable max if memory becomes a problem.
        max_k_c = self.cols
        max_k_r = self.rows
        k_c = torch.arange(max_k_c, device=device, dtype=torch.long)
        col_idx = c_start.unsqueeze(1) + k_c.unsqueeze(0)
        cell_left = col_idx.float() * self.cell_w
        cell_right = cell_left + self.cell_w
        inter_w = torch.clamp(
            torch.minimum(cell_right, ux.unsqueeze(1)) -
            torch.maximum(cell_left, lx.unsqueeze(1)),
            min=0.0,
        )
        inter_w = inter_w * (col_idx <= c_end.unsqueeze(1))
        k_r = torch.arange(max_k_r, device=device, dtype=torch.long)
        row_idx = r_start.unsqueeze(1) + k_r.unsqueeze(0)
        cell_bot = row_idx.float() * self.cell_h
        cell_top = cell_bot + self.cell_h
        inter_h = torch.clamp(
            torch.minimum(cell_top, ty.unsqueeze(1)) -
            torch.maximum(cell_bot, by.unsqueeze(1)),
            min=0.0,
        )
        inter_h = inter_h * (row_idx <= r_end.unsqueeze(1))
        contrib = (
            scale.view(R, 1, 1)
            * inter_h.unsqueeze(2)
            * inter_w.unsqueeze(1)
        )
        # Zero out contribution for invalid chains.
        contrib = contrib * valid.view(R, 1, 1)
        _scatter_rect_into_grid(self.congestion_grid, contrib, r_start, c_start, sign=1.0)

    def _gather_net_hpwls(self, net_idx: torch.Tensor) -> torch.Tensor:
        """Gather self.net_hpwls[r, net_idx[r, k]] → [R, K]."""
        R, K = net_idx.shape
        valid = net_idx >= 0
        safe = net_idx.clamp(min=0)
        R_idx = torch.arange(R, device=self.device).view(R, 1).expand_as(safe)
        out = self.net_hpwls[R_idx, safe]                                # [R, K]
        return out * valid

    def _scatter_net_hpwls(self, net_idx: torch.Tensor, new_hpwls: torch.Tensor,
                            valid: torch.Tensor):
        """
        Write new_hpwls[r, k] into self.net_hpwls[r, net_idx[r, k]] for
        every (r, k) where valid[r, k] is True.

        Subtle: `net_idx` has -1 padding which we clamp to 0 for indexing
        safety, but those padded slots' indices coincide with the real
        index 0, so we can't just write everything — the padded slots
        would overwrite legitimate slot-0 writes through duplicate-index
        assignment (whose order is unspecified in PyTorch).

        Solution: flatten to a 1-D index list, drop the invalid entries,
        and use index_put_ on (chain, net) tuples. This way invalid
        slots contribute nothing.
        """
        R, K = net_idx.shape
        flat_net = net_idx.view(-1)                   # [R*K]
        flat_valid = valid.view(-1)                   # [R*K]
        flat_vals = new_hpwls.view(-1)                # [R*K]
        # Build chain index 0,0,...,0,1,1,...,1,...
        chain_idx = torch.arange(R, device=self.device).repeat_interleave(K)
        # Filter to valid entries only.
        mask = flat_valid
        if not mask.any():
            return
        chains_sel = chain_idx[mask]
        nets_sel = flat_net[mask]
        vals_sel = flat_vals[mask]
        # Direct assignment is safe now: no duplicate (chain, net) pairs
        # arise from a single move (each affected net appears at most
        # once per chain).
        self.net_hpwls[chains_sel, nets_sel] = vals_sel

    def _overlap_after_single_move(
        self, macro_idx: torch.Tensor, old_pos: torch.Tensor, new_pos: torch.Tensor,
    ):
        """
        Recompute total_overlap_area and overlap_count after one macro
        per chain moved. Uses the trick: only overlaps involving the
        moved macro can change, so we subtract old contributions and
        add new ones for that macro alone.

        Returns (new_count [R], new_area [R]).
        """
        R = self.R
        num_hard = self.num_hard
        device = self.device

        # If the moved macro is soft, overlaps don't change at all.
        is_hard = macro_idx < num_hard
        if not is_hard.any():
            return self.overlap_count.clone(), self.total_overlap_area.clone()

        # For hard moves: compute old vs new overlap area with every other
        # hard macro. Positions of ALL hard macros (per chain) at current
        # state — note: at the point this is called, placement is already
        # set to NEW positions, so we need to compute "new" from
        # self.placement[:, :num_hard, :] and "old" by substituting old_pos
        # back in for the moved macro.

        # New positions [R, H, 2]
        new_hp = self.placement[:, :num_hard, :]
        # Sizes [H, 2]
        hs = self.macro_sizes[:num_hard]

        # For "old": copy new_hp, but at row macro_idx[r] put old_pos[r].
        # Only meaningful where is_hard.
        old_hp = new_hp.clone()
        if is_hard.any():
            r_idx = torch.arange(R, device=device)[is_hard]
            m_idx = macro_idx[is_hard]
            # old_hp[r, m_idx, :] = old_pos[r]
            old_hp[r_idx, m_idx, :] = old_pos[is_hard]

        # We compute pairwise overlap (moved macro vs all others) for new and old.
        # The moved macro index per chain is macro_idx (need a [R, 1, 2] lookup).
        chain_arange = torch.arange(R, device=device)

        def _pair_overlap_for_moved(hp: torch.Tensor) -> torch.Tensor:
            """For each chain r: overlap area between moved macro and each other hard macro. [R, H]"""
            moved = hp[chain_arange, macro_idx, :]                       # [R, 2]
            # Distances to each hard macro: [R, H]
            dx = (moved[:, 0:1] - hp[:, :, 0]).abs()
            dy = (moved[:, 1:2] - hp[:, :, 1]).abs()
            # Min sep: depends on the moved macro's size and each other's.
            moved_sizes = self.macro_sizes[macro_idx]                    # [R, 2]
            sep_x = (moved_sizes[:, 0:1] + hs[:, 0].unsqueeze(0)) / 2.0  # [R, H]
            sep_y = (moved_sizes[:, 1:2] + hs[:, 1].unsqueeze(0)) / 2.0  # [R, H]
            ox = torch.clamp(sep_x - dx, min=0.0)
            oy = torch.clamp(sep_y - dy, min=0.0)
            area = ox * oy
            # Zero out self-overlap (where index == macro_idx)
            self_mask = torch.arange(num_hard, device=device).unsqueeze(0) == macro_idx.unsqueeze(1)
            area = area.masked_fill(self_mask, 0.0)
            return area

        new_area_per_other = _pair_overlap_for_moved(new_hp)             # [R, H]
        old_area_per_other = _pair_overlap_for_moved(old_hp)             # [R, H]

        # Delta in total area: sum over the H axis. Note: each pairwise
        # overlap is counted once (we only iterated over moved vs others,
        # not all pairs). Self-overlaps are zeroed.
        delta_area = (new_area_per_other - old_area_per_other).sum(dim=1)  # [R]
        # Delta in count: new>0 minus old>0.
        delta_count = ((new_area_per_other > 0).sum(dim=1)
                       - (old_area_per_other > 0).sum(dim=1))            # [R]

        # Only update where the moved macro was hard. (Soft moves don't
        # affect overlap accounting.)
        new_area_total = self.total_overlap_area + delta_area * is_hard
        new_count_total = self.overlap_count + delta_count * is_hard
        return new_count_total, new_area_total

    def _partial_undo(
        self, undo_mask, macro_idx, old_pos, new_pos, sizes,
        contrib_old, r_s_old, c_s_old,
        contrib_new, r_s_new, c_s_new,
        affected_nets, old_hpwls, new_hpwls, valid_net,
        weights_for_affected, delta_weighted_hpwl,
        old_overlap_count, old_overlap_area,
    ):
        """
        For chains where undo_mask is True, reverse all the optimistic
        state changes done in step_moves. We do this by zeroing the
        contrib tensors for chains we DON'T need to undo, then scattering
        the reverse direction.
        """
        device = self.device
        R = self.R
        mask_f = undo_mask.float().view(R, 1, 1)                          # [R, 1, 1]

        # 1) Density: subtract (-1 × undo) means: add contrib_old back,
        # subtract contrib_new. Multiply both by undo_mask to zero out
        # the chains we don't undo.
        _scatter_rect_into_grid(
            self.density_grid, contrib_old * mask_f, r_s_old, c_s_old, sign=+1.0,
        )
        _scatter_rect_into_grid(
            self.density_grid, contrib_new * mask_f, r_s_new, c_s_new, sign=-1.0,
        )

        # 2) Congestion grid: re-run with reversed signs on undo chains.
        # This is the expensive part because per-net iteration. For each
        # affected net per chain, we need to ADD old congestion and
        # SUBTRACT new congestion — but only for undo chains. The
        # easiest correct path: run _update_congestion_for_chain_nets
        # again with sign=+1 (add back old) using positions = old_pos
        # for the undo chains, then sign=-1 (remove new) using new_pos
        # for the undo chains. But our placement is currently at new_pos,
        # so we juggle.
        # Save current positions (= new positions, applied earlier).
        chain_arange = torch.arange(R, device=device)
        cur_pos = self.placement[chain_arange, macro_idx, :].clone()       # = new_pos
        # Step A: subtract NEW contribution at undo chains.
        # Only do anything on undo chains; for accept chains we set
        # affected_nets[r,:] = -1 so the per-net loop is a no-op.
        affected_nets_masked = torch.where(
            undo_mask.view(R, 1).expand_as(affected_nets),
            affected_nets,
            torch.full_like(affected_nets, -1),
        )
        self._update_congestion_for_chain_nets(affected_nets_masked, sign=-1.0)
        # Step B: put positions back to OLD for undo chains, add OLD contribution.
        self.placement[chain_arange[undo_mask], macro_idx[undo_mask], :] = old_pos[undo_mask]
        self.all_owner_pos[chain_arange[undo_mask], macro_idx[undo_mask], :] = old_pos[undo_mask]
        self._update_congestion_for_chain_nets(affected_nets_masked, sign=+1.0)

        # 3) HPWL: for undo chains, the affected nets need old_hpwls
        # restored. For accept chains, leave new_hpwls in place.
        # Total weighted hpwl: subtract delta on undo chains.
        self.total_weighted_hpwl -= delta_weighted_hpwl * undo_mask.float()
        # Per-net HPWLs: rewrite to old where undo, new where accept.
        # _scatter_net_hpwls used affected_nets — we re-write with
        # values chosen per chain.
        restored = torch.where(undo_mask.view(R, 1), old_hpwls, new_hpwls)
        self._scatter_net_hpwls(affected_nets, restored, valid_net)

        # 4) Overlap state: restore old where undo.
        self.overlap_count = torch.where(undo_mask, old_overlap_count, self.overlap_count)
        self.total_overlap_area = torch.where(undo_mask, old_overlap_area, self.total_overlap_area)

    def _undo_move_state(self, *args, **kwargs):
        """Undo ALL chains' optimistic changes (used only on first
        step_moves call, before _last_cost cache is populated). Just
        calls _partial_undo with everyone selected."""
        undo_all = torch.ones(self.R, dtype=torch.bool, device=self.device)
        self._partial_undo(undo_all, *args, **kwargs)

    def _redo_move_state(
        self, macro_idx, old_pos, new_pos, sizes,
        contrib_old, r_s_old, c_s_old,
        contrib_new, r_s_new, c_s_new,
        affected_nets, old_hpwls, new_hpwls, valid_net,
        weights_for_affected, delta_weighted_hpwl,
        new_overlap_count_total, new_overlap_area_total,
    ):
        """Re-apply ALL chains' changes after a temporary undo. Inverse
        of _undo_move_state."""
        chain_arange = torch.arange(self.R, device=self.device)
        # Density
        _scatter_rect_into_grid(self.density_grid, contrib_old, r_s_old, c_s_old, sign=-1.0)
        _scatter_rect_into_grid(self.density_grid, contrib_new, r_s_new, c_s_new, sign=+1.0)
        # Positions
        self.placement[chain_arange, macro_idx, :] = new_pos
        self.all_owner_pos[chain_arange, macro_idx, :] = new_pos
        # Congestion: subtract old (using old positions — but we just
        # set new positions). For correctness we toggle back.
        self.placement[chain_arange, macro_idx, :] = old_pos
        self.all_owner_pos[chain_arange, macro_idx, :] = old_pos
        self._update_congestion_for_chain_nets(affected_nets, sign=-1.0)
        self.placement[chain_arange, macro_idx, :] = new_pos
        self.all_owner_pos[chain_arange, macro_idx, :] = new_pos
        self._update_congestion_for_chain_nets(affected_nets, sign=+1.0)
        # HPWL
        self.total_weighted_hpwl += delta_weighted_hpwl
        self._scatter_net_hpwls(affected_nets, new_hpwls, valid_net)
        # Overlap
        self.overlap_count = new_overlap_count_total
        self.total_overlap_area = new_overlap_area_total

    # ---- PT swap ---------------------------------------------------------

    def swap_chains(self, i: int, j: int):
        """Exchange all per-chain state between chains i and j."""
        if i == j:
            return
        tensors = [
            self.placement,
            self.all_owner_pos,
            self.density_grid,
            self.congestion_grid,
            self.net_hpwls,
            self.total_weighted_hpwl,
            self.total_overlap_area,
            self.overlap_count,
        ]
        for T in tensors:
            tmp = T[i].clone()
            T[i] = T[j]
            T[j] = tmp
