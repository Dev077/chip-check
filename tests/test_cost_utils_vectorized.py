"""
Verify the vectorized _update_density_grid / _update_congestion_grid
produce numerically identical grids to the original double-loop math.

Self-contained: builds a tiny synthetic Benchmark, doesn't need
external/MacroPlacement data. The reference implementation is inlined
here as `_loop_density` / `_loop_congestion` (copies of the original
loops, kept verbatim).
"""

import sys
from pathlib import Path

import torch

# Make sure we import the modified cost_utils from our working tree.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macro_place.benchmark import Benchmark
from utils.cost_utils import CostEstimator


# ── Reference (original double-loop) implementations ──────────────────────


def _loop_density(grid, macro_sizes, macro_idx, pos, cell_w, cell_h,
                  rows, cols, sign):
    w, h = macro_sizes[macro_idx]
    x, y = pos
    m_lx, m_ux = x - w / 2, x + w / 2
    m_by, m_ty = y - h / 2, y + h / 2
    c_start = max(0, int(m_lx // cell_w))
    c_end = min(cols - 1, int(m_ux // cell_w))
    r_start = max(0, int(m_by // cell_h))
    r_end = min(rows - 1, int(m_ty // cell_h))
    for r in range(r_start, r_end + 1):
        for c in range(c_start, c_end + 1):
            inter_w = min(m_ux, (c + 1) * cell_w) - max(m_lx, c * cell_w)
            inter_h = min(m_ty, (r + 1) * cell_h) - max(m_by, r * cell_h)
            if inter_w > 0 and inter_h > 0:
                grid[r, c] += sign * (inter_w * inter_h)


def _loop_congestion(grid, lx, ux, by, ty, cell_w, cell_h, rows, cols,
                     vroutes, hroutes, sign):
    w = ux - lx
    h = ty - by
    if w == 0 and h == 0:
        return
    demand_density = (w + h) / (w * h + 1e-6)
    c_start = max(0, int(lx // cell_w))
    c_end = min(cols - 1, int(ux // cell_w))
    r_start = max(0, int(by // cell_h))
    r_end = min(rows - 1, int(ty // cell_h))
    avg_cap = (cell_w * vroutes + cell_h * hroutes) / 2.0
    for r in range(r_start, r_end + 1):
        for c in range(c_start, c_end + 1):
            inter_w = min(ux, (c + 1) * cell_w) - max(lx, c * cell_w)
            inter_h = min(ty, (r + 1) * cell_h) - max(by, r * cell_h)
            if inter_w > 0 and inter_h > 0:
                cell_demand = demand_density * (inter_w * inter_h)
                grid[r, c] += sign * (cell_demand / (avg_cap + 1e-6))


# ── Synthetic benchmark ───────────────────────────────────────────────────


def make_synthetic_benchmark():
    """Tiny but non-trivial: 4 hard macros, 2 soft, 3 nets, 1 port."""
    macro_positions = torch.tensor([
        [10.0, 10.0],   # hard 0: 4x6
        [25.0, 30.0],   # hard 1: 8x4
        [60.0, 50.0],   # hard 2: 10x10
        [80.0, 20.0],   # hard 3: 6x6
        [40.0, 40.0],   # soft 4: 3x3
        [50.0, 70.0],   # soft 5: 5x5
    ])
    macro_sizes = torch.tensor([
        [4.0, 6.0],
        [8.0, 4.0],
        [10.0, 10.0],
        [6.0, 6.0],
        [3.0, 3.0],
        [5.0, 5.0],
    ])
    macro_fixed = torch.tensor([False, False, False, True, False, False])
    macro_names = [f"m{i}" for i in range(6)]

    # Nets: just three small ones for the congestion test.
    # net 0: macros 0,1,2; net 1: macros 1,3 + port 0; net 2: macros 4,5
    # Pin-level (owner_idx, pin_slot). Soft + port pin_slot is 0 by convention.
    # We give hard macros one pin each at offset (0,0), to keep math simple.
    net_pin_nodes = [
        torch.tensor([[0, 0], [1, 0], [2, 0]], dtype=torch.long),
        torch.tensor([[1, 0], [3, 0], [6, 0]], dtype=torch.long),  # port idx = num_macros + 0
        torch.tensor([[4, 0], [5, 0]], dtype=torch.long),
    ]
    net_nodes = [t[:, 0] for t in net_pin_nodes]  # just used for indexing in other code
    net_weights = torch.tensor([1.0, 1.0, 1.0])

    port_positions = torch.tensor([[5.0, 90.0]])
    macro_pin_offsets = [
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
    ]

    bench = Benchmark(
        name="synthetic",
        canvas_width=100.0,
        canvas_height=100.0,
        num_macros=6,
        macro_positions=macro_positions,
        macro_sizes=macro_sizes,
        macro_fixed=macro_fixed,
        macro_names=macro_names,
        num_nets=3,
        net_nodes=net_nodes,
        net_weights=net_weights,
        grid_rows=10,
        grid_cols=10,
        port_positions=port_positions,
        macro_pin_offsets=macro_pin_offsets,
        net_pin_nodes=net_pin_nodes,
        num_hard_macros=4,
        num_soft_macros=2,
        hroutes_per_micron=1.5,
        vroutes_per_micron=1.2,
    )
    return bench


# ── Tests ─────────────────────────────────────────────────────────────────


def test_density_grid_match():
    """Estimator's density grid after full_recalculate must equal the loop result."""
    bench = make_synthetic_benchmark()
    est = CostEstimator(bench)  # uses vectorized update internally

    # Build the reference grid by running the loop on every macro.
    ref = torch.zeros((bench.grid_rows, bench.grid_cols))
    cell_w = bench.canvas_width / bench.grid_cols
    cell_h = bench.canvas_height / bench.grid_rows
    for i in range(bench.num_macros):
        _loop_density(
            ref, bench.macro_sizes, i, bench.macro_positions[i],
            cell_w, cell_h, bench.grid_rows, bench.grid_cols, sign=1.0,
        )

    diff = (est.density_grid - ref).abs().max().item()
    print(f"  density grid max abs diff: {diff:.2e}")
    assert diff < 1e-5, f"Density grid mismatch: max diff {diff}"
    print("  ✓ density grid matches reference")


def test_congestion_grid_match():
    bench = make_synthetic_benchmark()
    est = CostEstimator(bench)

    ref = torch.zeros((bench.grid_rows, bench.grid_cols))
    cell_w = bench.canvas_width / bench.grid_cols
    cell_h = bench.canvas_height / bench.grid_rows
    all_owner_pos = torch.cat([bench.macro_positions, bench.port_positions], dim=0)
    for net_pins in bench.net_pin_nodes:
        net_pos = all_owner_pos[net_pins[:, 0]]
        lx = net_pos[:, 0].min().item()
        ux = net_pos[:, 0].max().item()
        by = net_pos[:, 1].min().item()
        ty = net_pos[:, 1].max().item()
        _loop_congestion(
            ref, lx, ux, by, ty, cell_w, cell_h,
            bench.grid_rows, bench.grid_cols,
            bench.vroutes_per_micron, bench.hroutes_per_micron, sign=1.0,
        )

    diff = (est.congestion_grid - ref).abs().max().item()
    print(f"  congestion grid max abs diff: {diff:.2e}")
    assert diff < 1e-5, f"Congestion grid mismatch: max diff {diff}"
    print("  ✓ congestion grid matches reference")


def test_incremental_density_update():
    """Move a macro and check the grid stays consistent with a full rebuild."""
    bench = make_synthetic_benchmark()
    est = CostEstimator(bench)

    # Move macro 1 by (15, -8).
    new_pos = bench.macro_positions[1] + torch.tensor([15.0, -8.0])
    est.update_macro_pos(1, new_pos)

    # Independent rebuild from scratch with the new placement.
    bench2 = make_synthetic_benchmark()
    bench2.macro_positions = bench.macro_positions.clone()
    bench2.macro_positions[1] = new_pos
    est2 = CostEstimator(bench2)

    diff_d = (est.density_grid - est2.density_grid).abs().max().item()
    diff_c = (est.congestion_grid - est2.congestion_grid).abs().max().item()
    print(f"  density grid drift after move: {diff_d:.2e}")
    print(f"  congestion grid drift after move: {diff_c:.2e}")
    assert diff_d < 1e-4, f"Density drift {diff_d}"
    assert diff_c < 1e-4, f"Congestion drift {diff_c}"
    print("  ✓ incremental update stays consistent with full rebuild")


def test_edge_case_macro_on_grid_boundary():
    """Macro centered exactly on a grid cell boundary — common float-edge case."""
    bench = make_synthetic_benchmark()
    # Place a macro centered on a cell boundary (x=20 → col 2 boundary
    # when cell_w=10).
    bench.macro_positions[0] = torch.tensor([20.0, 20.0])
    est = CostEstimator(bench)

    ref = torch.zeros((bench.grid_rows, bench.grid_cols))
    cell_w = bench.canvas_width / bench.grid_cols
    cell_h = bench.canvas_height / bench.grid_rows
    for i in range(bench.num_macros):
        _loop_density(
            ref, bench.macro_sizes, i, bench.macro_positions[i],
            cell_w, cell_h, bench.grid_rows, bench.grid_cols, sign=1.0,
        )

    diff = (est.density_grid - ref).abs().max().item()
    print(f"  boundary case max abs diff: {diff:.2e}")
    assert diff < 1e-5, f"Boundary case mismatch: {diff}"
    print("  ✓ boundary case ok")


def test_macro_partially_outside_canvas():
    """A macro at the canvas edge should clip cleanly, not double-count."""
    bench = make_synthetic_benchmark()
    bench.macro_positions[0] = torch.tensor([1.0, 1.0])  # partially negative
    est = CostEstimator(bench)

    ref = torch.zeros((bench.grid_rows, bench.grid_cols))
    cell_w = bench.canvas_width / bench.grid_cols
    cell_h = bench.canvas_height / bench.grid_rows
    for i in range(bench.num_macros):
        _loop_density(
            ref, bench.macro_sizes, i, bench.macro_positions[i],
            cell_w, cell_h, bench.grid_rows, bench.grid_cols, sign=1.0,
        )

    diff = (est.density_grid - ref).abs().max().item()
    print(f"  edge-clip max abs diff: {diff:.2e}")
    assert diff < 1e-5, f"Edge-clip mismatch: {diff}"
    print("  ✓ edge-clipped macro ok")


if __name__ == "__main__":
    print("== test_density_grid_match ==")
    test_density_grid_match()
    print("\n== test_congestion_grid_match ==")
    test_congestion_grid_match()
    print("\n== test_incremental_density_update ==")
    test_incremental_density_update()
    print("\n== test_edge_case_macro_on_grid_boundary ==")
    test_edge_case_macro_on_grid_boundary()
    print("\n== test_macro_partially_outside_canvas ==")
    test_macro_partially_outside_canvas()
    print("\nAll vectorization tests passed.")
