"""
Verify BatchedCostEstimator matches the serial CostEstimator on initial
costs and overlap counts.

For now we only test the full-recompute path (no incremental moves);
that's what the batched file has implemented at this point.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from macro_place.benchmark import Benchmark
from utils.cost_utils import CostEstimator
from utils.batched_cost_utils import BatchedCostEstimator


def make_synthetic(num_hard=6, num_soft=4, num_nets=15, grid=10, canvas=100.0, seed=0):
    """Same shape as the SA test's synthetic — small, legal start."""
    g = torch.Generator().manual_seed(seed)
    num_macros = num_hard + num_soft
    side = int(num_hard ** 0.5) + 1
    spacing = canvas / (side + 1)
    hard_pos = []
    for i in range(num_hard):
        r, c = divmod(i, side)
        hard_pos.append([spacing * (c + 1), spacing * (r + 1)])
    hard_pos = torch.tensor(hard_pos, dtype=torch.float32)
    soft_pos = torch.rand(num_soft, 2, generator=g) * canvas
    macro_positions = torch.cat([hard_pos, soft_pos], dim=0)
    sizes_hard = torch.ones(num_hard, 2) * 6.0
    sizes_soft = torch.ones(num_soft, 2) * 3.0
    macro_sizes = torch.cat([sizes_hard, sizes_soft], dim=0)
    macro_fixed = torch.zeros(num_macros, dtype=torch.bool)
    net_pin_nodes = []
    net_nodes = []
    for _ in range(num_nets):
        k = int(torch.randint(2, 6, (1,), generator=g).item())
        owners = torch.randint(0, num_macros, (k,), generator=g)
        pins = torch.zeros((k, 2), dtype=torch.long)
        pins[:, 0] = owners
        net_pin_nodes.append(pins)
        net_nodes.append(torch.unique(owners))
    macro_pin_offsets = [
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) for _ in range(num_hard)
    ]
    return Benchmark(
        name="synth",
        canvas_width=canvas,
        canvas_height=canvas,
        num_macros=num_macros,
        macro_positions=macro_positions,
        macro_sizes=macro_sizes,
        macro_fixed=macro_fixed,
        macro_names=[f"m{i}" for i in range(num_macros)],
        num_nets=num_nets,
        net_nodes=net_nodes,
        net_weights=torch.ones(num_nets),
        grid_rows=grid,
        grid_cols=grid,
        port_positions=torch.zeros(0, 2),
        macro_pin_offsets=macro_pin_offsets,
        net_pin_nodes=net_pin_nodes,
        num_hard_macros=num_hard,
        num_soft_macros=num_soft,
        hroutes_per_micron=1.5,
        vroutes_per_micron=1.5,
    )


def test_density_grid_matches_serial():
    bench = make_synthetic()
    serial = CostEstimator(bench)
    batched = BatchedCostEstimator(bench, num_chains=3, device="cpu")

    # All R chains start identical; chain 0 should equal serial.
    for r in range(batched.R):
        diff = (batched.density_grid[r] - serial.density_grid).abs().max().item()
        assert diff < 1e-4, f"density mismatch on chain {r}: max diff {diff}"
    print(f"  ✓ density grid matches across all 3 chains (max diff < 1e-4)")


def test_congestion_grid_matches_serial():
    bench = make_synthetic()
    serial = CostEstimator(bench)
    batched = BatchedCostEstimator(bench, num_chains=3, device="cpu")
    for r in range(batched.R):
        diff = (batched.congestion_grid[r] - serial.congestion_grid).abs().max().item()
        assert diff < 1e-4, f"congestion mismatch on chain {r}: max diff {diff}"
    print(f"  ✓ congestion grid matches (max diff < 1e-4)")


def test_hpwl_matches_serial():
    bench = make_synthetic()
    serial = CostEstimator(bench)
    batched = BatchedCostEstimator(bench, num_chains=2, device="cpu")
    for r in range(batched.R):
        diff = (batched.net_hpwls[r] - serial.net_hpwls).abs().max().item()
        twd_diff = abs(batched.total_weighted_hpwl[r].item() - serial.total_weighted_hpwl)
        assert diff < 1e-3, f"per-net HPWL mismatch chain {r}: max diff {diff}"
        assert twd_diff < 1e-2, f"weighted HPWL sum mismatch chain {r}: {twd_diff}"
    print(f"  ✓ HPWL matches (per-net diff<1e-3, sum diff<1e-2)")


def test_overlap_matches_serial():
    """The synthetic benchmark has all hard macros on a coarse grid, so
    no overlaps. Both estimators should report 0."""
    bench = make_synthetic()
    serial = CostEstimator(bench)
    batched = BatchedCostEstimator(bench, num_chains=3, device="cpu")
    for r in range(batched.R):
        assert int(batched.overlap_count[r].item()) == int(serial.overlap_count), \
            f"overlap count chain {r}: batched={batched.overlap_count[r].item()} serial={serial.overlap_count}"
    print(f"  ✓ overlap count matches: both = {serial.overlap_count}")


def test_overlap_detects_overlaps():
    """Place two macros on top of each other and confirm both estimators
    catch it identically."""
    bench = make_synthetic()
    # Force macro 0 and macro 1 onto the same position.
    bench.macro_positions[1] = bench.macro_positions[0].clone()
    serial = CostEstimator(bench)
    batched = BatchedCostEstimator(bench, num_chains=2, device="cpu")
    assert serial.overlap_count > 0, "Setup failed; serial didn't see overlap"
    for r in range(batched.R):
        assert int(batched.overlap_count[r].item()) == int(serial.overlap_count), \
            f"chain {r}: batched={batched.overlap_count[r].item()} serial={serial.overlap_count}"
        area_diff = abs(batched.total_overlap_area[r].item() - serial.total_overlap_area)
        assert area_diff < 1e-3, f"chain {r}: area diff {area_diff}"
    print(f"  ✓ overlaps detected identically (count={serial.overlap_count}, "
          f"area={serial.total_overlap_area:.2f})")


def test_proxy_cost_matches_serial():
    bench = make_synthetic()
    serial = CostEstimator(bench)
    batched = BatchedCostEstimator(bench, num_chains=3, device="cpu")
    serial_costs = serial.get_costs()
    batched_costs = batched.get_costs()
    for r in range(batched.R):
        p_b = batched_costs["proxy_cost"][r].item()
        p_s = serial_costs["proxy_cost"]
        diff = abs(p_b - p_s)
        assert diff < 1e-4, f"chain {r} proxy diff {diff} (batched={p_b}, serial={p_s})"
    print(f"  ✓ proxy_cost matches: serial={serial_costs['proxy_cost']:.6f} "
          f"batched[0]={batched_costs['proxy_cost'][0].item():.6f}")


def test_distinct_chains_after_swap():
    """swap_chains should exchange state between two chains."""
    bench = make_synthetic()
    batched = BatchedCostEstimator(bench, num_chains=3, device="cpu")
    # Diverge chain 1 by manually mutating its placement.
    batched.placement[1, 0, 0] += 10.0
    batched._build_all_owner_pos()
    batched._full_recompute()

    pre_c0 = batched.placement[0].clone()
    pre_c1 = batched.placement[1].clone()
    assert not torch.equal(pre_c0, pre_c1), "Setup failed: chains identical"

    batched.swap_chains(0, 1)
    assert torch.equal(batched.placement[0], pre_c1)
    assert torch.equal(batched.placement[1], pre_c0)
    print("  ✓ swap_chains exchanges placement correctly")


def test_step_moves_accept_all_matches_serial():
    """
    Apply the same move on each chain via step_moves with T=inf (accept
    all), then verify each chain's grids match the serial estimator
    that applied the same move via update_macro_pos.
    """
    bench = make_synthetic()
    serial = CostEstimator(bench)
    batched = BatchedCostEstimator(bench, num_chains=3, device="cpu")
    # Prime _last_cost cache.
    batched._last_cost = batched.get_costs()["proxy_cost"].clone()

    # Move macro 2 by +(5, -3) on all chains.
    R = batched.R
    macro_idx = torch.tensor([2] * R, dtype=torch.long)
    new_pos = (bench.macro_positions[2] + torch.tensor([5.0, -3.0])).unsqueeze(0).expand(R, -1).contiguous()
    temps = torch.tensor([1e10] * R)  # accept everything
    rngv = torch.zeros(R)

    out = batched.step_moves(macro_idx, new_pos, temps, rngv, forbid_overlap=False)
    assert out["accepted"].all(), f"expected all chains to accept, got {out['accepted']}"

    # Apply the same move to serial.
    serial.update_macro_pos(2, bench.macro_positions[2] + torch.tensor([5.0, -3.0]))

    # Compare grids per chain.
    for r in range(R):
        d_diff = (batched.density_grid[r] - serial.density_grid).abs().max().item()
        c_diff = (batched.congestion_grid[r] - serial.congestion_grid).abs().max().item()
        h_diff = (batched.net_hpwls[r] - serial.net_hpwls).abs().max().item()
        twd_diff = abs(batched.total_weighted_hpwl[r].item() - serial.total_weighted_hpwl)
        ov_diff = abs(int(batched.overlap_count[r].item()) - int(serial.overlap_count))
        area_diff = abs(batched.total_overlap_area[r].item() - serial.total_overlap_area)
        assert d_diff < 1e-4, f"chain {r}: density diff {d_diff}"
        assert c_diff < 1e-4, f"chain {r}: congestion diff {c_diff}"
        assert h_diff < 1e-3, f"chain {r}: hpwl diff {h_diff}"
        assert twd_diff < 1e-2, f"chain {r}: weighted hpwl diff {twd_diff}"
        assert ov_diff == 0, f"chain {r}: overlap count diff {ov_diff}"
        assert area_diff < 1e-3, f"chain {r}: overlap area diff {area_diff}"
    print(f"  ✓ post-move state matches serial across all {R} chains")


def test_step_moves_reject_keeps_state():
    """With cold T and an uphill move, ALL chains must reject; state must
    be exactly as before."""
    bench = make_synthetic()
    batched = BatchedCostEstimator(bench, num_chains=3, device="cpu")
    batched._last_cost = batched.get_costs()["proxy_cost"].clone()

    # Snapshot pre-move state.
    pre_density = batched.density_grid.clone()
    pre_cong = batched.congestion_grid.clone()
    pre_hpwl = batched.net_hpwls.clone()
    pre_twd = batched.total_weighted_hpwl.clone()
    pre_placement = batched.placement.clone()
    pre_oc = batched.overlap_count.clone()
    pre_oa = batched.total_overlap_area.clone()

    R = batched.R
    # Move macro 0 to the corner — clearly uphill (it pushes wires longer
    # and consolidates density at one corner). Cold T + high rng → reject.
    macro_idx = torch.tensor([0] * R, dtype=torch.long)
    new_pos = torch.tensor([[3.0, 3.0]] * R)
    temps = torch.tensor([1e-8] * R)
    rngv = torch.tensor([0.99] * R)

    out = batched.step_moves(macro_idx, new_pos, temps, rngv, forbid_overlap=False)
    n_rejected = int((~out["accepted"]).sum().item())
    assert n_rejected == R, \
        f"expected all {R} chains to reject this uphill move at T=1e-8; got {n_rejected}"

    # For rejected chains: state must be exactly pre-move.
    for r in range(R):
        assert torch.equal(batched.placement[r], pre_placement[r]), \
            f"chain {r}: placement changed after rejection"
        d_diff = (batched.density_grid[r] - pre_density[r]).abs().max().item()
        c_diff = (batched.congestion_grid[r] - pre_cong[r]).abs().max().item()
        h_diff = (batched.net_hpwls[r] - pre_hpwl[r]).abs().max().item()
        twd_diff = abs(batched.total_weighted_hpwl[r].item() - pre_twd[r].item())
        oc_diff = abs(int(batched.overlap_count[r].item()) - int(pre_oc[r].item()))
        oa_diff = abs(batched.total_overlap_area[r].item() - pre_oa[r].item())
        assert d_diff < 1e-4, f"chain {r}: density drift after reject {d_diff}"
        assert c_diff < 1e-4, f"chain {r}: congestion drift after reject {c_diff}"
        assert h_diff < 1e-4, f"chain {r}: hpwl drift after reject {h_diff}"
        assert twd_diff < 1e-3, f"chain {r}: weighted hpwl drift {twd_diff}"
        assert oc_diff == 0, f"chain {r}: overlap count changed by {oc_diff}"
        assert oa_diff < 1e-3, f"chain {r}: overlap area drift {oa_diff}"
    print(f"  ✓ all {R} rejected chains have state identical to pre-move")


def test_step_moves_mixed_accept_reject():
    """Half chains accept, half reject — confirm each half is correct."""
    bench = make_synthetic()
    batched = BatchedCostEstimator(bench, num_chains=4, device="cpu")
    batched._last_cost = batched.get_costs()["proxy_cost"].clone()

    R = batched.R
    macro_idx = torch.tensor([2] * R, dtype=torch.long)
    new_pos = (bench.macro_positions[2] + torch.tensor([3.0, 2.0])).unsqueeze(0).expand(R, -1).contiguous()
    # Two chains at infinite T (accept), two at 0 T (only accept downhill).
    temps = torch.tensor([1e10, 1e10, 0.0, 0.0])
    rngv = torch.zeros(R)

    pre_placement = batched.placement.clone()
    out = batched.step_moves(macro_idx, new_pos, temps, rngv, forbid_overlap=False)

    # Chains 0, 1 must accept; 2, 3 depend on whether the move is downhill.
    assert out["accepted"][0].item() and out["accepted"][1].item()
    print(f"  ✓ mixed accept/reject ran (accept mask: {out['accepted'].tolist()})")

    # For each chain: if accepted, placement[r, 2] == new_pos[r]; if rejected, unchanged.
    for r in range(R):
        if out["accepted"][r]:
            assert torch.allclose(batched.placement[r, 2], new_pos[r]), \
                f"chain {r}: accepted but placement wasn't updated"
        else:
            assert torch.equal(batched.placement[r], pre_placement[r]), \
                f"chain {r}: rejected but placement changed"
    print(f"  ✓ per-chain placement matches accept/reject decisions")


def test_multiple_step_moves_match_serial():
    """Run a sequence of moves through step_moves (all accept) and
    serial.update_macro_pos. Final states should match."""
    bench = make_synthetic()
    serial = CostEstimator(bench)
    batched = BatchedCostEstimator(bench, num_chains=2, device="cpu")
    batched._last_cost = batched.get_costs()["proxy_cost"].clone()

    moves = [
        (2, [5.0, -3.0]),
        (5, [-2.0, 4.0]),
        (1, [0.5, 0.5]),
        (3, [-1.0, -2.0]),
    ]
    R = batched.R
    temps = torch.tensor([1e10] * R)
    rngv = torch.zeros(R)

    for m_idx, delta in moves:
        target = bench.macro_positions[m_idx] + torch.tensor(delta)
        # Apply to serial.
        serial.update_macro_pos(m_idx, target)
        # Apply to batched.
        macro_idx = torch.tensor([m_idx] * R, dtype=torch.long)
        new_pos = target.unsqueeze(0).expand(R, -1).contiguous()
        out = batched.step_moves(macro_idx, new_pos, temps, rngv, forbid_overlap=False)
        assert out["accepted"].all()

    # Final state comparison.
    for r in range(R):
        d_diff = (batched.density_grid[r] - serial.density_grid).abs().max().item()
        c_diff = (batched.congestion_grid[r] - serial.congestion_grid).abs().max().item()
        twd_diff = abs(batched.total_weighted_hpwl[r].item() - serial.total_weighted_hpwl)
        assert d_diff < 1e-3, f"chain {r}: density drift after 4 moves {d_diff}"
        assert c_diff < 1e-3, f"chain {r}: congestion drift after 4 moves {c_diff}"
        assert twd_diff < 1e-1, f"chain {r}: HPWL drift {twd_diff}"
    print(f"  ✓ 4-move sequence: batched state matches serial within tolerance")


if __name__ == "__main__":
    print("== test_density_grid_matches_serial ==")
    test_density_grid_matches_serial()
    print("\n== test_congestion_grid_matches_serial ==")
    test_congestion_grid_matches_serial()
    print("\n== test_hpwl_matches_serial ==")
    test_hpwl_matches_serial()
    print("\n== test_overlap_matches_serial ==")
    test_overlap_matches_serial()
    print("\n== test_overlap_detects_overlaps ==")
    test_overlap_detects_overlaps()
    print("\n== test_proxy_cost_matches_serial ==")
    test_proxy_cost_matches_serial()
    print("\n== test_distinct_chains_after_swap ==")
    test_distinct_chains_after_swap()
    print("\n== test_step_moves_accept_all_matches_serial ==")
    test_step_moves_accept_all_matches_serial()
    print("\n== test_step_moves_reject_keeps_state ==")
    test_step_moves_reject_keeps_state()
    print("\n== test_step_moves_mixed_accept_reject ==")
    test_step_moves_mixed_accept_reject()
    print("\n== test_multiple_step_moves_match_serial ==")
    test_multiple_step_moves_match_serial()
    print("\nAll batched estimator tests passed.")
