import torch
import math
from typing import Dict, Optional
from macro_place.benchmark import Benchmark

def calculate_hpwl(placement: torch.Tensor, benchmark: Benchmark) -> float:
    """
    Calculate the exact Half-Perimeter Wirelength (HPWL) using pin-level connectivity.
    Matches the normalization logic in PlacementCost.get_cost().
    
    This function uses the net_weights provided in the Benchmark object for both
    the weighted summation and the normalization factor.
    """
    # Combine macro positions and port positions for easier indexing
    all_owner_pos = torch.cat([placement, benchmark.port_positions], dim=0)
    
    total_hpwl = 0.0
    
    for net_idx, net_pins in enumerate(benchmark.net_pin_nodes):
        owner_idx = net_pins[:, 0]
        pin_in_owner_idx = net_pins[:, 1]
        
        # Get base positions [N, 2]
        base_pos = all_owner_pos[owner_idx]
        
        # Get offsets [N, 2]
        offsets = torch.zeros_like(base_pos)
        
        # Vectorized offset lookup for hard macros
        hard_macro_mask = owner_idx < benchmark.num_hard_macros
        if hard_macro_mask.any():
            for i in torch.where(hard_macro_mask)[0]:
                o_idx = owner_idx[i].item()
                p_idx = pin_in_owner_idx[i].item()
                offsets[i] = benchmark.macro_pin_offsets[o_idx][p_idx]
        
        # Absolute pin positions
        abs_pins = base_pos + offsets
        
        # Net Bounding Box
        x_min, x_max = abs_pins[:, 0].min(), abs_pins[:, 0].max()
        y_min, y_max = abs_pins[:, 1].min(), abs_pins[:, 1].max()
        
        hpwl = (x_max - x_min) + (y_max - y_min)
        
        # Apply weight from benchmark
        weight = benchmark.net_weights[net_idx].item()
        total_hpwl += (hpwl.item() * weight)

    # Normalization from PlacementCost.get_cost(): sum(weighted_hpwl) / ((W+H) * plc.net_cnt)
    # plc.net_cnt is equivalent to the sum of weights in the benchmark object.
    total_connections = benchmark.net_weights.sum().item()
    norm_factor = (benchmark.canvas_width + benchmark.canvas_height) * total_connections
    
    if norm_factor == 0:
        return 0.0
    return total_hpwl / norm_factor

def calculate_density_cost(placement: torch.Tensor, benchmark: Benchmark) -> float:
    """
    Calculate density cost using a grid-based approach.
    Matches the top-10% logic in PlacementCost.get_density_cost().
    """
    rows, cols = benchmark.grid_rows, benchmark.grid_cols
    canvas_w, canvas_h = benchmark.canvas_width, benchmark.canvas_height
    
    grid = torch.zeros((rows, cols), device=placement.device)
    cell_w = canvas_w / cols
    cell_h = canvas_h / rows
    cell_area = cell_w * cell_h
    
    # Calculate overlap area for each macro in each grid cell
    for i in range(benchmark.num_macros):
        w, h = benchmark.macro_sizes[i]
        x, y = placement[i]
        
        m_lx, m_ux = x - w/2, x + w/2
        m_by, m_ty = y - h/2, y + h/2
        
        c_start = max(0, int(m_lx // cell_w))
        c_end = min(cols - 1, int(m_ux // cell_w))
        r_start = max(0, int(m_by // cell_h))
        r_end = min(rows - 1, int(m_ty // cell_h))
        
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                inter_w = min(m_ux, (c+1)*cell_w) - max(m_lx, c*cell_w)
                inter_y = min(m_ty, (r+1)*cell_h) - max(m_by, r*cell_h)
                if inter_w > 0 and inter_y > 0:
                    grid[r, c] += (inter_w * inter_y)
    
    density_map = grid / cell_area
    top_k = max(1, int(rows * cols * 0.1))
    top_values, _ = torch.topk(density_map.view(-1), top_k)
    return 0.5 * top_values.mean().item()

def calculate_congestion_rudy(placement: torch.Tensor, benchmark: Benchmark) -> float:
    """
    Calculate an approximate congestion cost using the RUDY model.
    Matches the top-5% logic (abu 0.05) in PlacementCost.get_congestion_cost().
    """
    rows, cols = benchmark.grid_rows, benchmark.grid_cols
    canvas_w, canvas_h = benchmark.canvas_width, benchmark.canvas_height
    
    cell_w = canvas_w / cols
    cell_h = canvas_h / rows
    
    congestion_grid = torch.zeros((rows, cols), device=placement.device)
    all_owner_pos = torch.cat([placement, benchmark.port_positions], dim=0)

    for net_pins in benchmark.net_pin_nodes:
        owner_idx = net_pins[:, 0]
        net_pos = all_owner_pos[owner_idx]
        
        lx, ux = net_pos[:, 0].min().item(), net_pos[:, 0].max().item()
        by, ty = net_pos[:, 1].min().item(), net_pos[:, 1].max().item()
        
        w = ux - lx
        h = ty - by
        
        if w == 0 and h == 0:
            continue
            
        demand_density = (w + h) / (w * h + 1e-6)
        
        c_start = max(0, int(lx // cell_w))
        c_end = min(cols - 1, int(ux // cell_w))
        r_start = max(0, int(by // cell_h))
        r_end = min(rows - 1, int(ty // cell_h))
        
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                inter_w = min(ux, (c+1)*cell_w) - max(lx, c*cell_w)
                inter_h = min(ty, (r+1)*cell_h) - max(by, r*cell_h)
                if inter_w > 0 and inter_h > 0:
                    cell_demand = demand_density * (inter_w * inter_h)
                    avg_cap = (cell_w * benchmark.vroutes_per_micron + cell_h * benchmark.hroutes_per_micron) / 2.0
                    congestion_grid[r, c] += cell_demand / (avg_cap + 1e-6)

    top_k = max(1, int(rows * cols * 0.05))
    top_values, _ = torch.topk(congestion_grid.view(-1), top_k)
    return top_values.mean().item()

def compute_proxy_cost_benchmark(
    placement: torch.Tensor, 
    benchmark: Benchmark, 
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute total proxy cost using only Benchmark data.
    """
    if weights is None:
        weights = {"wirelength": 1.0, "density": 0.5, "congestion": 0.5}
        
    wl_cost = calculate_hpwl(placement, benchmark)
    den_cost = calculate_density_cost(placement, benchmark)
    cong_cost = calculate_congestion_rudy(placement, benchmark)
    
    proxy = (
        weights["wirelength"] * wl_cost +
        weights["density"] * den_cost +
        weights["congestion"] * cong_cost
    )
    
    return {
        "proxy_cost": proxy,
        "wirelength_cost": wl_cost,
        "density_cost": den_cost,
        "congestion_cost": cong_cost
    }

if __name__ == "__main__":
    import os
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    
    benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    
    if os.path.exists(benchmark_dir):
        benchmark, plc = load_benchmark_from_dir(benchmark_dir)
        
        # NOTE: Since the local loader.py is original/broken, benchmark.net_weights are all 1.0.
        # To verify the PyTorch logic matches Ground Truth, we manually fix the weights 
        # from the PLC object here (simulating what the deployment Benchmark would look like).
        print("Syncing net weights from PLC to Benchmark for validation...")
        for i, (driver, sinks) in enumerate(plc.nets.items()):
            driver_idx = plc.mod_name_to_indices.get(driver)
            if driver_idx is not None:
                benchmark.net_weights[i] = float(plc.modules_w_pins[driver_idx].get_weight())
        
        print("\n--- Comparison ---")
        gt_metrics = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
        pt_metrics = compute_proxy_cost_benchmark(benchmark.macro_positions, benchmark)
        
        for k in pt_metrics.keys():
            gt_val = gt_metrics.get(k, 0.0)
            pt_val = pt_metrics[k]
            diff = pt_val - gt_val
            pct = (diff / gt_val * 100) if gt_val != 0 else 0
            print(f"{k:20}: GT={gt_val:.6f}, PT={pt_val:.6f}, Delta={diff:+.6f} ({pct:+.2f}%)")
