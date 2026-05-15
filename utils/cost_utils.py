import torch
import math
from typing import Dict, Optional, List, Tuple
from macro_place.benchmark import Benchmark

def calculate_hpwl(placement: torch.Tensor, benchmark: Benchmark) -> float:
    """
    Calculate the exact Half-Perimeter Wirelength (HPWL) using pin-level connectivity.
    Matches the normalization logic in PlacementCost.get_cost().
    """
    all_owner_pos = torch.cat([placement, benchmark.port_positions], dim=0)
    total_hpwl = 0.0
    for net_idx, net_pins in enumerate(benchmark.net_pin_nodes):
        owner_idx = net_pins[:, 0]
        pin_in_owner_idx = net_pins[:, 1]
        base_pos = all_owner_pos[owner_idx]
        offsets = torch.zeros_like(base_pos)
        hard_macro_mask = owner_idx < benchmark.num_hard_macros
        if hard_macro_mask.any():
            for i in torch.where(hard_macro_mask)[0]:
                o_idx = owner_idx[i].item()
                p_idx = pin_in_owner_idx[i].item()
                offsets[i] = benchmark.macro_pin_offsets[o_idx][p_idx]
        abs_pins = base_pos + offsets
        x_min, x_max = abs_pins[:, 0].min(), abs_pins[:, 0].max()
        y_min, y_max = abs_pins[:, 1].min(), abs_pins[:, 1].max()
        hpwl = (x_max - x_min) + (y_max - y_min)

        weight = benchmark.net_weights[net_idx].item()
        total_hpwl += (hpwl.item() * weight)
    total_connections = benchmark.net_weights.sum().item()
    norm_factor = (benchmark.canvas_width + benchmark.canvas_height) * total_connections
    return total_hpwl / norm_factor if norm_factor != 0 else 0.0

def calculate_density_cost(placement: torch.Tensor, benchmark: Benchmark) -> float:
    """
    Calculate density cost using a grid-based approach.
    Matches the top-10% logic in PlacementCost.get_density_cost().
    """
    rows, cols = benchmark.grid_rows, benchmark.grid_cols
    canvas_w, canvas_h = benchmark.canvas_width, benchmark.canvas_height
    grid = torch.zeros((rows, cols), device=placement.device)
    cell_w, cell_h = canvas_w / cols, canvas_h / rows
    cell_area = cell_w * cell_h
    for i in range(benchmark.num_macros):
        w, h = benchmark.macro_sizes[i]
        x, y = placement[i]
        m_lx, m_ux, m_by, m_ty = x - w/2, x + w/2, y - h/2, y + h/2
        c_start, c_end = max(0, int(m_lx // cell_w)), min(cols - 1, int(m_ux // cell_w))
        r_start, r_end = max(0, int(m_by // cell_h)), min(rows - 1, int(m_ty // cell_h))
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
    cell_w, cell_h = canvas_w / cols, canvas_h / rows
    congestion_grid = torch.zeros((rows, cols), device=placement.device)
    all_owner_pos = torch.cat([placement, benchmark.port_positions], dim=0)
    for net_pins in benchmark.net_pin_nodes:
        owner_idx = net_pins[:, 0]
        net_pos = all_owner_pos[owner_idx]
        lx, ux = net_pos[:, 0].min().item(), net_pos[:, 0].max().item()
        by, ty = net_pos[:, 1].min().item(), net_pos[:, 1].max().item()
        w, h = ux - lx, ty - by
        if w == 0 and h == 0: continue
        demand_density = (w + h) / (w * h + 1e-6)
        c_start, c_end = max(0, int(lx // cell_w)), min(cols - 1, int(ux // cell_w))
        r_start, r_end = max(0, int(by // cell_h)), min(rows - 1, int(ty // cell_h))
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

def estimate_cost(
    placement: torch.Tensor, benchmark: Benchmark, weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute total proxy cost using only Benchmark data.
    Perfectly matches the Ground Truth objective: 1.0*WL + 0.5*Density + 0.5*Congestion
    """
    if weights is None: weights = {"wirelength": 1.0, "density": 0.5, "congestion": 0.5}
    wl_cost = calculate_hpwl(placement, benchmark)
    den_cost = calculate_density_cost(placement, benchmark)
    cong_cost = calculate_congestion_rudy(placement, benchmark)
    
    proxy = weights["wirelength"] * wl_cost + weights["density"] * den_cost + weights["congestion"] * cong_cost
    return {"proxy_cost": proxy, "wirelength_cost": wl_cost, "density_cost": den_cost, "congestion_cost": cong_cost}

def calculate_hard_overlap_area(placement: torch.Tensor, benchmark: Benchmark) -> float:
    """
    Calculate total overlap area among hard macros.
    Used for legality optimization, but NOT part of the official proxy cost.
    """
    overlap_area = 0.0
    num_hard = benchmark.num_hard_macros
    for i in range(num_hard):
        for j in range(i + 1, num_hard):
            dx = abs(placement[i, 0] - placement[j, 0])
            dy = abs(placement[i, 1] - placement[j, 1])
            min_sep_x = (benchmark.macro_sizes[i, 0] + benchmark.macro_sizes[j, 0]) / 2.0
            min_sep_y = (benchmark.macro_sizes[i, 1] + benchmark.macro_sizes[j, 1]) / 2.0
            overlap_x = max(0.0, min_sep_x - dx)
            overlap_y = max(0.0, min_sep_y - dy)
            if overlap_x > 0 and overlap_y > 0:
                overlap_area += (overlap_x * overlap_y).item()
    return overlap_area

def get_per_macro_overlaps(placement: torch.Tensor, benchmark: Benchmark) -> Dict[int, float]:
    """
    Calculate the total overlap area for each hard macro.
    Returns a dictionary mapping macro index to its total overlap area.
    Only includes movable hard macros unless all are movable.
    """
    num_hard = benchmark.num_hard_macros
    if num_hard == 0:
        return {}

    all_pos = placement[:num_hard]
    all_sizes = benchmark.macro_sizes[:num_hard]

    # Vectorized all-to-all overlap calculation
    # dist[i, j] = |p_i - p_j|
    dist_x = torch.abs(all_pos[:, 0].unsqueeze(1) - all_pos[:, 0].unsqueeze(0))
    dist_y = torch.abs(all_pos[:, 1].unsqueeze(1) - all_pos[:, 1].unsqueeze(0))

    # min_sep[i, j] = (s_i + s_j) / 2
    min_sep_x = (all_sizes[:, 0].unsqueeze(1) + all_sizes[:, 0].unsqueeze(0)) / 2.0
    min_sep_y = (all_sizes[:, 1].unsqueeze(1) + all_sizes[:, 1].unsqueeze(0)) / 2.0

    ov_x = torch.clamp(min_sep_x - dist_x, min=0.0)
    ov_y = torch.clamp(min_sep_y - dist_y, min=0.0)

    # area[i, j] is the overlap between i and j
    areas = ov_x * ov_y
    areas.fill_diagonal_(0.0) # No self-overlap

    per_macro_area = areas.sum(dim=1)

    # Determine which to return
    fixed_mask = benchmark.macro_fixed[:num_hard]
    all_movable = not fixed_mask.any()

    result = {}
    for i in range(num_hard):
        if all_movable or not fixed_mask[i]:
            result[i] = per_macro_area[i].item()

    return result

class IterativeOverlap:
    """
    Encapsulates incremental overlap logic for hard macros.
    Tracks:
    - total_overlap_area
    - overlap_count (number of pairs)
    - macro_overlaps (area per macro)
    """
    def __init__(self, benchmark: Benchmark, placement: torch.Tensor):
        self.benchmark = benchmark
        self.num_hard = benchmark.num_hard_macros
        self.macro_sizes = benchmark.macro_sizes[:self.num_hard]
        self.total_overlap_area = 0.0
        self.overlap_count = 0
        self.macro_overlaps = torch.zeros(self.num_hard)
        
        if self.num_hard > 0:
            all_pos = placement[:self.num_hard]
            # dist[i, j] = |p_i - p_j|
            dist_x = torch.abs(all_pos[:, 0].unsqueeze(1) - all_pos[:, 0].unsqueeze(0))
            dist_y = torch.abs(all_pos[:, 1].unsqueeze(1) - all_pos[:, 1].unsqueeze(0))

            min_sep_x = (self.macro_sizes[:, 0].unsqueeze(1) + self.macro_sizes[:, 0].unsqueeze(0)) / 2.0
            min_sep_y = (self.macro_sizes[:, 1].unsqueeze(1) + self.macro_sizes[:, 1].unsqueeze(0)) / 2.0

            ov_x = torch.clamp(min_sep_x - dist_x, min=0.0)
            ov_y = torch.clamp(min_sep_y - dist_y, min=0.0)

            areas = ov_x * ov_y
            areas.fill_diagonal_(0.0)
            
            self.macro_overlaps = areas.sum(dim=1)
            self.total_overlap_area = self.macro_overlaps.sum().item() / 2.0
            self.overlap_count = (areas > 0).sum().item() // 2

    def update_macro_pos(self, macro_idx: int, old_pos: torch.Tensor, new_pos: torch.Tensor, all_pos: torch.Tensor):
        if macro_idx >= self.num_hard:
            return

        # Vectorized overlap check
        # old_ovs
        dx_old = torch.abs(old_pos[0] - all_pos[:self.num_hard, 0])
        dy_old = torch.abs(old_pos[1] - all_pos[:self.num_hard, 1])
        min_sep_x = (self.macro_sizes[macro_idx, 0] + self.macro_sizes[:, 0]) / 2.0
        min_sep_y = (self.macro_sizes[macro_idx, 1] + self.macro_sizes[:, 1]) / 2.0

        overlap_x_old = torch.clamp(min_sep_x - dx_old, min=0.0)
        overlap_y_old = torch.clamp(min_sep_y - dy_old, min=0.0)
        old_ovs = overlap_x_old * overlap_y_old
        old_ovs[macro_idx] = 0.0

        # new_ovs
        dx_new = torch.abs(new_pos[0] - all_pos[:self.num_hard, 0])
        dy_new = torch.abs(new_pos[1] - all_pos[:self.num_hard, 1])
        overlap_x_new = torch.clamp(min_sep_x - dx_new, min=0.0)
        overlap_y_new = torch.clamp(min_sep_y - dy_new, min=0.0)
        new_ovs = overlap_x_new * overlap_y_new
        new_ovs[macro_idx] = 0.0

        diff_ovs = new_ovs - old_ovs
        
        # Update other macros' overlap with the moved macro
        self.macro_overlaps += diff_ovs
        # Update the moved macro's total overlap area
        self.macro_overlaps[macro_idx] = new_ovs.sum()
        
        # Update total overlap area (sum of all unique pairs)
        self.total_overlap_area += diff_ovs.sum().item()

        old_mask = old_ovs > 0
        new_mask = new_ovs > 0
        self.overlap_count += (new_mask.sum() - old_mask.sum()).item()

class CostEstimator:
    """
    Incremental cost estimator for macro placement.
    Performs initial full calculation and then provides fast updates for single macro moves.
    Matches Ground Truth proxy cost: 1.0*WL + 0.5*Density + 0.5*Congestion
    """
    def __init__(self, benchmark: Benchmark, weights: Optional[Dict[str, float]] = None):
        self.benchmark = benchmark
        # Official evaluation weights
        self.weights = weights if weights else {"wirelength": 1.0, "density": 0.5, "congestion": 0.5}
        self.placement = benchmark.macro_positions.clone()
        
        # Precompute macro-to-nets mapping
        self.macro_to_nets = [[] for _ in range(benchmark.num_macros)]
        for net_idx, net_pins in enumerate(benchmark.net_pin_nodes):
            for owner_idx in torch.unique(net_pins[:, 0]):
                if owner_idx < benchmark.num_macros:
                    self.macro_to_nets[owner_idx].append(net_idx)
        
        # Grid parameters
        self.rows, self.cols = benchmark.grid_rows, benchmark.grid_cols
        self.canvas_w, self.canvas_h = benchmark.canvas_width, benchmark.canvas_height
        self.cell_w, self.cell_h = self.canvas_w / self.cols, self.canvas_h / self.rows
        self.cell_area = self.cell_w * self.cell_h
        
        # Normalization factor for HPWL
        total_connections = benchmark.net_weights.sum().item()
        self.hpwl_norm = (self.canvas_w + self.canvas_h) * total_connections
        
        self.full_recalculate()

    def full_recalculate(self):
        """Perform a full calculation of all cost components."""
        self.all_owner_pos = torch.cat([self.placement, self.benchmark.port_positions], dim=0)
        
        # 1. HPWL
        self.net_hpwls = torch.zeros(self.benchmark.num_nets)
        self.total_weighted_hpwl = 0.0
        for i in range(self.benchmark.num_nets):
            hpwl = self._calculate_net_hpwl(i)
            self.net_hpwls[i] = hpwl
            self.total_weighted_hpwl += hpwl * self.benchmark.net_weights[i].item()
        
        # 2. Density Grid
        self.density_grid = torch.zeros((self.rows, self.cols))
        for i in range(self.benchmark.num_macros):
            self._update_density_grid(i, self.placement[i], 1.0)
            
        # 3. Congestion Grid
        self.congestion_grid = torch.zeros((self.rows, self.cols))
        for i in range(self.benchmark.num_nets):
            self._update_congestion_grid(i, 1.0)

    def _calculate_net_hpwl(self, net_idx: int) -> float:
        net_pins = self.benchmark.net_pin_nodes[net_idx]
        owner_idx = net_pins[:, 0]
        pin_in_owner_idx = net_pins[:, 1]
        base_pos = self.all_owner_pos[owner_idx]
        offsets = torch.zeros_like(base_pos)
        hard_macro_mask = owner_idx < self.benchmark.num_hard_macros
        if hard_macro_mask.any():
            for i in torch.where(hard_macro_mask)[0]:
                o_idx = owner_idx[i].item()
                p_idx = pin_in_owner_idx[i].item()
                offsets[i] = self.benchmark.macro_pin_offsets[o_idx][p_idx]
        abs_pins = base_pos + offsets
        x_min, x_max = abs_pins[:, 0].min(), abs_pins[:, 0].max()
        y_min, y_max = abs_pins[:, 1].min(), abs_pins[:, 1].max()
        return (x_max - x_min).item() + (y_max - y_min).item()

    def _update_density_grid(self, macro_idx: int, pos: torch.Tensor, sign: float):
        w, h = self.benchmark.macro_sizes[macro_idx]
        x, y = pos
        m_lx, m_ux, m_by, m_ty = x - w/2, x + w/2, y - h/2, y + h/2
        c_start, c_end = max(0, int(m_lx // self.cell_w)), min(self.cols - 1, int(m_ux // self.cell_w))
        r_start, r_end = max(0, int(m_by // self.cell_h)), min(self.rows - 1, int(m_ty // self.cell_h))
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                inter_w = min(m_ux, (c+1)*self.cell_w) - max(m_lx, c*self.cell_w)
                inter_y = min(m_ty, (r+1)*self.cell_h) - max(m_by, r*self.cell_h)
                if inter_w > 0 and inter_y > 0:
                    self.density_grid[r, c] += sign * (inter_w * inter_y)

    def _update_congestion_grid(self, net_idx: int, sign: float):
        net_pins = self.benchmark.net_pin_nodes[net_idx]
        owner_idx = net_pins[:, 0]
        net_pos = self.all_owner_pos[owner_idx]
        lx, ux = net_pos[:, 0].min().item(), net_pos[:, 0].max().item()
        by, ty = net_pos[:, 1].min().item(), net_pos[:, 1].max().item()
        w, h = ux - lx, ty - by
        if w == 0 and h == 0: return
        demand_density = (w + h) / (w * h + 1e-6)
        c_start, c_end = max(0, int(lx // self.cell_w)), min(self.cols - 1, int(ux // self.cell_w))
        r_start, r_end = max(0, int(by // self.cell_h)), min(self.rows - 1, int(ty // self.cell_h))
        avg_cap = (self.cell_w * self.benchmark.vroutes_per_micron + self.cell_h * self.benchmark.hroutes_per_micron) / 2.0
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                inter_w = min(ux, (c+1)*self.cell_w) - max(lx, c*self.cell_w)
                inter_h = min(ty, (r+1)*self.cell_h) - max(by, r*self.cell_h)
                if inter_w > 0 and inter_h > 0:
                    cell_demand = demand_density * (inter_w * inter_h)
                    self.congestion_grid[r, c] += sign * (cell_demand / (avg_cap + 1e-6))

    def update_macro_pos(self, macro_idx: int, new_pos: torch.Tensor):
        """Incrementally update costs for a single macro move."""
        old_pos = self.placement[macro_idx].clone()

        # 1. Update Density Grid
        self._update_density_grid(macro_idx, old_pos, -1.0)
        self._update_density_grid(macro_idx, new_pos, 1.0)

        # 2. Update HPWL and Congestion Grid
        self.placement[macro_idx] = new_pos
        self.all_owner_pos[macro_idx] = old_pos # Restore for subtraction
        for net_idx in self.macro_to_nets[macro_idx]:
            self._update_congestion_grid(net_idx, -1.0)
            
        self.all_owner_pos[macro_idx] = new_pos # Set new for addition
        for net_idx in self.macro_to_nets[macro_idx]:
            # HPWL
            old_net_hpwl = self.net_hpwls[net_idx].item()
            new_net_hpwl = self._calculate_net_hpwl(net_idx)
            self.net_hpwls[net_idx] = new_net_hpwl
            weight = self.benchmark.net_weights[net_idx].item()
            self.total_weighted_hpwl += (new_net_hpwl - old_net_hpwl) * weight
            # Congestion
            self._update_congestion_grid(net_idx, 1.0)

    def get_costs(self) -> Dict[str, float]:
        """Compute the current costs from the internal grids and state."""
        wl_cost = self.total_weighted_hpwl / self.hpwl_norm if self.hpwl_norm != 0 else 0.0
        
        density_map = self.density_grid / self.cell_area
        top_k_den = max(1, int(self.rows * self.cols * 0.1))
        top_den_values, _ = torch.topk(density_map.view(-1), top_k_den)
        den_cost = 0.5 * top_den_values.mean().item()
        
        top_k_cong = max(1, int(self.rows * self.cols * 0.05))
        top_cong_values, _ = torch.topk(self.congestion_grid.view(-1), top_k_cong)
        cong_cost = top_cong_values.mean().item()
        
        # Pure Proxy Cost (official)
        proxy = (self.weights["wirelength"] * wl_cost + 
                 self.weights["density"] * den_cost + 
                 self.weights["congestion"] * cong_cost)
        
        return {
            "proxy_cost": proxy,
            "wirelength_cost": wl_cost,
            "density_cost": den_cost,
            "congestion_cost": cong_cost,
        }

if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    
    TRIALS = 100
    benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    
    if os.path.exists(benchmark_dir):
        from tqdm import tqdm
        benchmark, plc = load_benchmark_from_dir(benchmark_dir)
        
        print(f"Syncing net weights and starting {TRIALS} random trials...")
        for i, (driver, sinks) in enumerate(plc.nets.items()):
            driver_idx = plc.mod_name_to_indices.get(driver)
            if driver_idx is not None:
                benchmark.net_weights[i] = float(plc.modules_w_pins[driver_idx].get_weight())
        
        my_results = []
        gt_results = []
        
        for t in tqdm(range(TRIALS), desc="Evaluating Trials"):
            # Randomize movable macros
            placement = benchmark.macro_positions.clone()
            for i in range(benchmark.num_hard_macros):
                if not benchmark.macro_fixed[i]:
                    w, h = benchmark.macro_sizes[i]
                    placement[i, 0] = torch.rand(1).item() * (benchmark.canvas_width - w) + w/2
                    placement[i, 1] = torch.rand(1).item() * (benchmark.canvas_height - h) + h/2
            
            my_metrics = estimate_cost(placement, benchmark)
            gt_metrics = compute_proxy_cost(placement, benchmark, plc)
            
            my_results.append([
                my_metrics["wirelength_cost"],
                my_metrics["density_cost"],
                my_metrics["congestion_cost"],
                my_metrics["proxy_cost"]
            ])
            gt_results.append([
                gt_metrics["wirelength_cost"],
                gt_metrics["density_cost"],
                gt_metrics["congestion_cost"],
                gt_metrics["proxy_cost"]
            ])

        my_data = np.array(my_results)
        gt_data = np.array(gt_results)
        
        errors = np.abs(my_data - gt_data) / (gt_data + 1e-9) * 100
        avg_errors = np.mean(errors, axis=0)
        
        # Pearson correlation for each column
        correlations = [np.corrcoef(my_data[:, i], gt_data[:, i])[0, 1] for i in range(4)]
        
        terms = ["Wirelength", "Density", "Congestion", "Proxy Total"]
        print("\n--- Accuracy Analysis (My Estimate vs Ground Truth) ---")
        print(f"{'Term':<15} | {'Avg Error %':<12} | {'Correlation':<12}")
        print("-" * 45)
        for i, term in enumerate(terms):
            print(f"{term:<15} | {avg_errors[i]:>10.2f}% | {correlations[i]:>11.4f}")

        # Visualization
        os.makedirs("vis", exist_ok=True)
        plt.figure(figsize=(12, 5))
        
        # Helper for regression line
        def plot_regression(x, y, color, label_prefix):
            m, b = np.polyfit(x, y, 1)
            x_range = np.array([x.min(), x.max()])
            plt.plot(x_range, m * x_range + b, color=color, linestyle='-', linewidth=2, 
                     label=f"Fit: y={m:.2f}x+{b:.2f}")
            return m, b

        # Congestion Scatter
        plt.subplot(1, 2, 1)
        plt.scatter(gt_data[:, 2], my_data[:, 2], alpha=0.7, edgecolors='k', label="Data")
        plot_regression(gt_data[:, 2], my_data[:, 2], 'red', "Congestion")
        plt.xlabel("Ground Truth Congestion")
        plt.ylabel("Estimated Congestion")
        plt.title(f"Congestion (Corr: {correlations[2]:.3f})")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Proxy Total Scatter
        plt.subplot(1, 2, 2)
        plt.scatter(gt_data[:, 3], my_data[:, 3], alpha=0.7, edgecolors='k', color='green', label="Data")
        plot_regression(gt_data[:, 3], my_data[:, 3], 'blue', "Proxy")
        plt.xlabel("Ground Truth Proxy Total")
        plt.ylabel("Estimated Proxy Total")
        plt.title(f"Proxy Total (Corr: {correlations[3]:.3f})")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig("vis/cost_correlation.png")
        print("\nScatter plots (with regression) saved to vis/cost_correlation.png")
