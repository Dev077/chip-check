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

class CostEstimator:
    """
    Incremental cost estimator for macro placement.
    Performs initial full calculation and then provides fast updates for single macro moves.
    Matches Ground Truth proxy cost + extra legality metrics.
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
            
        # 4. Overlaps (HARD macros only)
        self.total_overlap_area = 0.0
        self.overlap_count = 0
        num_hard = self.benchmark.num_hard_macros
        for i in range(num_hard):
            for j in range(i + 1, num_hard):
                ov = self._calculate_pair_overlap(i, j, self.placement[i], self.placement[j])
                if ov > 0:
                    self.total_overlap_area += ov
                    self.overlap_count += 1

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
        # Vectorized: the per-cell overlap with a macro's footprint is
        # separable into a column-only width and a row-only height, so we
        # compute those as 1-D tensors and add their outer product to the
        # affected rectangle in one go. Mathematically identical to the
        # original double-loop; verified against ground truth in __main__.
        w, h = self.benchmark.macro_sizes[macro_idx]
        x, y = pos
        m_lx = float(x - w / 2)
        m_ux = float(x + w / 2)
        m_by = float(y - h / 2)
        m_ty = float(y + h / 2)

        c_start = max(0, int(m_lx // self.cell_w))
        c_end = min(self.cols - 1, int(m_ux // self.cell_w))
        r_start = max(0, int(m_by // self.cell_h))
        r_end = min(self.rows - 1, int(m_ty // self.cell_h))
        if c_end < c_start or r_end < r_start:
            return

        # Column overlaps: inter_w[c] = min(m_ux, (c+1)*cell_w) - max(m_lx, c*cell_w)
        cols = torch.arange(c_start, c_end + 1, device=self.density_grid.device, dtype=torch.float32)
        cell_left = cols * self.cell_w
        cell_right = cell_left + self.cell_w
        inter_w = torch.clamp(torch.minimum(cell_right, torch.tensor(m_ux)) -
                              torch.maximum(cell_left, torch.tensor(m_lx)), min=0.0)

        # Row overlaps: inter_h[r] = min(m_ty, (r+1)*cell_h) - max(m_by, r*cell_h)
        rows = torch.arange(r_start, r_end + 1, device=self.density_grid.device, dtype=torch.float32)
        cell_bot = rows * self.cell_h
        cell_top = cell_bot + self.cell_h
        inter_h = torch.clamp(torch.minimum(cell_top, torch.tensor(m_ty)) -
                              torch.maximum(cell_bot, torch.tensor(m_by)), min=0.0)

        # Outer product -> contribution rectangle.
        contrib = sign * torch.outer(inter_h, inter_w)
        self.density_grid[r_start:r_end + 1, c_start:c_end + 1] += contrib

    def _update_congestion_grid(self, net_idx: int, sign: float):
        # Same separable-outer-product idea as _update_density_grid, plus
        # the constant per-cell scaling factor (demand_density / avg_cap).
        net_pins = self.benchmark.net_pin_nodes[net_idx]
        owner_idx = net_pins[:, 0]
        net_pos = self.all_owner_pos[owner_idx]
        lx = net_pos[:, 0].min().item()
        ux = net_pos[:, 0].max().item()
        by = net_pos[:, 1].min().item()
        ty = net_pos[:, 1].max().item()
        w = ux - lx
        h = ty - by
        if w == 0 and h == 0:
            return
        demand_density = (w + h) / (w * h + 1e-6)

        c_start = max(0, int(lx // self.cell_w))
        c_end = min(self.cols - 1, int(ux // self.cell_w))
        r_start = max(0, int(by // self.cell_h))
        r_end = min(self.rows - 1, int(ty // self.cell_h))
        if c_end < c_start or r_end < r_start:
            return

        avg_cap = (self.cell_w * self.benchmark.vroutes_per_micron +
                   self.cell_h * self.benchmark.hroutes_per_micron) / 2.0

        cols = torch.arange(c_start, c_end + 1, device=self.congestion_grid.device, dtype=torch.float32)
        cell_left = cols * self.cell_w
        cell_right = cell_left + self.cell_w
        inter_w = torch.clamp(torch.minimum(cell_right, torch.tensor(ux)) -
                              torch.maximum(cell_left, torch.tensor(lx)), min=0.0)

        rows = torch.arange(r_start, r_end + 1, device=self.congestion_grid.device, dtype=torch.float32)
        cell_bot = rows * self.cell_h
        cell_top = cell_bot + self.cell_h
        inter_h = torch.clamp(torch.minimum(cell_top, torch.tensor(ty)) -
                              torch.maximum(cell_bot, torch.tensor(by)), min=0.0)

        scale = sign * demand_density / (avg_cap + 1e-6)
        contrib = scale * torch.outer(inter_h, inter_w)
        self.congestion_grid[r_start:r_end + 1, c_start:c_end + 1] += contrib

    def _calculate_pair_overlap(self, i: int, j: int, pos_i: torch.Tensor, pos_j: torch.Tensor) -> float:
        dx = abs(pos_i[0] - pos_j[0])
        dy = abs(pos_i[1] - pos_j[1])
        min_sep_x = (self.benchmark.macro_sizes[i, 0] + self.benchmark.macro_sizes[j, 0]) / 2.0
        min_sep_y = (self.benchmark.macro_sizes[i, 1] + self.benchmark.macro_sizes[j, 1]) / 2.0
        overlap_x = max(0.0, min_sep_x - dx)
        overlap_y = max(0.0, min_sep_y - dy)
        return (overlap_x * overlap_y).item() if overlap_x > 0 and overlap_y > 0 else 0.0

    def update_macro_pos(self, macro_idx: int, new_pos: torch.Tensor):
        """Incrementally update costs for a single macro move."""
        old_pos = self.placement[macro_idx].clone()

        # 1. Update Density Grid
        self._update_density_grid(macro_idx, old_pos, -1.0)
        self._update_density_grid(macro_idx, new_pos, 1.0)

        # 2. Update Overlaps (Vectorized for HARD macros)
        if macro_idx < self.benchmark.num_hard_macros:
            num_hard = self.benchmark.num_hard_macros
            # Get dimensions for all macros
            all_pos = self.placement[:num_hard]
            all_sizes = self.benchmark.macro_sizes[:num_hard]

            # Vectorized overlap check
            # old_ovs
            dx_old = torch.abs(old_pos[0] - all_pos[:, 0])
            dy_old = torch.abs(old_pos[1] - all_pos[:, 1])
            min_sep_x = (self.benchmark.macro_sizes[macro_idx, 0] + all_sizes[:, 0]) / 2.0
            min_sep_y = (self.benchmark.macro_sizes[macro_idx, 1] + all_sizes[:, 1]) / 2.0

            overlap_x_old = torch.clamp(min_sep_x - dx_old, min=0.0)
            overlap_y_old = torch.clamp(min_sep_y - dy_old, min=0.0)
            old_ovs = overlap_x_old * overlap_y_old
            old_ovs[macro_idx] = 0.0 # Don't count self-overlap

            # new_ovs
            dx_new = torch.abs(new_pos[0] - all_pos[:, 0])
            dy_new = torch.abs(new_pos[1] - all_pos[:, 1])
            overlap_x_new = torch.clamp(min_sep_x - dx_new, min=0.0)
            overlap_y_new = torch.clamp(min_sep_y - dy_new, min=0.0)
            new_ovs = overlap_x_new * overlap_y_new
            new_ovs[macro_idx] = 0.0

            self.total_overlap_area += (new_ovs.sum() - old_ovs.sum()).item()

            # Update overlap count
            old_mask = old_ovs > 0
            new_mask = new_ovs > 0
            self.overlap_count += (new_mask.sum() - old_mask.sum()).item()



        # 3. Update HPWL and Congestion Grid
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
            "hard_overlap_area": self.total_overlap_area,
            "overlap_count": self.overlap_count
        }

if __name__ == "__main__":
    import os
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    
    benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    
    if os.path.exists(benchmark_dir):
        benchmark, plc = load_benchmark_from_dir(benchmark_dir)
        
        print("Syncing net weights from PLC to Benchmark for validation...")
        for i, (driver, sinks) in enumerate(plc.nets.items()):
            driver_idx = plc.mod_name_to_indices.get(driver)
            if driver_idx is not None:
                benchmark.net_weights[i] = float(plc.modules_w_pins[driver_idx].get_weight())
        
        print("\n--- Initial Cost Verification ---")
        estimator = CostEstimator(benchmark)
        gt_metrics = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
        pt_metrics = estimator.get_costs()
        
        for k in pt_metrics.keys():
            # Map PT keys to GT keys if necessary
            gt_key = k
            if k == "hard_overlap_area": gt_key = "total_overlap_area"
            
            gt_val = gt_metrics.get(gt_key, 0.0)
            pt_val = pt_metrics[k]
            print(f"{k:20}: GT={gt_val:.6f}, PT={pt_val:.6f}, Delta={pt_val - gt_val:+.6f}")
        
        print("\n--- Incremental Update Verification (Moving macro 0) ---")
        macro_idx = 0
        new_pos = benchmark.macro_positions[macro_idx] + torch.tensor([10.0, 10.0])
        
        # 1. Full recalculation on new placement
        new_placement = benchmark.macro_positions.clone()
        new_placement[macro_idx] = new_pos
        full_metrics = estimate_cost(new_placement, benchmark)
        
        # 2. Incremental update
        estimator.update_macro_pos(macro_idx, new_pos)
        inc_metrics = estimator.get_costs()
        
        for k in inc_metrics.keys():
            if k == "hard_overlap_area": continue
            full_val = full_metrics[k]
            inc_val = inc_metrics[k]
            print(f"{k:20}: Full={full_val:.6f}, Inc={inc_val:.6f}, Delta={inc_val - full_val:+.6f}")
        
        print(f"hard_overlap_area    : Inc={inc_metrics['hard_overlap_area']:.6f}")
