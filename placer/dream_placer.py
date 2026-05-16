import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from macro_place.benchmark import Benchmark

class DreamPlacer:
    """
    An analytical placer inspired by DREAMPlace.
    Uses differentiable Weighted-Average (WA) wirelength and a grid-based density potential.
    Optimizes using gradient descent (Adam).
    """

    def __init__(
        self,
        iterations: int = 200,
        lr: float = 1e-1,
        gamma: float = 10.0,
        init_density_weight: float = 0.1,
        max_density_weight: float = 2.0,
        soft_density_weight: float = 0.1,
        grid_size: int = 128,
        seed: int = 42
    ):
        self.iterations = iterations
        self.lr = lr
        self.gamma = gamma  # Smoothing parameter for WA wirelength
        self.init_density_weight = init_density_weight
        self.max_density_weight = max_density_weight
        self.soft_density_weight = soft_density_weight
        self.grid_size = grid_size
        self.seed = seed


    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Prepare data
        num_macros = benchmark.num_macros
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height
        
        # Initialize positions as parameters (only movable ones)
        pos = benchmark.macro_positions.clone().to(device).detach()
        movable_mask = ~benchmark.macro_fixed.to(device)
        
        # Optimization variable: only movable macro positions
        movable_pos = pos[movable_mask].clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([movable_pos], lr=self.lr)
        
        # Pre-process nets and pin offsets for device
        net_pin_nodes = [net.to(device) for net in benchmark.net_pin_nodes]
        macro_pin_offsets = [off.to(device) for off in benchmark.macro_pin_offsets]
        port_positions = benchmark.port_positions.to(device)
        macro_sizes = benchmark.macro_sizes.to(device)
        net_weights = benchmark.net_weights.to(device)
        
        print(f"Starting DREAMPlace-style optimization on {device}")
        
        # 2. Optimization Loop
        for i in range(self.iterations):
            optimizer.zero_grad()
            
            # Reconstruct full placement
            current_pos = pos.clone()
            current_pos[movable_mask] = movable_pos
            
            # Loss 1: Wirelength (WA)
            wl_loss = self._compute_wa_wirelength(current_pos, benchmark, net_pin_nodes, macro_pin_offsets, port_positions, net_weights, device)
            
            # Loss 2: Density (Split into Hard and Soft)
            # Hard macros: Indices [0, num_hard_macros)
            # Soft macros: Indices [num_hard_macros, num_macros)
            
            # Hard Macro Density - Targeting 100% occupancy to prevent overlaps
            hard_mask = torch.arange(num_macros, device=device) < benchmark.num_hard_macros
            hard_density_loss = self._compute_density_potential(
                current_pos[hard_mask], 
                macro_sizes[hard_mask], 
                canvas_w, canvas_h, device,
                target_density=1.0 # Strict target for legalization
            )
            
            # Soft Macro Density - Targeting global average to keep them spread but flexible
            soft_mask = ~hard_mask
            if soft_mask.any():
                total_soft_area = (macro_sizes[soft_mask, 0] * macro_sizes[soft_mask, 1]).sum()
                soft_target = (total_soft_area / (canvas_w * canvas_h)).item()
                soft_density_loss = self._compute_density_potential(
                    current_pos[soft_mask],
                    macro_sizes[soft_mask],
                    canvas_w, canvas_h, device,
                    target_density=soft_target
                )
            else:
                soft_density_loss = torch.tensor(0.0, device=device)
            
            # Total Loss
            # Ramping weight only applies to the "illegal" hard macro overlaps
            progress = i / max(1, self.iterations - 1)
            cur_hard_weight = self.init_density_weight + (self.max_density_weight - self.init_density_weight) * progress
            
            total_loss = wl_loss + (cur_hard_weight * hard_density_loss) + (self.soft_density_weight * soft_density_loss)
            
            total_loss.backward()
            optimizer.step()
            
            # Post-step: Keep macros within canvas
            with torch.no_grad():
                idx_movable = 0
                for macro_idx in range(num_macros):
                    if movable_mask[macro_idx]:
                        w, h = macro_sizes[macro_idx]
                        movable_pos.data[idx_movable, 0].clamp_(w/2, canvas_w - w/2)
                        movable_pos.data[idx_movable, 1].clamp_(h/2, canvas_h - h/2)
                        idx_movable += 1

            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1:3d}/{self.iterations} | WL: {wl_loss.item():.6f} | HardDen: {hard_density_loss.item():.6f} (x{cur_hard_weight:.2f}) | SoftDen: {soft_density_loss.item():.6f} (x{self.soft_density_weight:.2f}) | Total: {total_loss.item():.6f}")

        # Final placement
        final_pos = pos.clone()
        final_pos[movable_mask] = movable_pos.detach()
        return final_pos.cpu()

    def _compute_wa_wirelength(self, pos: torch.Tensor, benchmark: Benchmark, net_pin_nodes, macro_pin_offsets, port_positions, net_weights, device) -> torch.Tensor:
        """Differentiable Weighted-Average (WA) wirelength."""
        total_wa = torch.tensor(0.0, device=device)
        all_owner_pos = torch.cat([pos, port_positions], dim=0)
        
        # Optimization: Loop over nets. 
        # For large benchmarks, this should be vectorized with padding or scatter.
        for net_idx, net_pins in enumerate(net_pin_nodes):
            owner_idx = net_pins[:, 0]
            pin_idx = net_pins[:, 1]
            base_pos = all_owner_pos[owner_idx]
            
            # Add pin offsets
            pins_abs = base_pos.clone()
            # Hard macro offsets
            hard_macro_mask = owner_idx < benchmark.num_hard_macros
            if hard_macro_mask.any():
                for pin_in_net_idx in torch.where(hard_macro_mask)[0]:
                    o_idx = owner_idx[pin_in_net_idx].item()
                    p_idx = pin_idx[pin_in_net_idx].item()
                    pins_abs[pin_in_net_idx] = pins_abs[pin_in_net_idx] + macro_pin_offsets[o_idx][p_idx]
            
            net_wa = self._wa_1d(pins_abs[:, 0]) + self._wa_1d(pins_abs[:, 1])
            total_wa = total_wa + net_wa * net_weights[net_idx]
            
        total_connections = net_weights.sum().item()
        norm_factor = (benchmark.canvas_width + benchmark.canvas_height) * total_connections
        return total_wa / norm_factor if norm_factor != 0 else total_wa

    def _wa_1d(self, x: torch.Tensor) -> torch.Tensor:
        """1D WA wirelength for a single net."""
        if x.size(0) <= 1: return torch.tensor(0.0, device=x.device)
        
        # Max/Min for numerical stability
        x_max = torch.max(x)
        x_min = torch.min(x)
        
        exp_u = torch.exp((x - x_max) / self.gamma)
        sum_exp_u = torch.sum(exp_u)
        wa_u = torch.sum(x * exp_u) / (sum_exp_u + 1e-6)
        
        exp_l = torch.exp((x_min - x) / self.gamma)
        sum_exp_l = torch.sum(exp_l)
        wa_l = torch.sum(x * exp_l) / (sum_exp_l + 1e-6)
        
        return wa_u - wa_l

    def _compute_density_potential(self, pos: torch.Tensor, sizes: torch.Tensor, canvas_w: float, canvas_h: float, device, target_density: float = 1.0) -> torch.Tensor:
        """Differentiable grid-based density potential for a subset of macros."""
        if pos.size(0) == 0:
            return torch.tensor(0.0, device=device)
            
        grid_size = self.grid_size
        cell_w, cell_h = canvas_w / grid_size, canvas_h / grid_size
        
        # Grid coordinates
        grid_x = torch.linspace(cell_w/2, canvas_w - cell_w/2, grid_size, device=device)
        grid_y = torch.linspace(cell_h/2, canvas_h - cell_h/2, grid_size, device=device)
        
        macro_x, macro_y = pos[:, 0], pos[:, 1]
        macro_w, macro_h = sizes[:, 0], sizes[:, 1]
        
        def get_weights(c, s, g, cs):
            # c: [N], s: [N], g: [G], cs: float
            dist = torch.abs(g.unsqueeze(0) - c.unsqueeze(1))
            # Use a Gaussian kernel as a smooth approximation of a box
            sigma = s.unsqueeze(1) * 0.5 + cs * 0.5
            weight = torch.exp(-0.5 * (dist / sigma)**2)
            return weight / (weight.sum(dim=1, keepdim=True) + 1e-6)

        wx = get_weights(macro_x, macro_w, grid_x, cell_w)
        wy = get_weights(macro_y, macro_h, grid_y, cell_h)
        
        areas = (macro_w * macro_h).unsqueeze(1)
        density = torch.matmul(wx.t(), wy * areas)
        
        density_norm = density / (cell_w * cell_h)
        
        # Penalty for exceeding target occupancy
        return torch.mean((density_norm - target_density).clamp(min=0)**2)
