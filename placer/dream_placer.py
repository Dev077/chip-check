import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
from typing import List, Tuple, Optional
from macro_place.benchmark import Benchmark

class DreamPlacer:
    """
    A high-fidelity analytical placer inspired by DREAMPlace.
    Uses:
    1. Differentiable WA Wirelength.
    2. Poisson-based Global Density Potential via FFT.
    3. Per-iteration Gradient Normalization.
    """

    def __init__(
        self,
        iterations: int = 500,
        lr: float = 1.0,
        gamma: float = 10.0,
        init_density_weight: float = 0.1,
        max_density_weight: float = 10.0,
        soft_density_weight: float = 0.01,
        grid_size: int = 128,
        seed: int = 42
    ):
        self.iterations = iterations
        self.lr = lr
        self.gamma = gamma
        self.init_density_weight = init_density_weight
        self.max_density_weight = max_density_weight
        self.soft_density_weight = soft_density_weight
        self.grid_size = grid_size
        self.seed = seed
        
        # Precompute frequency coefficients for Poisson solver
        self.w_inv = None

    def _init_poisson_coefficients(self, device):
        """Precompute 1/(kx^2 + ky^2) for the FFT-based Poisson solver."""
        M, N = self.grid_size, self.grid_size
        
        # Frequency indices
        u = torch.arange(M, device=device).float()
        v = torch.arange(N // 2 + 1, device=device).float() # for rfft2
        
        # For DCT-like behavior or periodic FFT, these are the eigenvalues of the Laplacian
        # Using the standard FFT approach:
        u = torch.where(u > M/2, u - M, u)
        v = torch.where(v > N/2, v - N, v) # Though rfft2 only has half v
        
        # k^2 = (2*pi*u/M)^2 + (2*pi*v/N)^2
        # We use the discrete version: 2 - 2*cos(2*pi*u/M)
        cos_u = torch.cos(2 * torch.pi * u / M)
        cos_v = torch.cos(2 * torch.pi * v / N)
        
        # Laplacian in freq domain: lambda = (2 - 2*cos_u) + (2 - 2*cos_v)
        # We add 1e-6 to avoid div by zero at (0,0)
        self.w_inv = 1.0 / ((2 - 2*cos_u).unsqueeze(1) + (2 - 2*cos_v).unsqueeze(0) + 1e-8)
        self.w_inv[0, 0] = 0.0 # Zero out DC component

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.w_inv is None:
            self._init_poisson_coefficients(device)

        # 1. Prepare data
        num_macros = benchmark.num_macros
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height
        
        pos = benchmark.macro_positions.clone().to(device).detach()
        movable_mask = ~benchmark.macro_fixed.to(device)
        movable_pos = pos[movable_mask].clone().detach().requires_grad_(True)
        
        # Use Adam but with smaller lr if normalization is working
        optimizer = optim.Adam([movable_pos], lr=self.lr)
        
        net_pin_nodes = [net.to(device) for net in benchmark.net_pin_nodes]
        macro_pin_offsets = [off.to(device) for off in benchmark.macro_pin_offsets]
        port_positions = benchmark.port_positions.to(device)
        macro_sizes = benchmark.macro_sizes.to(device)
        net_weights = benchmark.net_weights.to(device)
        
        print(f"Starting Poisson DREAMPlace optimization on {device}")
        
        for i in range(self.iterations):
            optimizer.zero_grad()
            
            # Current placement
            current_pos = pos.clone()
            current_pos[movable_mask] = movable_pos
            
            # Loss 1: Wirelength
            wl_loss = self._compute_wa_wirelength(current_pos, benchmark, net_pin_nodes, macro_pin_offsets, port_positions, net_weights, device)
            
            # Loss 2: Poisson Density (Hard Macros)
            hard_mask = torch.arange(num_macros, device=device) < benchmark.num_hard_macros
            hard_density_loss = self._compute_poisson_density(
                current_pos[hard_mask], 
                macro_sizes[hard_mask], 
                canvas_w, canvas_h, device,
                target_density=1.0 
            )
            
            # Loss 3: Soft Density (Global Spreading)
            soft_mask = ~hard_mask
            if soft_mask.any():
                total_soft_area = (macro_sizes[soft_mask, 0] * macro_sizes[soft_mask, 1]).sum()
                soft_target = (total_soft_area / (canvas_w * canvas_h)).item()
                soft_density_loss = self._compute_poisson_density(
                    current_pos[soft_mask],
                    macro_sizes[soft_mask],
                    canvas_w, canvas_h, device,
                    target_density=soft_target
                )
            else:
                soft_density_loss = torch.tensor(0.0, device=device)
            
            # --- Gradient Normalization (The Critical Step) ---
            # 1. Backprop components separately to get their individual gradients
            wl_loss.backward(retain_graph=True)
            wl_grad = movable_pos.grad.clone()
            optimizer.zero_grad()
            
            hard_density_loss.backward(retain_graph=True)
            den_grad = movable_pos.grad.clone()
            optimizer.zero_grad()
            
            # 2. Compute Norms
            wl_norm = torch.norm(wl_grad) + 1e-8
            den_norm = torch.norm(den_grad) + 1e-8
            
            # 3. Dynamic Weighting
            # We want |Density Gradient| = cur_weight * |WL Gradient|
            progress = i / max(1, self.iterations - 1)
            target_weight = self.init_density_weight + (self.max_density_weight - self.init_density_weight) * progress
            
            # Scaling factor to make gradients comparable
            lambda_den = (wl_norm / den_norm) * target_weight
            
            # Combined Loss
            total_loss = wl_loss + (lambda_den * hard_density_loss) + (self.soft_density_weight * soft_density_loss)
            
            # Final backprop
            total_loss.backward()
            optimizer.step()
            
            # Post-step clamping
            with torch.no_grad():
                idx_m = 0
                for m_idx in range(num_macros):
                    if movable_mask[m_idx]:
                        w, h = macro_sizes[m_idx]
                        movable_pos.data[idx_m, 0].clamp_(w/2, canvas_w - w/2)
                        movable_pos.data[idx_m, 1].clamp_(h/2, canvas_h - h/2)
                        idx_m += 1

            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1:3d} | WL: {wl_loss.item():.4f} | HardDen: {hard_density_loss.item():.4f} | L_Den: {lambda_den:.2f} | Total: {total_loss.item():.4f}")

        final_pos = pos.clone()
        final_pos[movable_mask] = movable_pos.detach()
        return final_pos.cpu()

    def _compute_wa_wirelength(self, pos, benchmark, net_pin_nodes, macro_pin_offsets, port_positions, net_weights, device):
        total_wa = torch.tensor(0.0, device=device)
        all_owner_pos = torch.cat([pos, port_positions], dim=0)
        for net_idx, net_pins in enumerate(net_pin_nodes):
            owner_idx = net_pins[:, 0]
            pin_idx = net_pins[:, 1]
            pins_abs = all_owner_pos[owner_idx].clone()
            hard_macro_mask = owner_idx < benchmark.num_hard_macros
            if hard_macro_mask.any():
                for idx in torch.where(hard_macro_mask)[0]:
                    o_idx = owner_idx[idx].item()
                    p_idx = pin_idx[idx].item()
                    pins_abs[idx] = pins_abs[idx] + macro_pin_offsets[o_idx][p_idx]
            total_wa = total_wa + (self._wa_1d(pins_abs[:, 0]) + self._wa_1d(pins_abs[:, 1])) * net_weights[net_idx]
        return total_wa / ((benchmark.canvas_width + benchmark.canvas_height) * net_weights.sum() + 1e-8)

    def _wa_1d(self, x):
        if x.size(0) <= 1: return torch.tensor(0.0, device=x.device)
        x_max, x_min = torch.max(x), torch.min(x)
        exp_u = torch.exp((x - x_max) / self.gamma)
        exp_l = torch.exp((x_min - x) / self.gamma)
        wa_u = torch.sum(x * exp_u) / (torch.sum(exp_u) + 1e-6)
        wa_l = torch.sum(x * exp_l) / (torch.sum(exp_l) + 1e-6)
        return wa_u - wa_l

    def _compute_poisson_density(self, pos, sizes, canvas_w, canvas_h, device, target_density):
        if pos.size(0) == 0: return torch.tensor(0.0, device=device)
        
        grid_size = self.grid_size
        cell_w, cell_h = canvas_w / grid_size, canvas_h / grid_size
        
        # 1. Differentiable Rasterization (Smooth Mapping)
        grid_x = torch.linspace(cell_w/2, canvas_w - cell_w/2, grid_size, device=device)
        grid_y = torch.linspace(cell_h/2, canvas_h - cell_h/2, grid_size, device=device)
        
        def get_weights(c, s, g, cs):
            dist = torch.abs(g.unsqueeze(0) - c.unsqueeze(1))
            sigma = s.unsqueeze(1) * 0.5 + cs * 0.5
            weight = torch.exp(-0.5 * (dist / sigma)**2)
            return weight / (weight.sum(dim=1, keepdim=True) + 1e-6)

        wx = get_weights(pos[:, 0], sizes[:, 0], grid_x, cell_w)
        wy = get_weights(pos[:, 1], sizes[:, 1], grid_y, cell_h)
        density = torch.matmul(wx.t(), wy * (sizes[:, 0] * sizes[:, 1]).unsqueeze(1)) / (cell_w * cell_h)
        
        # 2. Poisson Solver via FFT
        # rho = density - target
        rho = density - target_density
        
        # FFT to frequency domain
        rho_hat = torch.fft.rfft2(rho)
        
        # Energy = \sum |rho_hat|^2 / k^2
        # This is equivalent to \sum Potential * rho in spatial domain
        energy = torch.sum(torch.abs(rho_hat)**2 * self.w_inv)
        
        return energy

if __name__ == "__main__":
    print("Poisson DreamPlacer ready.")
