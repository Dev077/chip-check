"""
Soft-macro spreader for macro placement.

Runs as a post-process after hard-macro legalization. Given a legal hard-
macro placement, optimizes soft-macro positions to minimize wirelength and
density cost, treating the hard macros as fixed obstacles.

This is the analytical equivalent of TILOS's `plc.optimize_stdcells()` —
the SA baseline runs it between hard-macro batches; we run it once at the
end. Soft macros are ~40% of canvas area and contribute heavily to all
three proxy-cost terms (wirelength, density, congestion), so leaving them
where DREAM dropped them wastes significant cost.

Uses the same building blocks as DreamPlacer:
  - Weighted-average wirelength on every net (hard pin offsets included)
  - FFT-based Poisson density potential
  - Per-iteration gradient normalization between wirelength and density

Differences from DREAM's joint optimization:
  - Only soft positions are trainable. Hard positions are fixed (legalized).
  - Density target is set from soft area / canvas area (so we spread to
    the natural density floor, not to 1.0).
  - No overlap penalty — soft macros are allowed to overlap each other
    and hard macros per the evaluator's rules.
"""

import torch
import torch.optim as optim

from macro_place.benchmark import Benchmark


class SoftSpreader:
    """Force-directed soft-macro optimizer."""

    def __init__(
        self,
        iterations: int = 200,
        lr: float = 1.0,
        gamma: float = 10.0,
        density_weight: float = 1.0,
        grid_size: int = 128,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.iterations = iterations
        self.lr = lr
        self.gamma = gamma
        # Target ratio between density-grad norm and wirelength-grad norm.
        # 1.0 means we push density just as hard as wirelength. Soft macros
        # benefit from a higher value than hard macros because they have
        # nothing else stopping them from piling up.
        self.density_weight = density_weight
        self.grid_size = grid_size
        self.seed = seed
        self.verbose = verbose
        self.w_inv = None

    # ----------------------------------------------------------------- entry

    def spread(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """
        Return a new placement with soft-macro positions optimized.

        Hard-macro positions in `placement` are preserved exactly.
        """
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.w_inv is None:
            self._init_poisson_coefficients(device)

        num_hard = benchmark.num_hard_macros
        num_macros = benchmark.num_macros
        num_soft = num_macros - num_hard

        if num_soft == 0:
            if self.verbose:
                print("[SoftSpreader] no soft macros — skipping")
            return placement

        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        pos = placement.clone().detach().to(device)
        macro_sizes = benchmark.macro_sizes.to(device)
        net_weights = benchmark.net_weights.to(device)
        port_positions = benchmark.port_positions.to(device)
        net_pin_nodes = [n.to(device) for n in benchmark.net_pin_nodes]
        macro_pin_offsets = [o.to(device) for o in benchmark.macro_pin_offsets]

        # Only soft positions are trainable. Hard positions are detached
        # constants — they contribute to wirelength and density but their
        # gradients are dropped.
        soft_pos = pos[num_hard:].clone().detach().requires_grad_(True)
        hard_pos = pos[:num_hard].detach()

        optimizer = optim.Adam([soft_pos], lr=self.lr)

        # Target density: total movable area / canvas area. This is the
        # uniform-spread floor — pushing past it would mean leaving the
        # canvas, which is impossible. Hard macros are *included* in the
        # density field (they take up real space) but we set the target
        # to all-macro area / canvas so the density loss is calibrated to
        # the achievable spread.
        total_area = (macro_sizes[:, 0] * macro_sizes[:, 1]).sum()
        target_density = (total_area / (canvas_w * canvas_h)).item()
        target_density = min(target_density, 1.0)

        if self.verbose:
            print(
                f"[SoftSpreader] spreading {num_soft} soft macros on {device} | "
                f"target_density={target_density:.3f}"
            )

        for it in range(self.iterations):
            # Assemble full position tensor (hard fixed, soft trainable).
            current_pos = torch.cat([hard_pos, soft_pos], dim=0)

            # ── Wirelength on every net ────────────────────────────────
            wl_loss = self._compute_wa_wirelength(
                current_pos, benchmark, net_pin_nodes, macro_pin_offsets,
                port_positions, net_weights, device,
            )

            # ── Density across all macros, target = movable-area ratio ──
            den_loss = self._compute_poisson_density(
                current_pos, macro_sizes, canvas_w, canvas_h, device, target_density,
            )

            # ── Per-iteration gradient normalization ───────────────────
            # Same trick DREAM uses: keep the density push and wirelength
            # push at a controllable ratio regardless of their raw scales.
            optimizer.zero_grad()
            wl_loss.backward(retain_graph=True)
            wl_grad = soft_pos.grad.clone()

            optimizer.zero_grad()
            den_loss.backward(retain_graph=True)
            den_grad = soft_pos.grad.clone()
            optimizer.zero_grad()

            wl_norm = torch.norm(wl_grad) + 1e-8
            den_norm = torch.norm(den_grad) + 1e-8
            lambda_den = (wl_norm / den_norm) * self.density_weight

            total_loss = wl_loss + lambda_den * den_loss
            total_loss.backward()
            optimizer.step()

            # Clamp soft positions inside the canvas.
            with torch.no_grad():
                w = macro_sizes[num_hard:, 0]
                h = macro_sizes[num_hard:, 1]
                soft_pos.data[:, 0].clamp_(w * 0.5, canvas_w - w * 0.5)
                soft_pos.data[:, 1].clamp_(h * 0.5, canvas_h - h * 0.5)

            if self.verbose and (it + 1) % 25 == 0:
                print(
                    f"[SoftSpreader]   iter {it + 1:3d} | "
                    f"WL: {wl_loss.item():.4f} | Den: {den_loss.item():.4f} | "
                    f"L_Den: {lambda_den.item():.2f}"
                )

        # Stitch the optimized soft positions back into the placement and
        # return on CPU (caller expects a CPU tensor, matching DREAM).
        out = placement.clone()
        out[num_hard:] = soft_pos.detach().cpu()
        if self.verbose:
            mean_disp = (out[num_hard:] - placement[num_hard:]).norm(dim=1).mean().item()
            print(f"[SoftSpreader] done. mean soft displacement = {mean_disp:.2f} µm")
        return out

    # -------------------------------------------------------------- internals

    def _init_poisson_coefficients(self, device):
        """Same setup as DreamPlacer — eigenvalues of the discrete Laplacian."""
        M, N = self.grid_size, self.grid_size
        u = torch.arange(M, device=device).float()
        v = torch.arange(N // 2 + 1, device=device).float()
        u = torch.where(u > M / 2, u - M, u)
        v = torch.where(v > N / 2, v - N, v)
        cos_u = torch.cos(2 * torch.pi * u / M)
        cos_v = torch.cos(2 * torch.pi * v / N)
        self.w_inv = 1.0 / ((2 - 2 * cos_u).unsqueeze(1) + (2 - 2 * cos_v).unsqueeze(0) + 1e-8)
        self.w_inv[0, 0] = 0.0

    # The wirelength and density helpers are intentionally written to match
    # DreamPlacer's so future bug-fixes can be ported across in lockstep.
    # If you find a bug here, check dream_placer.py for the same one.

    def _compute_wa_wirelength(
        self, pos, benchmark, net_pin_nodes, macro_pin_offsets,
        port_positions, net_weights, device,
    ):
        total_wa = torch.tensor(0.0, device=device)
        all_owner_pos = torch.cat([pos, port_positions], dim=0)
        for net_idx, net_pins in enumerate(net_pin_nodes):
            owner_idx = net_pins[:, 0]
            pin_idx = net_pins[:, 1]
            pins_abs = all_owner_pos[owner_idx].clone()
            # Add per-pin offsets for hard macros (soft macros + ports have
            # implicit zero offsets — their pins coincide with the center).
            hard_macro_mask = owner_idx < benchmark.num_hard_macros
            if hard_macro_mask.any():
                for idx in torch.where(hard_macro_mask)[0]:
                    o_idx = owner_idx[idx].item()
                    p_idx = pin_idx[idx].item()
                    pins_abs[idx] = pins_abs[idx] + macro_pin_offsets[o_idx][p_idx]
            total_wa = total_wa + (
                self._wa_1d(pins_abs[:, 0]) + self._wa_1d(pins_abs[:, 1])
            ) * net_weights[net_idx]
        return total_wa / (
            (benchmark.canvas_width + benchmark.canvas_height) * net_weights.sum() + 1e-8
        )

    def _wa_1d(self, x):
        if x.size(0) <= 1:
            return torch.tensor(0.0, device=x.device)
        x_max, x_min = torch.max(x), torch.min(x)
        exp_u = torch.exp((x - x_max) / self.gamma)
        exp_l = torch.exp((x_min - x) / self.gamma)
        wa_u = torch.sum(x * exp_u) / (torch.sum(exp_u) + 1e-6)
        wa_l = torch.sum(x * exp_l) / (torch.sum(exp_l) + 1e-6)
        return wa_u - wa_l

    def _compute_poisson_density(self, pos, sizes, canvas_w, canvas_h, device, target_density):
        if pos.size(0) == 0:
            return torch.tensor(0.0, device=device)
        grid_size = self.grid_size
        cell_w, cell_h = canvas_w / grid_size, canvas_h / grid_size

        grid_x = torch.linspace(cell_w / 2, canvas_w - cell_w / 2, grid_size, device=device)
        grid_y = torch.linspace(cell_h / 2, canvas_h - cell_h / 2, grid_size, device=device)

        def get_weights(c, s, g, cs):
            dist = torch.abs(g.unsqueeze(0) - c.unsqueeze(1))
            sigma = s.unsqueeze(1) * 0.5 + cs * 0.5
            weight = torch.exp(-0.5 * (dist / sigma) ** 2)
            return weight / (weight.sum(dim=1, keepdim=True) + 1e-6)

        wx = get_weights(pos[:, 0], sizes[:, 0], grid_x, cell_w)
        wy = get_weights(pos[:, 1], sizes[:, 1], grid_y, cell_h)
        density = torch.matmul(
            wx.t(), wy * (sizes[:, 0] * sizes[:, 1]).unsqueeze(1)
        ) / (cell_w * cell_h)

        rho = density - target_density
        rho_hat = torch.fft.rfft2(rho)
        energy = torch.sum(torch.abs(rho_hat) ** 2 * self.w_inv)
        return energy