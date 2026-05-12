import torch
import math
import random
from macro_place.benchmark import Benchmark

class AnnealingPlacer:
    """
    Simulated Annealing Placer.
    
    Optimizes placement by minimizing HPWL and overlap penalties through 
    random perturbations and an exponential cooling schedule.
    """

    def __init__(self, iterations: int = 10000, initial_temp: float = 1., cooling_rate: float = 0.99):
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def compute_cost(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        """Computes HPWL + Overlap Penalty."""
        # 1. HPWL (Wirelength)
        # Combined positions: macros then ports
        all_pos = torch.cat([placement, benchmark.port_positions], dim=0)
        
        hpwl = 0.0
        for net_indices in benchmark.net_nodes:
            if len(net_indices) < 2: continue
            net_pos = all_pos[net_indices]
            x_min = net_pos[:, 0].min()
            x_max = net_pos[:, 0].max()
            y_min = net_pos[:, 1].min()
            y_max = net_pos[:, 1].max()
            hpwl += (x_max - x_min) + (y_max - y_min)

        # 2. Overlap Penalty (only for hard macros)
        overlap_penalty = 0.0
        num_hard = benchmark.num_hard_macros
        pos = placement[:num_hard]
        sizes = benchmark.macro_sizes[:num_hard]
        
        # N^2 overlap check (semi-vectorized)
        for i in range(num_hard):
            dist = torch.abs(pos[i] - pos[i+1:])
            min_sep = (sizes[i] + sizes[i+1:]) / 2.0
            overlap = torch.clamp(min_sep - dist, min=0)
            overlap_area = overlap[:, 0] * overlap[:, 1]
            overlap_penalty += overlap_area.sum().item()

        # Weighted sum: HPWL + High penalty for overlaps to drive legality
        return hpwl.item() + (overlap_penalty * 100.0)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Performs simulated annealing."""
        torch.manual_seed(42)
        random.seed(42)

        # Start with current placement
        current_placement = benchmark.macro_positions.clone()
        movable_mask = benchmark.get_movable_mask()
        movable_indices = torch.where(movable_mask)[0]
        
        if len(movable_indices) == 0:
            return current_placement

        current_cost = self.compute_cost(current_placement, benchmark)
        best_placement = current_placement.clone()
        best_cost = current_cost
        
        temp = self.initial_temp
        canvas_size = torch.tensor([benchmark.canvas_width, benchmark.canvas_height])
        
        print(f"Starting SA: initial_cost={current_cost:.2f}")

        for i in range(self.iterations):
            # 1. Propose a move: pick one movable macro and nudge it
            idx = random.choice(movable_indices).item()
            old_pos = current_placement[idx].clone()
            
            # Nudge up to 10% of canvas, decreasing with temperature
            scale = 0.1 * (temp / self.initial_temp)
            nudge = (torch.rand(2) - 0.5) * scale * canvas_size
            
            half_size = benchmark.macro_sizes[idx] / 2
            new_pos = torch.clamp(old_pos + nudge, min=half_size, max=canvas_size - half_size)
            
            current_placement[idx] = new_pos
            new_cost = self.compute_cost(current_placement, benchmark)
            
            # 2. Accept or Reject (Metropolis Criterion)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_placement = current_placement.clone()
            else:
                current_placement[idx] = old_pos # Backtrack
            
            # 3. Cooling
            if i % 100 == 0:
                temp *= self.cooling_rate
                print(f"  Iteration {i}/{self.iterations}, cost={current_cost:.2f}, temp={temp:.2f}")
            if temp < 0.1:
                break

        print(f"SA Finished: best_cost={best_cost:.2f}")
        return best_placement
