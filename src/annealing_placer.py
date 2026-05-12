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

    def __init__(self, iterations: int = 1000, initial_temp: float = 1.0, cooling_rate: float = 0.99, initial_prob: float = 0.5, step_scale: float = 0.5, margin: float = 0.1):
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.initial_prob = initial_prob
        self.step_scale = step_scale
        self.margin = margin

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
            # Include margin around hard macros
            min_sep = (sizes[i] + sizes[i+1:]) / 2.0 + self.margin
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
        prob = self.initial_prob
        canvas_size = torch.tensor([benchmark.canvas_width, benchmark.canvas_height])
        num_hard = benchmark.num_hard_macros
        
        print(f"Starting SA: initial_cost={current_cost:.2f}, prob={prob:.2f}")

        for i in range(self.iterations):
            # 1. Identify overlapping hard macros (100% chance to move)
            is_overlapping = torch.zeros(benchmark.num_macros, dtype=torch.bool)
            if num_hard > 0:
                pos_hard = current_placement[:num_hard]
                size_hard = benchmark.macro_sizes[:num_hard]
                # dist: [num_hard, num_hard, 2]
                dist = torch.abs(pos_hard.unsqueeze(1) - pos_hard.unsqueeze(0))
                min_sep = (size_hard.unsqueeze(1) + size_hard.unsqueeze(0)) / 2.0 + self.margin
                overlap_matrix = torch.all(dist < min_sep, dim=2)
                overlap_matrix.fill_diagonal_(False)
                is_overlapping[:num_hard] = overlap_matrix.any(dim=1)

            moved_indices = []
            old_positions = []
            
            # 2. Propose batch moves
            for idx_tensor in movable_indices:
                idx = idx_tensor.item()
                # 100% chance if overlapping, otherwise probabilistic
                if is_overlapping[idx] or random.random() < prob:
                    old_pos = current_placement[idx].clone()
                    
                    # Nudge based on configurable step_scale, decreasing with temperature
                    scale = self.step_scale * (temp / self.initial_temp)
                    nudge = (torch.rand(2) - 0.5) * scale * canvas_size
                    
                    half_size = benchmark.macro_sizes[idx] / 2
                    new_pos = torch.clamp(old_pos + nudge, min=half_size, max=canvas_size - half_size)
                    
                    # Prevent invalid movements: hard macro overlap check
                    if idx < num_hard:
                        # Compare moved macro against all other hard macros in their CURRENT state
                        other_hard_indices = torch.cat([torch.arange(0, idx), torch.arange(idx + 1, num_hard)])
                        if len(other_hard_indices) > 0:
                            pos_others = current_placement[other_hard_indices]
                            size_others = benchmark.macro_sizes[other_hard_indices]
                            size_idx = benchmark.macro_sizes[idx]
                            
                            dist = torch.abs(new_pos - pos_others)
                            # Include margin around hard macros
                            min_sep = (size_idx + size_others) / 2.0 + self.margin
                            overlapping = torch.all(dist < min_sep, dim=1).any().item()
                            
                            if overlapping:
                                continue # Skip this specific macro's move

                    # Apply move and track for potential backtracking
                    current_placement[idx] = new_pos
                    moved_indices.append(idx)
                    old_positions.append(old_pos)

            if not moved_indices:
                # No macros selected or all moves were invalid overlaps
                # Still cool the system to ensure termination/progress
                if i % max(1, self.iterations // 10) == 0:
                    temp *= self.cooling_rate
                    prob *= self.cooling_rate
                continue

            new_cost = self.compute_cost(current_placement, benchmark)
            
            # 2. Accept or Reject batch (Metropolis Criterion)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_placement = current_placement.clone()
            else:
                # Backtrack entire batch
                for idx, pos in zip(moved_indices, old_positions):
                    current_placement[idx] = pos
            
            # 3. Cooling
            if i % 100 == 0:
                temp *= self.cooling_rate
                prob *= self.cooling_rate
                print(f"  Iteration {i}/{self.iterations}, cost={current_cost:.2f}, temp={temp:.2f}, prob={prob:.2f}")

        print(f"SA Finished: best_cost={best_cost:.2f}")
        return best_placement
