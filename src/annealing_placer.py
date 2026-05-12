import torch
import math
import random
from typing import List, Tuple
from macro_place.benchmark import Benchmark

class AnnealingPlacer:
    """
    Simulated Annealing Placer.
    
    Optimizes placement by minimizing HPWL and overlap penalties through 
    random perturbations and an exponential cooling schedule.
    """

    def __init__(self, iterations: int = 1000, initial_temp: float = 1.0, 
                 cooling_rate: float = 0.95, initial_prob: float = 0.5, 
                 step_scale: float = 0.1, margin: float = 0.1):
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.initial_prob = initial_prob
        self.step_scale = step_scale
        self.margin = margin

    def _calculate_hpwl(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        """Calculates the Half-Perimeter Wirelength (HPWL)."""
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
        return hpwl.item()

    def _get_overlap_mask(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Returns a boolean mask of macros currently in an overlapping state."""
        num_hard = benchmark.num_hard_macros
        is_overlapping = torch.zeros(benchmark.num_macros, dtype=torch.bool)
        
        if num_hard > 0:
            pos_hard = placement[:num_hard]
            size_hard = benchmark.macro_sizes[:num_hard]
            dist = torch.abs(pos_hard.unsqueeze(1) - pos_hard.unsqueeze(0))
            min_sep = (size_hard.unsqueeze(1) + size_hard.unsqueeze(0)) / 2.0 + self.margin
            overlap_matrix = torch.all(dist < min_sep, dim=2)
            overlap_matrix.fill_diagonal_(False)
            is_overlapping[:num_hard] = overlap_matrix.any(dim=1)
            
        return is_overlapping

    def _calculate_overlap_penalty(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        """Calculates total overlap area penalty for hard macros."""
        penalty = 0.0
        num_hard = benchmark.num_hard_macros
        pos = placement[:num_hard]
        sizes = benchmark.macro_sizes[:num_hard]
        
        for i in range(num_hard):
            dist = torch.abs(pos[i] - pos[i+1:])
            min_sep = (sizes[i] + sizes[i+1:]) / 2.0 + self.margin
            overlap = torch.clamp(min_sep - dist, min=0)
            overlap_area = overlap[:, 0] * overlap[:, 1]
            penalty += overlap_area.sum().item()
            
        return penalty

    def _is_valid_move(self, idx: int, new_pos: torch.Tensor, placement: torch.Tensor, benchmark: Benchmark) -> bool:
        """Checks if a proposed move for a hard macro is legal (no overlaps)."""
        if idx >= benchmark.num_hard_macros:
            return True
            
        num_hard = benchmark.num_hard_macros
        other_indices = torch.cat([torch.arange(0, idx), torch.arange(idx + 1, num_hard)])
        
        if len(other_indices) == 0:
            return True
            
        pos_others = placement[other_indices]
        size_others = benchmark.macro_sizes[other_indices]
        size_idx = benchmark.macro_sizes[idx]
        
        dist = torch.abs(new_pos - pos_others)
        min_sep = (size_idx + size_others) / 2.0 + self.margin
        
        return not torch.all(dist < min_sep, dim=1).any().item()

    def _propose_batch_moves(self, current_placement: torch.Tensor, is_overlapping: torch.Tensor, 
                             movable_indices: torch.Tensor, temp: float, canvas_size: torch.Tensor, 
                             benchmark: Benchmark) -> Tuple[List[int], List[torch.Tensor]]:
        """Selects a batch of macros and proposes new positions for them."""
        moved_indices, old_positions = [], []
        prob = self.initial_prob * (temp / self.initial_temp) # Dynamic prob scaling

        for idx_tensor in movable_indices:
            idx = idx_tensor.item()
            # Selection: Overlapping macros always move; others are probabilistic
            if is_overlapping[idx] or random.random() < prob:
                old_pos = current_placement[idx].clone()
                
                # Nudge
                scale = self.step_scale * (temp / self.initial_temp)
                nudge = (torch.rand(2) - 0.5) * scale * canvas_size
                half_size = benchmark.macro_sizes[idx] / 2
                new_pos = torch.clamp(old_pos + nudge, min=half_size, max=canvas_size - half_size)
                
                # Legality
                if self._is_valid_move(idx, new_pos, current_placement, benchmark):
                    current_placement[idx] = new_pos
                    moved_indices.append(idx)
                    old_positions.append(old_pos)
        
        return moved_indices, old_positions

    def _evaluate_and_accept_batch(self, current_placement: torch.Tensor, current_cost: float, 
                                   moved_indices: List[int], old_positions: List[torch.Tensor], 
                                   temp: float, best_cost: float, best_placement: torch.Tensor, 
                                   benchmark: Benchmark) -> Tuple[float, float, torch.Tensor]:
        """Evaluates batch moves using Metropolis criterion and updates best state."""
        new_cost = self.compute_cost(current_placement, benchmark)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost, best_placement = current_cost, current_placement.clone()
        else:
            # Backtrack
            for idx, pos in zip(moved_indices, old_positions):
                current_placement[idx] = pos
                
        return current_cost, best_cost, best_placement

    def _perform_cooling(self, i: int, temp: float, prob: float, current_cost: float) -> Tuple[float, float]:
        """Applies cooling schedule and prints progress periodically."""
        interval = max(1, self.iterations // 10)
        if i % interval == 0:
            temp *= self.cooling_rate
            prob *= self.cooling_rate
            print(f"  Iteration {i}/{self.iterations}, cost={current_cost:.2f}, temp={temp:.2f}, prob={prob:.2f}")
        return temp, prob

    def compute_cost(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        """Computes HPWL + Overlap Penalty."""
        hpwl = self._calculate_hpwl(placement, benchmark)
        overlap_penalty = self._calculate_overlap_penalty(placement, benchmark)
        return hpwl + (overlap_penalty * 100.0)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Performs simulated annealing with batch moves."""
        torch.manual_seed(42)
        random.seed(42)

        current_placement = benchmark.macro_positions.clone()
        movable_indices = torch.where(benchmark.get_movable_mask())[0]
        
        if len(movable_indices) == 0:
            return current_placement

        current_cost = self.compute_cost(current_placement, benchmark)
        best_placement, best_cost = current_placement.clone(), current_cost
        temp, prob = self.initial_temp, self.initial_prob
        canvas_size = torch.tensor([benchmark.canvas_width, benchmark.canvas_height])
        
        print(f"Starting SA: initial_cost={current_cost:.2f}, prob={prob:.2f}")

        for i in range(self.iterations):
            is_overlapping = self._get_overlap_mask(current_placement, benchmark)
            
            moved_indices, old_positions = self._propose_batch_moves(
                current_placement, is_overlapping, movable_indices, temp, canvas_size, benchmark
            )
            
            if moved_indices:
                current_cost, best_cost, best_placement = self._evaluate_and_accept_batch(
                    current_placement, current_cost, moved_indices, old_positions, 
                    temp, best_cost, best_placement, benchmark
                )
            
            temp, prob = self._perform_cooling(i, temp, prob, current_cost)

        print(f"SA Finished: best_cost={best_cost:.2f}")
        return best_placement
