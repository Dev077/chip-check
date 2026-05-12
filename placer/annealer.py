import torch
import math
import random
from typing import List, Tuple
from macro_place.benchmark import Benchmark

class AnnealingPlacer:
    """
    Simulated Annealing Placer with Batch Moves and Hard Macro Constraints.
    
    Uses distance-based checks for overlaps and random perturbations for moves.
    """

    def __init__(self, iterations: int = 1000, initial_temp: float = 1.0, 
                 cooling_rate: float = 0.95, initial_prob: float = 0.5, 
                 step_scale: float = 0.1, margin: float = 0.1, 
                 cool_interval: int = 1):
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.initial_prob = initial_prob
        self.step_scale = step_scale
        self.margin = margin
        self.cool_interval = cool_interval

    def _calculate_hpwl(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        """Calculates the Half-Perimeter Wirelength (HPWL)."""
        all_pos = torch.cat([placement, benchmark.port_positions], dim=0)
        hpwl = 0.0
        for net_indices in benchmark.net_nodes:
            if len(net_indices) < 2: continue
            net_pos = all_pos[net_indices]
            x_min, x_max = net_pos[:, 0].min(), net_pos[:, 0].max()
            y_min, y_max = net_pos[:, 1].min(), net_pos[:, 1].max()
            hpwl += (x_max - x_min) + (y_max - y_min)
        return hpwl.item()

    def _get_overlap_mask(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Returns a mask of overlapping HARD macros only using distance checks."""
        num_hard = benchmark.num_hard_macros
        is_overlapping = torch.zeros(benchmark.num_macros, dtype=torch.bool)
        if num_hard <= 1: return is_overlapping
        
        pos_hard = placement[:num_hard]
        size_hard = benchmark.macro_sizes[:num_hard]
        
        # O(N^2) Distance Matrix
        dist = torch.abs(pos_hard.unsqueeze(1) - pos_hard.unsqueeze(0))
        min_sep = (size_hard.unsqueeze(1) + size_hard.unsqueeze(0)) / 2.0 + self.margin
        
        overlap_matrix = torch.all(dist < min_sep, dim=2)
        overlap_matrix.fill_diagonal_(False)
        is_overlapping[:num_hard] = overlap_matrix.any(dim=1)
        
        return is_overlapping

    
    
    def _propose_batch_moves(self, current_placement: torch.Tensor, 
                             movable_indices: torch.Tensor, temp: float, prob: float, 
                             benchmark: Benchmark) -> Tuple[List[int], List[torch.Tensor], int]:
        """Proposes a batch of moves using random nudging (no legality checks)."""
        moved_indices, old_positions = [], []
        attempted_count = 0
        canvas_size = torch.tensor([benchmark.canvas_width, benchmark.canvas_height])

        for idx_tensor in movable_indices:
            idx = idx_tensor.item()

            if random.random() < prob:
                attempted_count += 1
                old_pos = current_placement[idx].clone()
                size = benchmark.macro_sizes[idx]

                # Nudge
                scale = self.step_scale * (temp / self.initial_temp)
                nudge = (torch.rand(2) - 0.5) * scale * canvas_size
                half_size = size / 2.0
                new_pos = torch.clamp(old_pos + nudge, min=half_size, max=canvas_size - half_size)

                # Apply move without overlap legality checks
                current_placement[idx] = new_pos
                moved_indices.append(idx)
                old_positions.append(old_pos)

        return moved_indices, old_positions, attempted_count

    def _evaluate_and_accept_batch(self, current_placement: torch.Tensor, current_cost: float, 
                                   moved_indices: List[int], old_positions: List[torch.Tensor], 
                                   temp: float, best_cost: float, best_placement: torch.Tensor, 
                                   benchmark: Benchmark) -> Tuple[float, float, torch.Tensor]:
        """Evaluates batch moves and handles coordinates backtracking."""
        new_cost = self.compute_cost(current_placement, benchmark)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost, best_placement = current_cost, current_placement.clone()
        else:
            # Backtrack coordinates
            for idx, pos in zip(moved_indices, old_positions):
                current_placement[idx] = pos
                
        return current_cost, best_cost, best_placement

    def _perform_cooling(self, i: int, temp: float, prob: float, current_cost: float, 
                         attempted: int, moved: int, illegal: int, total_hard: int) -> Tuple[float, float]:
        """Applies cooling and prints progress."""
        if i % self.cool_interval == 0:
            temp *= self.cooling_rate
            prob *= self.cooling_rate
            print(f"  Iter {i}/{self.iterations}: cost={current_cost:.2f}, temp={temp:.4f}, prob={prob:.3f}, "
                  f"moved={moved}/{attempted}, illegal={illegal}/{total_hard}")
        return temp, prob

    def compute_cost(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        """HPWL + All Macro Overlap Penalty."""
        # 1. HPWL
        all_pos = torch.cat([placement, benchmark.port_positions], dim=0)
        hpwl = 0.0
        for net_indices in benchmark.net_nodes:
            if len(net_indices) < 2: continue
            net_pos = all_pos[net_indices]
            x_min, x_max = net_pos[:, 0].min(), net_pos[:, 0].max()
            y_min, y_max = net_pos[:, 1].min(), net_pos[:, 1].max()
            hpwl += (x_max - x_min) + (y_max - y_min)
            
        # 2. Overlap Penalty (All macros)
        overlap_penalty = 0.0
        num_macros = benchmark.num_macros
        pos, sizes = placement, benchmark.macro_sizes
        for i in range(num_macros):
            dist = torch.abs(pos[i] - pos[i+1:])
            min_sep = (sizes[i] + sizes[i+1:]) / 2.0 + self.margin
            overlap = torch.clamp(min_sep - dist, min=0)
            overlap_area = overlap[:, 0] * overlap[:, 1]
            overlap_penalty += overlap_area.sum().item()
            
        return hpwl.item() + (overlap_penalty * 100.0)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Simulated annealing with batch moves and distance-based constraints."""
        torch.manual_seed(42)
        random.seed(42)

        num_fixed = benchmark.macro_fixed.sum().item()
        print(f"Benchmark: {benchmark.num_hard_macros} hard, {benchmark.num_soft_macros} soft, {num_fixed} fixed")

        current_placement = benchmark.macro_positions.clone()
        movable_indices = torch.where(benchmark.get_movable_mask())[0]
        if len(movable_indices) == 0: return current_placement

        current_cost = self.compute_cost(current_placement, benchmark)
        best_placement, best_cost = current_placement.clone(), current_cost
        temp, prob = self.initial_temp, self.initial_prob
        
        initial_invalid = self._get_overlap_mask(current_placement, benchmark).sum().item()
        print(f"Starting SA: initial_cost={current_cost:.2f}, prob={prob:.2f}, illegal_hards={initial_invalid}")

        for i in range(self.iterations):
            is_overlapping = self._get_overlap_mask(current_placement, benchmark)
            num_illegal = is_overlapping.sum().item()
            
            moved_indices, old_positions, attempted = self._propose_batch_moves(
                current_placement, movable_indices, temp, prob, benchmark
            )
            
            if moved_indices:
                current_cost, best_cost, best_placement = self._evaluate_and_accept_batch(
                    current_placement, current_cost, moved_indices, old_positions, 
                    temp, best_cost, best_placement, benchmark
                )
            
            temp, prob = self._perform_cooling(i, temp, prob, current_cost, attempted, len(moved_indices), 
                                               num_illegal, benchmark.num_hard_macros)

        print(f"SA Finished: best_cost={best_cost:.2f}")
        return best_placement
