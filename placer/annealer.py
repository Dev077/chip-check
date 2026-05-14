import os
import sys
from pathlib import Path

# Add project root to sys.path to ensure 'utils' package is findable
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import math
import random
from macro_place.benchmark import Benchmark
try:
    from utils.cost_utils import CostEstimator, IterativeOverlap
except ImportError:
    # Fallback for different execution environments
    from chip_check.utils.cost_utils import CostEstimator, IterativeOverlap

class AnnealerPlacer:
    """
    A fast Simulated Annealing placer using incremental cost updates.
    Optimizes for Wirelength, Density, Congestion, and Legal Placement (overlaps).
    """
    def __init__(self, seed: int = 42, iterations: int = 20000, initial_temp: float = 1.0, cooling_rate: float = 0.99, min_temp: float = 1e-4):
        self.seed = seed
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def place(self, benchmark: Benchmark) -> torch.Tensor:

        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # Initialize incremental trackers
        estimator = CostEstimator(benchmark)
        overlap_tracker = IterativeOverlap(benchmark, benchmark.macro_positions)
        
        # Internal objective weights
        overlap_weight = 2000.0
        pair_penalty = 5.0 # Very aggressive penalty for even a single overlap pair
        
        def get_internal_cost(proxy, area, count):
            return (proxy + 
                    overlap_weight * area + 
                    pair_penalty * count)

        current_metrics = estimator.get_costs()
        current_cost = get_internal_cost(
            current_metrics["proxy_cost"],
            overlap_tracker.total_overlap_area,
            overlap_tracker.overlap_count
        )
        
        best_placement = benchmark.macro_positions.clone()
        best_cost = current_cost
        best_proxy = current_metrics["proxy_cost"]
        best_overlap = overlap_tracker.total_overlap_area
        best_count = overlap_tracker.overlap_count
        
        temp = self.initial_temp
        
        movable_indices = []
        for i in range(benchmark.num_hard_macros):
            if not benchmark.macro_fixed[i]:
                movable_indices.append(i)
        
        if not movable_indices:
            return benchmark.macro_positions

        print(f"Starting Legalizing Fast SA with {len(movable_indices)} movable macros.")
        print(f"Initial Cost: {current_cost:.4f} | Overlap Area: {overlap_tracker.total_overlap_area:.6f} | Count: {overlap_tracker.overlap_count}")

        for i in range(self.iterations):
            idx = random.choice(movable_indices)
            old_pos = estimator.placement[idx].clone()
            
            # Dynamic step size
            scale = 0.1 * (temp / self.initial_temp + 0.01)
            dx = (random.random() * 2 - 1) * benchmark.canvas_width * scale
            dy = (random.random() * 2 - 1) * benchmark.canvas_height * scale
            
            w, h = benchmark.macro_sizes[idx]
            new_x = (old_pos[0] + dx).clamp(w/2, benchmark.canvas_width - w/2)
            new_y = (old_pos[1] + dy).clamp(h/2, benchmark.canvas_height - h/2)
            new_pos = torch.tensor([new_x, new_y])
            
            # Incremental updates
            estimator.update_macro_pos(idx, new_pos)
            overlap_tracker.update_macro_pos(idx, old_pos, new_pos, estimator.placement)
            
            new_metrics = estimator.get_costs()
            new_cost = get_internal_cost(
                new_metrics["proxy_cost"],
                overlap_tracker.total_overlap_area,
                overlap_tracker.overlap_count
            )
            
            delta = new_cost - current_cost
            
            # Metropolis Criterion
            if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_placement = estimator.placement.clone()
                    best_proxy = new_metrics["proxy_cost"]
                    best_overlap = overlap_tracker.total_overlap_area
                    best_count = overlap_tracker.overlap_count
            else:
                # Reject
                estimator.update_macro_pos(idx, old_pos)
                overlap_tracker.update_macro_pos(idx, new_pos, old_pos, estimator.placement)
            
            # Cooling
            temp = max(self.min_temp, temp * self.cooling_rate)

            if (i + 1) % 1000 == 0:
                print(f"  It {i+1}/{self.iterations} | C: {new_cost:.4f} | "
                      f"OA: {overlap_tracker.total_overlap_area:.6f} | "
                      f"OC: {overlap_tracker.overlap_count} | T: {temp:.4f}")

        print(f"Fast SA Finished. Final Best Proxy: {best_proxy:.4f} | Overlap Area: {best_overlap:.6f} | Count: {best_count}")
        return best_placement
