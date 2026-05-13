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
    from utils.cost_utils import CostEstimator
except ImportError:
    # Fallback for different execution environments
    from chip_check.utils.cost_utils import CostEstimator

class AnnealerPlacer:
    """
    A fast Simulated Annealing placer using incremental cost updates.
    Optimizes for Wirelength, Density, Congestion, and Legal Placement (overlaps).
    """
    def __init__(self, seed: int = 42, iterations: int = 20000, initial_temp: float = 1.0, cooling_rate: float = 0.9998):
        self.seed = seed
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # Initialize incremental estimator
        estimator = CostEstimator(benchmark)
        
        # Internal objective: ProxyCost + OverlapPenalty
        overlap_weight = 2000.0
        pair_penalty = 5.0 # Very aggressive penalty for even a single overlap pair
        
        def get_internal_cost(metrics):
            return (metrics["proxy_cost"] + 
                    overlap_weight * metrics["hard_overlap_area"] + 
                    pair_penalty * metrics["overlap_count"])

        current_metrics = estimator.get_costs()
        current_cost = get_internal_cost(current_metrics)
        
        best_placement = benchmark.macro_positions.clone()
        best_cost = current_cost
        best_proxy = current_metrics["proxy_cost"]
        best_overlap = current_metrics["hard_overlap_area"]
        best_count = current_metrics["overlap_count"]
        
        temp = self.initial_temp
        
        movable_indices = []
        for i in range(benchmark.num_hard_macros):
            if not benchmark.macro_fixed[i]:
                movable_indices.append(i)
        
        if not movable_indices:
            return benchmark.macro_positions

        print(f"Starting Legalizing Fast SA with {len(movable_indices)} movable macros.")
        print(f"Initial Proxy: {current_metrics['proxy_cost']:.4f} | Overlap Area: {current_metrics['hard_overlap_area']:.6f} | Count: {current_metrics['overlap_count']}")

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
            
            # Incremental update
            estimator.update_macro_pos(idx, new_pos)
            new_metrics = estimator.get_costs()
            new_cost = get_internal_cost(new_metrics)
            
            delta = new_cost - current_cost
            
            # Metropolis Criterion
            if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_placement = estimator.placement.clone()
                    best_proxy = new_metrics["proxy_cost"]
                    best_overlap = new_metrics["hard_overlap_area"]
                    best_count = new_metrics["overlap_count"]
            else:
                # Reject
                estimator.update_macro_pos(idx, old_pos)
            
            # Cooling
            temp *= self.cooling_rate
            
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i+1}/{self.iterations} | Proxy: {new_metrics['proxy_cost']:.4f} | Overlap Area: {new_metrics['hard_overlap_area']:.6f} | Count: {new_metrics['overlap_count']} | Temp: {temp:.4f}")

        print(f"Fast SA Finished. Final Best Proxy: {best_proxy:.4f} | Overlap Area: {best_overlap:.6f} | Count: {best_count}")
        return best_placement
