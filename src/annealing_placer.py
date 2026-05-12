import torch
import torch.nn.functional as F
import math
import random
from typing import List, Tuple, Optional
from macro_place.benchmark import Benchmark

class AnnealingPlacer:
    """
    Advanced Simulated Annealing Placer using Convolutional Occupancy Grids.
    """

    def __init__(self, iterations: int = 1000, initial_temp: float = 1.0, 
                 cooling_rate: float = 0.95, initial_prob: float = 0.5, 
                 step_scale: float = 0.1, margin: float = 0.1, 
                 grid_res: float = 0.01, cool_interval: int = 100):
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.initial_prob = initial_prob
        self.step_scale = step_scale
        self.margin = margin
        self.grid_res = grid_res
        self.cool_interval = cool_interval
        
        self.occupancy_grid: Optional[torch.Tensor] = None
        self.grid_w: int = 0
        self.grid_h: int = 0

    # ── Grid Utilities ──────────────────────────────────────────────────────

    def _world_to_grid(self, x, y) -> Tuple[int, int]:
        x_val = x.item() if hasattr(x, "item") else x
        y_val = y.item() if hasattr(y, "item") else y
        return int(round(x_val / self.grid_res)), int(round(y_val / self.grid_res))

    def _grid_to_world(self, gi: int, gj: int) -> Tuple[float, float]:
        return gi * self.grid_res, gj * self.grid_res

    def _update_occupancy(self, pos: torch.Tensor, size: torch.Tensor, delta: float):
        """Additive update to the occupancy grid."""
        if self.occupancy_grid is None: return
        
        tl_x, tl_y = pos[0] - size[0] / 2.0, pos[1] - size[1] / 2.0
        gi, gj = self._world_to_grid(tl_x, tl_y)
        gw, gh = self._world_to_grid(size[0], size[1])
        
        i_start, i_end = max(0, gi), min(self.grid_w, gi + gw)
        j_start, j_end = max(0, gj), min(self.grid_h, gj + gh)
        
        if i_start < i_end and j_start < j_end:
            self.occupancy_grid[0, 0, j_start:j_end, i_start:i_end] += delta

    def _init_grid(self, benchmark: Benchmark, placement: torch.Tensor):
        self.grid_w = int(math.ceil(benchmark.canvas_width / self.grid_res))
        self.grid_h = int(math.ceil(benchmark.canvas_height / self.grid_res))
        
        try:
            self.occupancy_grid = torch.zeros((1, 1, self.grid_h, self.grid_w), dtype=torch.float32)
        except RuntimeError:
            print(f"Warning: Grid resolution {self.grid_res} too fine. Scaling to 0.1.")
            self.grid_res = 0.1
            self.grid_w = int(math.ceil(benchmark.canvas_width / self.grid_res))
            self.grid_h = int(math.ceil(benchmark.canvas_height / self.grid_res))
            self.occupancy_grid = torch.zeros((1, 1, self.grid_h, self.grid_w), dtype=torch.float32)

        for i in range(benchmark.num_macros):
            size_m = benchmark.macro_sizes[i] + self.margin
            self._update_occupancy(placement[i], size_m, 1.0)

    def _get_overlap_mask(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Returns a mask of overlapping macros by checking for grid values > 1."""
        if self.occupancy_grid is None: return torch.zeros(benchmark.num_macros, dtype=torch.bool)
        
        # Binary map of all overlapping pixels
        overlap_map = (self.occupancy_grid[0, 0] > 1.0)
        is_overlapping = torch.zeros(benchmark.num_macros, dtype=torch.bool)
        
        # Only check hard macros
        for i in range(benchmark.num_hard_macros):
            size_m = benchmark.macro_sizes[i] + self.margin
            tl_x, tl_y = placement[i, 0] - size_m[0] / 2.0, placement[i, 1] - size_m[1] / 2.0
            gi, gj = self._world_to_grid(tl_x, tl_y)
            gw, gh = self._world_to_grid(size_m[0], size_m[1])
            
            i_start, i_end = max(0, gi), min(self.grid_w, gi + gw)
            j_start, j_end = max(0, gj), min(self.grid_h, gj + gh)
            
            if i_start < i_end and j_start < j_end:
                if overlap_map[j_start:j_end, i_start:i_end].any():
                    is_overlapping[i] = True
                    
        return is_overlapping

    # ── Proposal Logic ──────────────────────────────────────────────────────

    def _get_availability_heatmap(self, macro_size: torch.Tensor) -> torch.Tensor:
        gw, gh = self._world_to_grid(macro_size[0] + self.margin, macro_size[1] + self.margin)
        if gw <= 0 or gh <= 0: return torch.ones_like(self.occupancy_grid[0, 0])
        
        kernel = torch.ones((1, 1, gh, gw), dtype=torch.float32)
        with torch.no_grad():
            overlap_counts = F.conv2d(self.occupancy_grid, kernel)
        
        # Heatmap: 1.0 if the sum of other macros in this spot is 0
        return (overlap_counts[0, 0] == 0).float()

    def _sample_new_position(self, heatmap: torch.Tensor, current_tl: Tuple[int, int], 
                             max_dist_grid: int) -> Optional[Tuple[int, int]]:
        hh, hw = heatmap.shape
        ci, cj = current_tl
        
        i_min, i_max = max(0, ci - max_dist_grid), min(hw - 1, ci + max_dist_grid)
        j_min, j_max = max(0, cj - max_dist_grid), min(hh - 1, cj + max_dist_grid)
        
        window = heatmap[j_min:j_max+1, i_min:i_max+1]
        valid_indices = torch.where(window > 0)
        
        if len(valid_indices[0]) == 0:
            return None
            
        pick = random.randrange(len(valid_indices[0]))
        return (valid_indices[1][pick].item() + i_min, 
                valid_indices[0][pick].item() + j_min)

    def _propose_batch_moves(self, current_placement: torch.Tensor, is_overlapping: torch.Tensor,
                             movable_indices: torch.Tensor, temp: float, prob: float, 
                             benchmark: Benchmark) -> Tuple[List[int], List[torch.Tensor]]:
        moved_indices, old_positions = [], []
        canvas_w, canvas_h = benchmark.canvas_width, benchmark.canvas_height
        
        for idx_tensor in movable_indices:
            idx = idx_tensor.item()
            if is_overlapping[idx] or random.random() < prob:
                old_pos = current_placement[idx].clone()
                size = benchmark.macro_sizes[idx]
                size_m = size + self.margin
                
                # Sequential update: remove current macro
                self._update_occupancy(old_pos, size_m, -1.0)
                
                heatmap = self._get_availability_heatmap(size)
                tl_world = old_pos - size / 2.0
                ci, cj = self._world_to_grid(tl_world[0], tl_world[1])
                
                scale = self.step_scale * (temp / self.initial_temp)
                max_dist_grid = int(max(1, scale * max(self.grid_w, self.grid_h)))
                
                new_grid_tl = self._sample_new_position(heatmap, (ci, cj), max_dist_grid)
                
                if new_grid_tl:
                    ni, nj = new_grid_tl
                    new_pos = torch.tensor(self._grid_to_world(ni, nj)) + size / 2.0
                    current_placement[idx] = new_pos
                    moved_indices.append(idx)
                    old_positions.append(old_pos)
                    # Add to NEW position
                    self._update_occupancy(new_pos, size_m, 1.0)
                else:
                    # Restore to OLD position
                    self._update_occupancy(old_pos, size_m, 1.0)
        
        return moved_indices, old_positions

    # ── Evaluation and Maintenance ──────────────────────────────────────────

    def _evaluate_and_accept_batch(self, current_placement: torch.Tensor, current_cost: float, 
                                   moved_indices: List[int], old_positions: List[torch.Tensor], 
                                   temp: float, best_cost: float, best_placement: torch.Tensor, 
                                   benchmark: Benchmark) -> Tuple[float, float, torch.Tensor]:
        new_cost = self.compute_cost(current_placement, benchmark)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost, best_placement = current_cost, current_placement.clone()
        else:
            # Backtrack
            for idx, pos in zip(moved_indices, old_positions):
                size_m = benchmark.macro_sizes[idx] + self.margin
                self._update_occupancy(current_placement[idx], size_m, -1.0)
                self._update_occupancy(pos, size_m, 1.0)
                current_placement[idx] = pos
                
        return current_cost, best_cost, best_placement

    def _perform_cooling(self, i: int, temp: float, prob: float, current_cost: float) -> Tuple[float, float]:
        if i % self.cool_interval == 0:
            temp *= self.cooling_rate
            prob *= self.cooling_rate
            print(f"  Iteration {i}/{self.iterations}, cost={current_cost:.2f}, temp={temp:.2f}, prob={prob:.2f}")
        return temp, prob

    def compute_cost(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        all_pos = torch.cat([placement, benchmark.port_positions], dim=0)
        hpwl = 0.0
        for net_indices in benchmark.net_nodes:
            if len(net_indices) < 2: continue
            net_pos = all_pos[net_indices]
            x_min, x_max = net_pos[:, 0].min(), net_pos[:, 0].max()
            y_min, y_max = net_pos[:, 1].min(), net_pos[:, 1].max()
            hpwl += (x_max - x_min) + (y_max - y_min)
        return hpwl.item()

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(42)
        random.seed(42)

        current_placement = torch.round(benchmark.macro_positions / self.grid_res) * self.grid_res
        movable_indices = torch.where(benchmark.get_movable_mask())[0]
        if len(movable_indices) == 0: return current_placement

        self._init_grid(benchmark, current_placement)
        current_cost = self.compute_cost(current_placement, benchmark)
        best_placement, best_cost = current_placement.clone(), current_cost
        temp, prob = self.initial_temp, self.initial_prob
        
        print(f"Starting SA: initial_cost={current_cost:.2f}, prob={prob:.2f}")

        for i in range(self.iterations):
            is_overlapping = self._get_overlap_mask(current_placement, benchmark)
            
            moved_indices, old_positions = self._propose_batch_moves(
                current_placement, is_overlapping, movable_indices, temp, prob, benchmark
            )
            
            if moved_indices:
                current_cost, best_cost, best_placement = self._evaluate_and_accept_batch(
                    current_placement, current_cost, moved_indices, old_positions, 
                    temp, best_cost, best_placement, benchmark
                )
            
            temp, prob = self._perform_cooling(i, temp, prob, current_cost)

        print(f"SA Finished: best_cost={best_cost:.2f}")
        return best_placement
