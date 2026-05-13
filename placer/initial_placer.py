import torch
from macro_place.benchmark import Benchmark

class InitialPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return benchmark.macro_positions
