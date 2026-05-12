import torch
import random
from src.annealing_placer import AnnealingPlacer
from macro_place.loader import load_benchmark_from_dir

benchmark, _ = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
placer = AnnealingPlacer(iterations=20)
placer.place(benchmark)
