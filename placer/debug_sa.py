import torch
import random
from placer.annealer import AnnealerPlacer
from macro_place.loader import load_benchmark_from_dir

benchmark, _ = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
placer = AnnealerPlacer(iterations=20)
placer.place(benchmark)
