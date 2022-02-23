import os
import sys
import matplotlib.pyplot as plt
from pianoq.results.waveplates_optimization_result import WavePlateOptimizationResult

path = sys.argv[1]
name = os.path.basename(path)

wr = WavePlateOptimizationResult(path)
# wr.show_heatmap()
# wr.show_cross_section()
wr.show_heatmap_with_cross_section()
print(f"min: {wr.heatmap.min()}")
print(f"max: {wr.heatmap.max()}")

plt.show()
