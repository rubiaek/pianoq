import os
import sys
import matplotlib.pyplot as plt
from pianoq.results.nmodes_to_piezos_result import NmodesToPiezosResult

path = sys.argv[1]
name = os.path.basename(path)

nn = NmodesToPiezosResult()
nn.loadfrom(path)
nn.show_all_ratios()

print(f'normalize_TMs_method: {nn.normalize_TMs_method}')
print(f'cost_func: {nn.cost_func}')
print(f'pso_n_pop: {nn.pso_n_pop}')
print(f'pso_n_iterations: {nn.pso_n_iterations}')
print(f'pso_stop_after_n_const_iterations: {nn.pso_stop_after_n_const_iterations}')

plt.show()
