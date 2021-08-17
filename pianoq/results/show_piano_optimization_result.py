import os
import sys
import matplotlib.pyplot as plt
from pianoq.results.PianoOptimizationResult import PianoPSOOptimizationResult

path = sys.argv[1]
name = os.path.basename(path)

ppo = PianoPSOOptimizationResult()
ppo.loadfrom(path)
ppo.show_result()

print(f'best cost: {ppo.costs[-1]}')
e = ppo.enhancement
print(f'enhancement: {e}')

plt.show()
