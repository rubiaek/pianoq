import os
import sys
import matplotlib.pyplot as plt
from pianoq.results.piano_optimization_result import PianoPSOOptimizationResult

path = sys.argv[1]
name = os.path.basename(path)

ppo = PianoPSOOptimizationResult()
ppo.loadfrom(path)
ppo.show_result()

print(f'best cost: {ppo.costs[-1]}')
e = ppo.enhancement
print(f'enhancement: {e:.3f}')


print(f'roi shape: {ppo.roi[0].stop - ppo.roi[0].start}X{ppo.roi[1].stop - ppo.roi[1].start}')
print(f'ROI shape: {ppo.roi}')

percentage1, percentage2 = ppo.power_in_L_Pol_before_after
# print(f'initial percentage: {percentage1:.3f}')
# print(f'final percentage: {percentage2:.3f}')


plt.show()
