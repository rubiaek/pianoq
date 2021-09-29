import os
import sys
import matplotlib.pyplot as plt
from pianoq.results.polarization_meas_result import PolarizationMeasResult

path = sys.argv[1]
name = os.path.basename(path)

pom = PolarizationMeasResult()
pom.loadfrom(path)
# pom.plot_polarization_speckle()
pom.plot_poincare()
pom.plot_stokes_params()

print(f'Degree of Polarization: {pom.get_degree_of_polarization()}')
print(f'DAC amplitudes: {pom.dac_amplitudes}')


plt.show()
