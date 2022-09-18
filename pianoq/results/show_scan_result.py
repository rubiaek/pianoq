import matplotlib as mpl
import matplotlib.pyplot as plt
from pianoq.results.scan_result import ScanResult

import os
import sys

# mpl.use('QT5Agg')

path = sys.argv[1]
name = os.path.basename(path)

r = ScanResult()
r.loadfrom(path)
r.show(True, title=name)
plt.show()
