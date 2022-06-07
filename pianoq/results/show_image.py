import os
import sys
import matplotlib.pyplot as plt
from pianoq.results.image_result import show_image

path = sys.argv[1]
title = os.path.basename(path)
show_image(path, title=title)

plt.show()
