import sys
import matplotlib.pyplot as plt
from pianoq.results.image_result import show_image

path = sys.argv[1]
show_image(path)

plt.show()
