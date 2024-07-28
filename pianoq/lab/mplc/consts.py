# TODO: fill and verifu
SLM_DIMS = (1024, 1280)
MASK_DIMS = (140, 360)

MASK_CENTERS = [(), (), ()]

thorlabs_x_serial = 27501989 # DC
thorlabs_y_serial = 26003414 # stepper

# TODO: check XY
# x = ThorlabsKcubeDC(27501989)
# y = ThorlabsKcubeStepper(26003414)

"""
Idler - d1 - Thorlabs, lower numbers in y 
Signal - d2 - Zaber, higher numbers in y (lower in picture) 

pos1 = y
pos2 = x  
"""