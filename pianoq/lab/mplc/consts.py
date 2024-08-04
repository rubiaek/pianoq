SLM_DIMS = (1024, 1280)
MASK_DIMS = (360, 140)
PIXEL_SIZE = 12.5e-6
N_SPOTS = 25  # per photon
D_BETWEEN_SPOTS_INPUT = 0.3e-3  # 300um between input spots
D_BETWEEN_SPOTS_OUTPUT = 0.3e-3  # 300um between output spots
SPOT_WAIST_IN = 80e-6

CENTERS_X = [132, 386, 639, 891, 1143, 1143, 890, 639, 386, 132]  #
CENTERS_Y = [279, 278, 276, 272, 268, 765, 768, 771, 774, 775]  # 17.7.24 with ronen fix retro X & Y

MASK_CENTERS = list(zip(CENTERS_X, CENTERS_Y))

thorlabs_x_serial = 27501989  # DC
thorlabs_y_serial = 26003414  # stepper


"""
In Matlab:
* Idler - d1 - Thorlabs, lower numbers in y 
* Signal - d2 - Zaber, higher numbers in y (lower in picture) 
* pos1 = y
* pos2 = x  
"""