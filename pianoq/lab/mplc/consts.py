import numpy as np

SLM_DIMS = (1024, 1280)
MASK_DIMS = (360, 140)
PIXEL_SIZE = 12.5e-6
N_SPOTS = 25  # per photon
D_BETWEEN_SPOTS_INPUT = 0.3e-3  # 300um between input spots
D_BETWEEN_SPOTS_OUTPUT = 0.3e-3  # 300um between output spots
SPOT_WAIST_IN = 80e-6
D1 = D_BETWEEN_PLANES = 84.5e-3  # in m +-1mm
D2 = D_BETWEEN_PLANES2 = 138e-3  # in m, distance between planes 5 and 6 +-7mm...
WAVELENGTH = 810e-9
K = 2*np.pi/WAVELENGTH

# CENTERS_X = [132, 386, 639, 891, 1143, 1143, 890, 639, 386, 132]  #
# CENTERS_Y = [279, 278, 276, 272, 268, 765, 768, 771, 774, 775]  # 17.7.24 with ronen fix retro X & Y
CENTERS_X = [132, 386, 639, 891, 1143, 1143, 890, 639, 386, 132]  #
CENTERS_Y = [279, 278, 276, 272, 268, 765, 768, 771, 774, 775]  # 07.08.24 ronen with python

MASK_CENTERS = list(zip(CENTERS_X, CENTERS_Y))

thorlabs_x_serial = 27501989  # DC
thorlabs_y_serial = 26003414  # stepper

TIMETAGGER_DELAYS = (0, 300)
TIMETAGGER_COIN_WINDOW = 2e-9


# contains plane numbers and focal lengths
imaging_configs = {
    # imaging1 configurations
    'none': ([], []),
    '1to2w1': ([1], [1]),  # good
    '1to3w2': ([2], [1 / 2]),  # good
    '1to4w2': ([2], [2 / 3]),
    '1to4w3': ([3], [2 / 3]),  # good
    '2to4w3': ([3], [1 / 2]),  #  This is for putting pi step in middle of 2 and of 4 and make sure that they are at the same place (after finding 1-5)
    '1to5w2': ([2], [3 / 4]),
    '1to5w3': ([3], [1]),
    '1to5w4': ([4], [3 / 4]),
    '1to5w4f': ([2, 4], [1, 1]),  # good
    '2to5w3': ([3], [2 / 3]),
    '2to5w4': ([4], [2 / 3]),
    '3to5w4': ([4], [1 / 2]),
    '1to6w25': ([2, 5], [2 / 3, D2 / (D1 + D2)]),
    '1to6w35': ([3, 5], [2 / 3, D2 / (D1 + D2)]),
    '1to6w4': ([4], [3 * (D1 + D2) / (4 * D1 + D2)]),  # good
    '2to6w4': ([4], [2 * (D1 + D2) / (3 * D1 + D2)]),
    '4to6w5': ([5], [D2 / (D1 + D2)]),
    '1to7w4': ([4], [3 * (2 * D1 + D2) / (5 * D1 + D2)]),  # good
    '1to7w5': ([5], [4 * (D1 + D2) / (5 * D1 + D2)]),
    '3to7w5': ([5], [2 * (D1 + D2) / (3 * D1 + D2)]),
    '1to8w4': ([4], [3 * (3 * D1 + D2) / (6 * D1 + D2)]),
    '1to8w5': ([5], [4 * (2 * D1 + D2) / (6 * D1 + D2)]),  # good
    '2to8w5': ([5], [3 * (2 * D1 + D2) / (5 * D1 + D2)]),
    '6to8w7': ([7], [1 / 2]),
    '4to8w6': ([6], [2 * (D1 + D2) / (3 * D1 + D2)]),
    '1to9w4f15and4f59': ([2, 4, 6, 8], [1, 1, 1, 1]),  # approximating D2=D1
    '1to9w5': ([5], [4 * (3 * D1 + D2) / (7 * D1 + D2)]),  # good
    '1to10w4': ([4], [3 * (5 * D1 + D2) / (8 * D1 + D2)]),  # good
    '1to10w5': ([5], [4 * (4 * D1 + D2) / (8 * D1 + D2)]),
    '1to10w4fand8': ([2, 4, 8], [1, 1, 2 * (2 * D1 + D2) / (4 * D1 + D2)]),
    # '2to10w6': ([6], [4 * (3 * D1 + D2) / (7 * D1 + D2)]),  # DUPLICATE. In matlab it was in different switches
    '1to11w4f': ([4, 9], [3, 2]),  # approximating D2=D1
    '1to11w4f_2': ([3, 8], [2, 3]),  # approximating D2=D1
    '1to11w6': ([6], [5 * (4 * D1 + D2) / (9 * D1 + D2)]),

    # imaging2 configurations
    '2to10w6': ([6], [4 * (3 * D1 + D2) / (7 * D1 + D2)]),  # good
    '3to11w7': ([7], [4 * (3 * D1 + D2) / (7 * D1 + D2)]),
    '3to11w5and9': ([5, 9], [2, 2]),  # assuming D1=D2, good
    '4to10w7': ([7], [3 * (2 * D1 + D2) / (5 * D1 + D2)]),
    '4to10w8': ([8], [2 * (3 * D1 + D2) / (5 * D1 + D2)]),
    '5to11w8': ([8], [3 * (2 * D1 + D2) / (5 * D1 + D2)]),
    '5to9w4f': ([6, 8], [1, 1]),  # assuming D1=D2
    '6to10w4f': ([7, 9], [1, 1]),  # good
    '6to10w8': ([8], [1]),
    '7to11w9': ([9], [1]),
    '7to10w9': ([9], [2 / 3]),
    '8to10w9': ([9], [1 / 2]),
    '9to11w10': ([10], [1 / 2]),
}


"""
In Matlab:
* Idler - d1 - Thorlabs, lower numbers in y 
* Signal - d2 - Zaber, higher numbers in y (lower in picture) 
* pos1 = y
* pos2 = x  
"""