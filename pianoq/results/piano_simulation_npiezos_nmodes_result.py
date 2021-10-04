import numpy as np
import matplotlib.pyplot as plt

import traceback


class PianoSimulationNPiezosNModesResult(object):
    # TODO: all this down here is bogus
    def __init__(self):
        self.version = 0.1
        self.TMs = []
        self.fiber_TM = None
        # we should save the initial anps and final amps for each optimization


    def get_pixels(self):
        simulation = PianoPopoffSimulation(...)  # with relevant parameters
        # and then we can use simulation.get_pixels to show pictures, and to calculate the ratios
