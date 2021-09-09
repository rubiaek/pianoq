import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates.funcs as coord
from colorsys import hls_to_rgb

from results.polarization_meas_result import PolarizationMeasResult
from scipy import ndimage
import qutip

import traceback


class MultiPolarizationMeasResult(object):
    def __init__(self):
        self.roi = None
        self.exposure_time = None

        self.meas1s = []  # qwp.angle = 0,  hwp.angle = 0
        self.meas2s = []  # qwp.angle = 45, hwp.angle = 22.5
        self.meas3s = []  # qwp.angle = 0,  hwp.angle = 22.5

        self.mask_of_interest = None

        self.dac_amplitudes = []

        self.pol_meass = []


    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     roi=self.roi,
                     exposure_time=self.exposure_time,
                     meas1s=self.meas1s,
                     meas2s=self.meas2s,
                     meas3s=self.meas3s,
                     mask_of_interest=self.mask_of_interest,
                     dac_amplitudes=self.dac_amplitudes,
                     )
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path):
        # path = path or self.DEFAULT_PATH
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.roi = data.get('roi', None)
        self.exposure_time = data.get('exposure_time', None)

        self.meas1s = data.get('meas1s', None)
        self.meas2s = data.get('meas2s', None)
        self.meas3s = data.get('meas3s', None)

        self.mask_of_interest = data.get('mask_of_interest', None)
        self.dac_amplitudes = data.get('dac_amplitudes', None)


        for i in range(len(self.meas1s)):
            pol_meas = PolarizationMeasResult()

            pol_meas.roi = self.roi
            pol_meas.exposure_time = self.exposure_time
            pol_meas.mask_of_interest = self.mask_of_interest

            pol_meas.meas1 = self.meas1s[i]
            pol_meas.meas2 = self.meas2s[i]
            pol_meas.meas3 = self.meas3s[i]


            pol_meas.dac_amplitudes = self.dac_amplitudes[i]

            self.pol_meass.append(pol_meas)
