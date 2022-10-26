import numpy as np
import  matplotlib.pyplot as plt
import glob
import sys
sys.path.append('C:\\code')
sys.path.append('C:\\code\\pianoq_results')

from pianoq_results import FITSImage

PATH = r'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\PPKTP\New-2022-10\f=250_before\Temperature\Image\*.fit'


def main():
    paths = glob.glob(PATH)
    Ts = []
    sigs = []
    sigs_err = []
    for path in paths:
        fi = FITSImage(path)
        fi.fit_to_gaus(680, 20, line_no=372) # TODO TBC

