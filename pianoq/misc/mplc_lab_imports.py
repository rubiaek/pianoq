# General
import matplotlib
matplotlib.use('TKAgg')
import os
import glob
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# MPLCSim
from pianoq.simulations.mplc_sim.mplc_sim import MPLCSim
from pianoq.simulations.mplc_sim.mplc_sim_result import MPLCMasks, MPLCSimResult
from pianoq.simulations.mplc_sim.create_wfm_masks import create_WFM_diffuser_masks, create_WFM_unitary_masks, create_WFM_QKD_masks

# MPLC Lab
from pianoq.lab.mplc.singles_scan import signal_scan, idler_scan, get_signal_scanner, get_idler_scanner
from pianoq.lab.mplc.phase_finder_result import PhaseFinderResult
from pianoq.lab.mplc.mask_utils import remove_input_modes, add_phase_input_spots, get_imaging_masks
from pianoq.lab.mplc.mplc_device import MPLCDevice
from pianoq.lab.mplc.discrete_photon_scanner import DiscretePhotonScanner
from pianoq.lab.mplc.phase_finder_result import PhaseFinderResult
from pianoq.lab.mplc.find_discreet_phases import PhaseFinder
from pianoq.lab.mplc.consts import thorlabs_x_serial, thorlabs_y_serial
from pianoq.lab.mplc.mask_aligner import MPLCAligner

# Misc pianoq
from pianoq.misc.misc import run_in_thread, run_in_thread_simple
from pianoq.misc.mplt import mimshow, mplot
from pianoq.lab.photon_scan import PhotonScanner
from pianoq.lab.scripts.live_camera import live_cam

# Results
from pianoq_results.scan_result import ScanResult
from pianoq.lab.mplc.phase_finder_result import PhaseFinderResult
from pianoq_results.fits_image import FITSImage

# Hardware
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.zaber_motor import ZaberMotors

from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.lab.mplc.consts import TIMETAGGER_DELAYS, TIMETAGGER_COIN_WINDOW

from pianoq.lab.pco_camera import PCOCamera