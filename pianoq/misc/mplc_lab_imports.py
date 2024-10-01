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
from scipy.optimize import curve_fit
from astropy.io import fits


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
from pianoq.misc.misc import detect_gaussian_spots_subpixel

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
from pianoq.lab.VimbaCamera import VimbaCamera, VimbaImage


modes_to_keep = np.array([3, 8, 13, 18, 23, 28, 33, 38, 43, 48])


def get_hardware(backlash=0, wait_after_move=0.3, coin_window=None, integration_time=1,
                 get_s_motors=True, get_i_motors=True, get_timetagger=True, get_mplc=True):

    zaber_ms, mxs, mys, mxi, myi, tt, mplc = None, None, None, None, None, None, None
    if get_s_motors:
        zaber_ms = ZaberMotors(backlash=backlash, wait_after_move=wait_after_move)
        mxs = zaber_ms.motors[1]
        mys = zaber_ms.motors[0]
        print('Got Zaber motors')

    if get_i_motors:
        mxi = ThorlabsKcubeDC(thorlabs_x_serial, backlash=backlash, wait_after_move=wait_after_move)
        myi = ThorlabsKcubeStepper(thorlabs_y_serial, backlash=backlash, wait_after_move=wait_after_move)
        print('Got Thorlabs motors')

    if get_timetagger:
        # Timetagger
        tt = QPTimeTagger(integration_time=integration_time, remote=True,
                          single_channel_delays=TIMETAGGER_DELAYS, coin_window=coin_window or TIMETAGGER_COIN_WINDOW)
        print('God timetagger')

    if get_mplc:
        mplc = MPLCDevice()
        mplc.restore_location()
        print('Got MPLC')

    return zaber_ms, mxs, mys, mxi, myi, tt, mplc


def get_good_masks(masks_path, modes_to_keep=None, phases_path=None):
    msks = MPLCMasks()
    msks.loadfrom(masks_path)
    masks = msks.real_masks
    if modes_to_keep is not None:
        assert isinstance(modes_to_keep, np.ndarray)
        masks = remove_input_modes(masks, modes_to_keep=modes_to_keep)
    if phases_path is not None:
        phases_result = PhaseFinderResult(phases_path)
        masks = add_phase_input_spots(masks, phases_result.phases)

    return masks
