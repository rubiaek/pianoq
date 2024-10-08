import sys
import numpy as np
import datetime
from pianoq.lab.photon_scan import PhotonScanner
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.lab.zaber_motor import ZaberMotors
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.mplc.consts import thorlabs_x_serial, thorlabs_y_serial, TIMETAGGER_DELAYS, TIMETAGGER_COIN_WINDOW
from pianoq.lab.mplc.mplc_device import MPLCDevice
from pianoq.lab.mplc.mask_utils import remove_input_modes, add_phase_input_spots
from pianoq.lab.mplc.phase_finder_result import PhaseFinderResult

DIR_PATH = r'G:\My Drive\People\Ronen\PHD\MPLC\results'


def idler_scan(name='', integration_time=1.0, coin_window=2e-9, resolution=1, out_path=None, half_scan=True):
    scanner, x_motor, y_motor, time_tagger = get_idler_scanner(name=name, integration_time=integration_time, coin_window=coin_window,
                      resolution=resolution, out_path=out_path, half_scan=half_scan)
    single1s, single2s, coincidences = scanner.scan(x_motor=x_motor, y_motor=y_motor, ph=time_tagger)
    # x_motor.close()
    # y_motor.close()  # pesky bug?
    time_tagger.close()

def get_idler_scanner(name='', integration_time=1.0, coin_window=2e-9, resolution=1, out_path=None, half_scan=True, no_hw_mode=False):
    start_x = 8.75
    end_x = 9.45
    start_y = 1.2
    end_y = 5.2

    x_pixels = 28 // resolution
    y_pixels = 160 // resolution
    pixel_size_x = 0.025 * resolution
    pixel_size_y = 0.025 * resolution

    if half_scan:
        y_pixels = (y_pixels // 2) + 6

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    path = out_path or f'{DIR_PATH}\\{timestamp}_{name}.scan'
    scanner = PhotonScanner(integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                            run_name=name, is_timetagger=True, coin_window=2e-9, saveto_path=path)

    if no_hw_mode:
        return scanner, None, None, None

    x_motor = ThorlabsKcubeDC(thorlabs_x_serial, backlash=0.2, wait_after_move=0.3)
    print('got x_motor!')
    y_motor = ThorlabsKcubeStepper(thorlabs_y_serial, backlash=0.2, wait_after_move=0.3)
    print('got y_motor!')

    time_tagger = QPTimeTagger(integration_time=integration_time, remote=True,
                               coin_window=coin_window or TIMETAGGER_COIN_WINDOW,
                               single_channel_delays=TIMETAGGER_DELAYS)
    print('got timetagger!')

    return scanner, x_motor, y_motor, time_tagger


def signal_scan(name='', integration_time=1.0, coin_window=2e-9, resolution=1, out_path=None, half_scan=True):
    scanner, x_motor, y_motor, time_tagger = get_signal_scanner(name=name, integration_time=integration_time, coin_window=coin_window,
                      resolution=resolution, out_path=out_path, half_scan=half_scan)
    single1s, single2s, coincidences = scanner.scan(x_motor=x_motor, y_motor=y_motor, ph=time_tagger)
    # x_motor.close()
    # y_motor.close()  # pesky bug?
    time_tagger.close()

def get_signal_scanner(name='', integration_time=1.0, coin_window=2e-9, resolution=1, out_path=None, half_scan=True, no_hw_mode=False):
    """ higher y values, zaber motors """
    start_x = 11.2
    end_x = 11.9
    start_y = 6.5
    end_y = 10.5

    x_pixels = 28 // resolution
    y_pixels = 160  // resolution
    pixel_size_x = 0.025 * resolution
    pixel_size_y = 0.025 * resolution

    if half_scan:
        start_y = start_y + pixel_size_y * (y_pixels / 2)
        y_pixels = y_pixels // 2

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    path = out_path or f'{DIR_PATH}\\{timestamp}_{name}.scan'
    scanner = PhotonScanner(integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                            run_name=name, is_timetagger=True, coin_window=2e-9, saveto_path=path)

    if no_hw_mode:
        return scanner, None, None, None

    zaber_ms = ZaberMotors(backlash=0.2, wait_after_move=0.3)
    x_motor = zaber_ms.motors[1]
    y_motor = zaber_ms.motors[0]
    print('Got motors!')

    time_tagger = QPTimeTagger(integration_time=integration_time, remote=True,
                               coin_window=coin_window or TIMETAGGER_COIN_WINDOW,
                               single_channel_delays=TIMETAGGER_DELAYS)
    print('got timetagger!')

    return scanner, x_motor, y_motor, time_tagger


if __name__ == '__main__':
    mplc = MPLCDevice()
    path = r"G:\My Drive\People\Ronen\PHD\MPLC\results\rss_wfm1.masks"
    masks = np.load(path)['masks']
    modes_to_keep = np.array([3, 8, 13, 18, 23, 28, 33, 38, 43, 48])
    masks = remove_input_modes(masks, modes_to_keep=modes_to_keep)
    phases_result = PhaseFinderResult()
    phases_result.loadfrom(r"G:\My Drive\People\Ronen\PHD\MPLC\results\2024_08_11_10_26_11_QKD_row3_phases.phases")
    masks = add_phase_input_spots(masks, phases_result.phases)
    mplc.load_masks(masks, linear_tilts=True)
    resolution = 1

    if len(sys.argv) < 2:
        print('Usage: singles_scan.py <sig|idl>')
    if sys.argv[1] in ['sig', 'signal', 's']:
        signal_scan(integration_time=1, name=f'signal_scan', resolution=resolution)
    elif sys.argv[1] in ['idl', 'idler', 'i']:
        idler_scan(integration_time=1, name=f'idler_scan', resolution=resolution)
    else:
        print('must use either "sig" or "idl"')
