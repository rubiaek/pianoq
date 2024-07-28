import sys
import datetime
from pianoq.lab.photon_scan import PhotonScanner
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.lab.zaber_motor import ZaberMotors
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.mplc.consts import thorlabs_x_serial, thorlabs_y_serial


def idler_scan(name='', integration_time=1.0, coin_window=2e-9):

    start_x = 8.75
    end_x = 9.45
    start_y = 1.2
    end_y = 5.2

    x_pixels = 14
    y_pixels = 80
    pixel_size_x = 0.05
    pixel_size_y = 0.05


    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir_path = r'C:\temp'
    path = f'{dir_path}\\{timestamp}_{name}.scan'
    scanner = PhotonScanner(integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                            run_name=name, is_timetagger=True, coin_window=2e-9, saveto_path=path)

    x_motor = ThorlabsKcubeDC(thorlabs_x_serial)
    print('got x_motor!')
    y_motor = ThorlabsKcubeStepper(thorlabs_y_serial)
    print('got y_motor!')

    time_tagger = QPTimeTagger(integration_time=integration_time, remote=True, coin_window=coin_window)
    # TODO: timetagger delay of -300 ps?
    print('got timetagger!')

    single1s, single2s, coincidences = scanner.scan(x_motor=x_motor, y_motor=y_motor, ph=time_tagger)
    # x_motor.close()
    # y_motor.close()  # pesky bug?
    time_tagger.close()


def signal_scan(name='', integration_time=1.0, coin_window=2e-9):
    start_x = 11.2
    end_x = 11.9
    start_y = 6.5
    end_y = 10.5

    x_pixels = 14
    y_pixels = 80
    pixel_size_x = 0.05
    pixel_size_y = 0.05

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir_path = r'C:\temp'
    path = f'{dir_path}\\{timestamp}_{name}.scan'
    scanner = PhotonScanner(integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                            run_name=name, is_timetagger=True, coin_window=2e-9, saveto_path=path)

    zaber_ms = ZaberMotors(backlash=0., wait_after_move=0.1)
    x_motor = zaber_ms.motors[1]
    y_motor = zaber_ms.motors[0]
    print('Got motors!')

    time_tagger = QPTimeTagger(integration_time=integration_time, remote=True, coin_window=coin_window)
    # TODO: timetagger delay of -300 ps?
    print('got timetagger!')

    single1s, single2s, coincidences = scanner.scan(x_motor=x_motor, y_motor=y_motor, ph=time_tagger, use_power_meter=False)
    x_motor.close()
    time_tagger.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: singless_scan.py <sig|idl>')
    if sys.argv[1] in ['sig', 'signal', 's']:
        signal_scan(integration_time=1, name=f'signal_scan', coin_window=2e-9)
    elif sys.argv[1] in ['idl', 'idler', 'i']:
        idler_scan(integration_time=1, name=f'idler_scan', coin_window=2e-9)
    else:
        print('must use either "sig" or "idl"')
