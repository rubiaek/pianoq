from pianoq.lab.mplc.consts import thorlabs_x_serial, thorlabs_y_serial
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.zaber_motor import ZaberMotors
from pianoq.lab.time_tagger import QPTimeTagger

backlash = 0.3
wait_after_move = 0.3

mxi = ThorlabsKcubeDC(thorlabs_x_serial, backlash=backlash, wait_after_move=wait_after_move)
myi = ThorlabsKcubeStepper(thorlabs_y_serial, backlash=backlash, wait_after_move=wait_after_move)

zaber_ms = ZaberMotors(backlash=backlash, wait_after_move=wait_after_move)
mxs = zaber_ms.motors[1]
mys = zaber_ms.motors[0]


integration_time = 1
tt = QPTimeTagger(integration_time=integration_time, remote=True)
