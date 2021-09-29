from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.lab.piano_optimization import PianoOptimization
from pianoq.misc.borders import Borders
try:
    from pianoq.lab.thorlabs_motor import ThorlabsRotatingServoMotor
except (ImportError, NameError) as e:
    print('could not import ThorlabsRotatingServoMotor')
