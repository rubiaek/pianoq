from pianoq.lab import VimbaCamera
from pianoq.lab.elliptec_stage import ElliptecMotor
from pianoq.lab.thorlabs_motor import ThorlabsRotatingServoMotor
from pianoq.misc.consts import DEFAULT_CAM_NO


class GlobalRotationOptimization(object):
    def __init__(self):
        self.hwm = ThorlabsRotatingServoMotor()
        self.qwm = ElliptecMotor()  # quicker
        self.cam = VimbaCamera(DEFAULT_CAM_NO)

    def run(self):
        
        # Run over angles with both WPs and find best position for self.cost_function
        pass

    def cost_function_H(self):
        # get the most light into H
        pass

    def close(self):
        self.hwm.close()
        self.qwm.close()


if __name__ == "__main__":
    gro = GlobalRotationOptimization()
    gro.run()
    gro.close()


