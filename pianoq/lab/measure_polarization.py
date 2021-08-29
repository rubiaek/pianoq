from misc.calc_correlation import get_correlations_mask
from misc.consts import DEFAULT_BORDERS
from results.polarization_meas_result import PolarizationMeasResult

LOGS_DIR = 'C:\\temp'


class MeasurePolarization(object):
    def __init__(self, exposure_time=900, saveto_path=None, roi=None):
        self.cam = VimbaCamera(2, exposure_time=exposure_time)
        roi = roi or DEFAULT_BORDERS
        self.cam.set_borders(DEFAULT_BORDERS)

        self.res = PolarizationMeasResult()
        self.res.exposure_time = exposure_time
        self.res.roi = roi
        self.res.mask_of_interest = get_correlations_mask()

        self.saveto_path = saveto_path


    def run(self):
        # So H and V get to wollaston prism, and WPs won't bother
        print("Make sure the fast axis of both QWP and HWP is on 0 degrees")
        input()
        self.res.meas1 = self.cam.get_image()

        # So +-45 will turb to H and V, and QWP won't bother
        print("Make sure the fast axis of QWP is on 45 degrees and HWP is on 22.5 degrees")
        input()
        self.res.meas2 = self.cam.get_image()

        # So QWP will change R,L to +-45m and then HWP will turn them to H, V
        print("Make sure the fast axis of QWP is on 0 degrees and HWP is on 22.5 degrees")
        input()
        self.res.meas3 = self.cam.get_image()


    def _save_result(self):
        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}.polm"
        self.res.saveto(saveto_path)

    def close(self):
        self.cam.close()


if __name__ == "__main__":
    mp = MeasurePolarization()
    mp.run()
    mp.close()
