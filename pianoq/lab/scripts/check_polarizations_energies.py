import numpy as np
from guizero import App, Text

from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.misc.consts import DEFAULT_BORDERS
from scipy import ndimage


def get_polarizations_ratio(im):
    cm_row, cm_col = ndimage.measurements.center_of_mass(im)
    cm_row, cm_col = int(cm_row), int(cm_col)
    pol1_energy = im[:, :cm_col].sum()
    pol2_energy = im[:, cm_col:].sum()
    return pol1_energy / pol2_energy


def different_configs():
    edac = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
    cam = VimbaCamera(2)
    cam.set_borders(DEFAULT_BORDERS)
    edac.set_amplitudes(0)

    im = cam.get_image()
    ratio = get_polarizations_ratio(im)
    print(f'ratio when piezos not pressing is: {ratio}')

    for i in range(50):
        amps = np.random.uniform(0, 1, 40)
        edac.set_amplitudes(amps)
        im = cam.get_image()
        ratio = get_polarizations_ratio(im)
        print(f'ratio with random piezos configuration is: {ratio:.3f}')

    cam.close()
    edac.close()


def empty_dac():
    edac = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
    cam = VimbaCamera(2)
    cam.set_borders(DEFAULT_BORDERS)
    edac.set_amplitudes(0)
    im = cam.get_image()
    ratio = get_polarizations_ratio(im)
    print(f'ratio when piezos not pressing is: {ratio}')
    cam.close()
    edac.close()


def continuous():
    edac = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
    cam = VimbaCamera(2)
    cam.set_borders(DEFAULT_BORDERS)
    edac.set_amplitudes(0)

    app = App(height=150, width=550)
    t = Text(app, text=1, size=50)

    def callback():
        # amps = np.random.uniform(0, 1, 40)
        # edac.set_amplitudes(amps)
        im = cam.get_image()
        ratio = get_polarizations_ratio(im)
        t.value = f'Polarization Ratio\n{ratio:.3f}'

    t.repeat(1000, callback)
    app.display()

    cam.close()
    edac.close()


if __name__ == "__main__":
    # empty_dac()
    # different_configs()
    continuous()
