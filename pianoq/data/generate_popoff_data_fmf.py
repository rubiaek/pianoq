import os
import pickle
import numpy as np
from pianoq.results.popoff_prx_result import PopoffPRXResult

data_folder = r'G:\My Drive\Projects\Quantum Piano\Random\Rodrigo FMF TMs'


def main():
    tms_path = os.path.join(data_folder, 'TMs_mod.npy')
    TM_modes = np.load(tms_path)
    res = PopoffPRXResult()

    res.TM_modes = TM_modes  # Transmission matrices in the mode basis for different dx values
    res.index_dx0 = 20
    step = 2
    res.dxs = step * (np.arange(TM_modes.shape[0]) - res.index_dx0)  # \mu m

    modes_data_path = os.path.join(data_folder, 'modes_hd')
    modes_data = pickle.load(open(modes_data_path, 'rb'), encoding='latin1')
    res.modes_out = np.array(modes_data['profiles'])

    res.modes_out_full = None
    res.L = None  # L value of the i'th mode
    res.M = None  # M value of the i'th mode
    res.path = './popoff_polarization_data_fmf.npz'

    res.saveto(res.path)


if __name__ == "__main__":
    main()
