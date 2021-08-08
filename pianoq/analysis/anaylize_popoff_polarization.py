import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from pianoq.results.PopoffPolarizationRotationResult import PopoffPolarizationRotationResult

PATH = "../data/popoff_polarization_data.npz"


def TM_ratios_figures():
    pop = PopoffPolarizationRotationResult(path=PATH)
    pop._initialize(method='TM')
    # pop._initialize(method='pixel')

    # pop.show_mode(10)
    pop.show_TM(pop.TM_modes[pop.index_dx0])
    # pop.show_all_polarizations_ratios()
    # pop.show_polarizations_ratios_per_mode(range(0, 55, 2), logscale=True, legend=True)
    # pop.show_polarizations_ratios_bar_plots([0, 20, 30, 40])
    # pop.show_mixing_of_mode(pop.TM_modes[5], 10)


def get_P(A):
    """ The eigvecs are the P matrix that satisfies A = P@D@P^-1"""
    eig_vals, eig_vecs = la.eig(A)
    return eig_vecs


def get_all_2by2s(TM):
    """ return [55X2X2]"""
    Nmodes = TM.shape[0] // 2
    assert Nmodes == 55

    all = []

    for i in range(Nmodes):
        A = np.array([
                np.array([TM[i, i],         TM[i, i+Nmodes]]),
                np.array([TM[i+Nmodes, i],  TM[i+Nmodes, i+Nmodes]]),
            ])

        all.append(A)

    return np.array(all)


def is_diag(A, threshold):
    """ Check that sum of abs of off diagonals is lower than some threshold """
    return np.abs(np.fliplr(A)).trace() / 2 < threshold


def find_P_amount(As):
    """
    find amount of transformations P needed to diagonalize all the 2X2 matrices.
    This is similar to the schmidt number, or the amount of entanglement, or amount of
    mode-dependant polarization rotation.
    """

    # TODO: Threshold per matrix and not global
    threshold = np.abs(As).mean() * 0.5
    n = 0
    while len(As) > 0:
        A0 = As[0]
        P = get_P(A0)
        Bs = [la.inv(P) @ A @ P for A in As]
        Cs = []
        for i, B in enumerate(Bs):
            if not is_diag(B, threshold):
               Cs.append(As[i])

        As = Cs
        n += 1

    return n


def schmidt(TM_index):
    pop = PopoffPolarizationRotationResult(path=PATH)
    pop._initialize(method='TM')
    TM = pop.TM_modes[TM_index]
    As = get_all_2by2s(TM)
    schmidt_num = find_P_amount(As)
    print(f'Schmidt number for dx={pop.dxs[TM_index]} is: {schmidt_num}')


if __name__ == "__main__":
    # TM_ratios_figures()
    for i in range(40):
        schmidt(i)
    plt.show()
