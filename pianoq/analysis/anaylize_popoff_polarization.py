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
    off_diagonal = np.abs(np.fliplr(A)).trace()
    diagonal = np.abs(A.trace())

    return off_diagonal / diagonal < threshold


def find_P_amount(As, threshold):
    """
    find amount of transformations P needed to diagonalize all the 2X2 matrices.
    This is similar to the schmidt number, or the amount of entanglement, or amount of
    mode-dependant polarization rotation.
    """

    all_steps = []
    curr_step = np.ones(len(As))
    all_steps.append(curr_step.copy())

    n = 0
    while any(curr_step == 1.0):
        n += 1
        A0_index = np.random.choice(np.where(curr_step == 1.0)[0])
        A0 = As[A0_index]
        P = get_P(A0)
        Bs = [la.inv(P) @ A @ P for A in As]

        for i, B in enumerate(Bs):
            # The 0.5 game is to visualize that this mode changed in this iteration
            if curr_step[i] in [0.3, 0.7]:
                curr_step[i] = 0
            if is_diag(B, threshold=threshold):
                curr_step[i] = 0.3

        curr_step[A0_index] = 0.7

        all_steps.append(curr_step.copy())

    return n, np.array(all_steps)


def schmidt(TM_index, threshold):
    pop = PopoffPolarizationRotationResult(path=PATH)
    pop._initialize(method='TM')
    TM = pop.TM_modes[TM_index]
    As = get_all_2by2s(TM)
    schmidt_num, all_steps = find_P_amount(As, threshold=threshold)
    print(f'Schmidt number for dx={pop.dxs[TM_index]:.3f} is: {schmidt_num}')
    return pop.dxs[TM_index], schmidt_num, all_steps


def plot_schmidt_per_dxs(threshold=0.5):
    schmidt_nums = []
    dxs = []
    for i in range(40):
        dx, sn, _ = schmidt(i, threshold)
        dxs.append(dx)
        schmidt_nums.append(sn)

    fig, ax = plt.subplots()
    ax.plot(dxs, schmidt_nums)
    ax.set_xlabel(r'dx ($ \mu m $)')
    ax.set_ylabel(r'Polarization "Schmidt number"')
    ax.set_title(f'Shmidt approx. with threshold={threshold}')
    fig.show()


def plot_schmidt_process(TM_index, threshold=0.3):
    dx, sn, all_steps = schmidt(TM_index, threshold)
    fig, ax = plt.subplots()
    ax.imshow(all_steps, cmap='Greys', origin='lower')
    ax.set_xlabel(r'different modes')
    ax.set_ylabel(r'iterations')
    ax.set_title(f'Shmidt approx. with threshold={threshold}, dx={dx}')
    fig.show()


if __name__ == "__main__":
    # TM_ratios_figures()
    # plot_schmidt_per_dxs(0.3)
    plot_schmidt_process(5, 0.3)
    plt.show()
