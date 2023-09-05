import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from pianoq.misc.mplt import *

# dimension
N = 64
# TM of the thick random medium
T = unitary_group.rvs(N)
# The two output modes optimizing coincidences to
I1 = 0
I2 = 1
O1 = N//3
O2 = 2*N//3

N_phases = 20

# TODO: see scaling of efficiency with degree of control (SLM will be stuck on x% of the pixels). Maybe Klyshko will be linear and beacon squared?
# TODO: see scaling also with thickness / amount of memory. Simulate with help from here: https://www.nature.com/articles/nphys3373
def optimize_beacon():
    best_cost1 = 0
    best_S1 = np.zeros(N)
    all_costs1 = []

    for mode_index in range(N * 4):
        S = best_S1.copy()
        for phase in np.linspace(0, 2 * np.pi, N_phases):
            S[mode_index % N] = phase
            final_mat = T @ np.diag(np.exp(1j * S))
            complex_cost = final_mat[O1, O1]
            cost = np.abs(complex_cost) ** 2
            all_costs1.append(cost)
            if cost > best_cost1:
                best_cost1 = cost
                best_S1 = S.copy()

    best_cost2 = 0
    best_S2 = np.zeros(N)
    all_costs2 = []

    for mode_index in range(N * 4):
        S = best_S2.copy()
        for phase in np.linspace(0, 2 * np.pi, N_phases):
            S[mode_index % N] = phase
            final_mat = T @ np.diag(np.exp(1j * S))  # make sure order TS. Also that this shouldn't be dagger
            complex_cost = final_mat[O2, O2]
            cost = np.abs(complex_cost) ** 2
            all_costs2.append(cost)
            if cost > best_cost2:
                best_cost2 = cost
                best_S2 = S.copy()

    return (best_S1 + best_S2)/2, [all_costs1, all_costs2]


def optimize_klyshko():
    best_cost = 0
    best_S = np.zeros(N)
    all_costs = []

    for mode_index in range(N*4):
        S = best_S.copy()
        for phase in np.linspace(0, 2*np.pi, N_phases):
            S[mode_index % N] = phase
            final_mat = T @ np.diag(np.exp(1j*S))**2 @ T.transpose().conjugate()
            complex_cost = final_mat[O1, O2]
            cost = np.abs(complex_cost)**2
            all_costs.append(cost)
            if cost > best_cost:
                best_cost = cost
                best_S = S.copy()
        # print(f'iter: {mode_index}, cost={best_cost}')

    return best_S, all_costs


def get_optimal_beacon_conj(inout1=True):
    if inout1:
        O = O1
        I = I1
    else:
        O = O2
        I = I2

    # returns angles (0, 2*pi)
    desired_vec = np.zeros(N, complex)
    desired_vec[O] = 1
    at_slm = T.transpose().conjugate() @ desired_vec  # dagger
    angles_from_out = np.angle(at_slm)

    in_vec = np.zeros(N, complex)
    in_vec[I] = 1
    in_vec = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(in_vec)))
    angles_from_in = np.angle(in_vec)

    S = -(angles_from_in + angles_from_out)  # TODO: this doesn't work well. Please understand it.

    return S % (2*np.pi)


def prop_beacon(S, in1=True):
    if in1:
        I = I1
    else:
        I = I2

    in_vec = np.zeros(N, complex)
    in_vec[I] = 1
    in_vec = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(in_vec)))

    out = T @ np.diag(np.exp(1j*S)) @ in_vec

    return out


def prop_klyshko(S):
    vec_from_O1 = np.zeros(N, complex)
    vec_from_O1[O2] = 1  # TODO: there is something here with O1/O2 that I need to understand better
    out = T @ np.diag(np.exp(1j*S))**2 @ T.transpose().conjugate() @ vec_from_O1
    return out


def get_optimal_klyshko_conj():
    """
        Returns phases to put on SLM (0, 2*pi)
        Explanation:
        (TS^2T^dag)_ij = T_ik S^2_kl T^*_jl
        ij = O1,O2 -> T_O1k T^*_O2l S^2_kl
        -> s_kl = (T_O1k T^*_O2l)^*
        s digonal -> k=l
    """
    S = np.zeros(N, complex)
    for k in range(N):
        S[k] = T[O1, k].conjugate() * T[O2, k]

    # dividing by 2 because we go twice on the SLM in Klyshko picture
    return np.angle(S) / 2


def plot_res(S, all_costs, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    final_mat = T @ np.diag(np.exp(1j*S))**2 @ T.transpose().conjugate()

    X_MARKER_COLOR = '#929591'
    X_MARKER_EDGEWITDH = 1.5

    axes[0].set_title('original $T*T^t$')
    imm = axes[0].imshow(np.abs(T @ T.transpose())**2)
    axes[0].plot(O2, O1, '+', markeredgecolor=X_MARKER_COLOR, markersize=11, markeredgewidth=X_MARKER_EDGEWITDH)
    axes[0].figure.colorbar(imm, ax=axes[0])

    axes[1].set_title('optimized $T*S^2*T^t$')
    imm = axes[1].imshow(np.abs(final_mat)**2)
    axes[1].plot(O2, O1, '+', markeredgecolor=X_MARKER_COLOR, markersize=11, markeredgewidth=X_MARKER_EDGEWITDH)
    axes[1].figure.colorbar(imm, ax=axes[1])
    axes[1].figure.show()

    print(S)
    plt.show(block=False)


if __name__ == "__main__":
    """
        SLM is in real basis, also the diffuser is in real basis, and also measurement. 
        The input state should be in Fourier basis, so the SLM will do something nontrivial. 
        
        In Klyshko: Out1 -> U^dag -> slm -> F -> mirror -> F^-1 -> slm -> U -> Out2, so we can just forget the Fourier, 
        everything is in real basis, so we just want to enhance the total matrix at [O1, O2]
        
        In beacon: I1 -> Fourier -> SLM -> U -> O1, and here we want to have only at O1. So there isn't a single index 
        to enhance (since we have many inputs), so analytically I feel a bit stuck. I guess I want the phases at SLM 
        plane to be the back-prop from O1, so take the phases from the backprop + phases from input, and conjugate that. 
    """
    # S, all_costs = optimize_klyshko()
    # fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    # plot_res(S, all_costs, axes)

    # S2, all_costs = optimize_beacon()
    # fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    # plot_res(S2, all_costs, axes)
    pass
