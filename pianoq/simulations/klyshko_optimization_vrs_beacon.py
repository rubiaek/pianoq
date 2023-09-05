import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from pianoq.misc.mplt import *

# dimension
N = 40
# TM of the thick random medium
T = unitary_group.rvs(N)
# The two output modes optimizing coincidences to
O1 = N//3
O2 = 2*N//3

N_phases = 20

# TODO: simply phase conjugate or something instead of searching
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


def plot_res(S, all_costs, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    final_mat = T @ np.diag(np.exp(1j*S))**2 @ T.transpose()

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
    # S, all_costs = optimize_klyshko()
    # fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    # plot_res(S, all_costs, axes)

    S2, all_costs = optimize_beacon()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    plot_res(S2, all_costs, axes)
