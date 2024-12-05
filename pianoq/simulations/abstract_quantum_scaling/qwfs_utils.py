from pianoq.simulations.abstract_quantum_scaling.qwfs_result import QWFSResult
from pianoq.simulations.abstract_quantum_scaling.qwfs_simulation import QWFSSimulation
import numpy as np
import matplotlib.pyplot as plt


def get_slm3_intensities(res, T_method='gaus_iid'):
    # This is relevant only for SLM3, and there we use only BFGS currently...
    alg = 'L-BFGS-B'
    config = 'SLM3'
    # T_method = 'unitary'
    try_nos = range(res.N_tries)

    I_outs = []
    I_middles = []
    I_focuss = []
    # for try_no in [2]:
    for try_no in try_nos:
        alg_ind = np.where(res.algos == alg)[0]
        conf_ind = np.where(res.configs == config)[0]
        T_method_ind = np.where(res.T_methods == T_method)[0]

        T_ind = res.N_T_methods * try_no + T_method_ind
        T = res.Ts[T_ind].squeeze()
        slm_phases = res.best_phases[T_method_ind, conf_ind, try_no, alg_ind].squeeze()
        N = len(slm_phases)

        sim = QWFSSimulation(N=N)
        sim.T = T
        sim.slm_phases = np.exp(1j * slm_phases)
        sim.config = config

        I_middle = np.abs(sim.T.transpose() @ (sim.slm_phases * sim.v_in)) ** 2
        I_out = np.abs(sim.propagate()) ** 2
        I_outs.append(I_out)
        I_middles.append(I_middle)
        I_focuss.append(res.results[T_method_ind, conf_ind, try_no, alg_ind].squeeze())

    I_outs = np.array(I_outs)
    I_middles = np.array(I_middles)
    I_focuss = np.array(I_focuss)

    return I_outs, I_middles, I_focuss

def get_slm1_intensities(res, config='SLM1-only-T', T_method='gaus_iid', alg='L-BFGS-B'):
    try_nos = range(res.N_tries)

    I_outs = []
    I_focuss = []
    # for try_no in [2]:
    for try_no in try_nos:
        alg_ind = np.where(res.algos == alg)[0]
        conf_ind = np.where(res.configs == config)[0]
        T_method_ind = np.where(res.T_methods == T_method)[0]

        T_ind = res.N_T_methods * try_no + T_method_ind
        T = res.Ts[T_ind].squeeze()
        slm_phases = res.best_phases[T_method_ind, conf_ind, try_no, alg_ind].squeeze()
        N = len(slm_phases)

        sim = QWFSSimulation(N=N)
        sim.T = T
        sim.slm_phases = np.exp(1j * slm_phases)
        sim.config = config

        I_out = np.abs(sim.propagate()) ** 2
        I_outs.append(I_out)
        I_focuss.append(res.results[T_method_ind, conf_ind, try_no, alg_ind].squeeze())

    I_outs = np.array(I_outs)
    I_focuss = np.array(I_focuss)

    return I_outs, I_focuss


def show_tot_energy_at_planes(I_middles, I_outs):
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))

    for i, ax in enumerate(axes):
        if i == 0:
            Is = I_middles.sum(axis=1)
            title = '$I_{tot}$ at Crystal plane'
        elif i == 1:
            title = '$I_{tot}$ at Output plane'
            Is = I_outs.sum(axis=1)
        elif i == 2:
            title = r'$I_{out}/I_{crystal}^2$'
            Is = I_outs.sum(axis=1) / I_middles.sum(axis=1) ** 2
        # Create the histogram
        ax.hist(Is, bins=10, edgecolor='black', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Intensity Values')
        ax.set_ylabel('Frequency')

        # Add statistical lines
        ax.axvline(np.mean(Is), color='red', linestyle='dashed', linewidth=2,
                   label=f'Mean: {np.mean(Is):.2f}')
        ax.axvline(np.median(Is), color='green', linestyle='dashed', linewidth=2,
                   label=f'Median: {np.median(Is):.2f}')

        ax.legend()
        ax.grid(True, alpha=0.3)

        # Adjust layout and display
        plt.tight_layout()


def show_I_out_I_middle(I_out, I_middle):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(I_out, label='I_out')
    axes[0].plot(I_middle, label='I_middle')
    # ax.set_ylim([0, 0.05])
    axes[0].legend()
    axes[1].plot(I_out, label='output plane')
    axes[1].plot(I_middle, label='crystal plane')
    axes[1].set_ylim([0, 0.05])
    axes[1].legend()
    axes[0].set_title('see peek')
    axes[1].set_title('zoom in on fluctuations')
    fig.suptitle('Intensity distribution')


def get_output_random_phases(config='SLM3', T_method='gaus_iid', N=1000):
    sim = QWFSSimulation(N=256)
    sim.config = config
    sim.T_method = T_method
    sim.reset_T()
    Is = []
    for i in range(N):
        random_phases = np.random.uniform(0, 2*np.pi, sim.N)
        sim.slm_phases = np.exp(1j*random_phases)
        # sim.slm_phases = np.exp(1j*slm_phases)
        v_out = sim.propagate()
        I_out = np.abs(v_out)**2
        Is.append(I_out.sum())
    return Is


def show_hist_intensities(Is):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the histogram
    ax.hist(Is, bins=100, edgecolor='black', alpha=0.7)
    ax.set_title('Histogram of Intensities (Is)')
    ax.set_xlabel('Intensity Values')
    ax.set_ylabel('Frequency')

    # Add statistical lines
    ax.axvline(np.mean(Is), color='red', linestyle='dashed', linewidth=2,
               label=f'Mean: {np.mean(Is):.2f}')
    ax.axvline(np.median(Is), color='green', linestyle='dashed', linewidth=2,
               label=f'Median: {np.median(Is):.2f}')

    ax.text(0.02, 0.98, 'Sample Histogram\nMean: {:.3f}\nStd Dev: {:.3f}'.format(np.mean(Is), np.std(Is)),
            transform=ax.transAxes,  # Use axes coordinates
            verticalalignment='top',  # Align to the top
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()


def show_max_norm_Ts(N_modes=256, N_Ts=1000):
    sim = QWFSSimulation(N=N_modes)
    sim.config = 'SLM3'
    sim.T_method = 'gaus_iid'
    max_Is = []
    for i in range(N_Ts):
        sim.reset_T()
        max_I = (np.abs(sim.T) ** 2).sum(axis=0).max()
        max_Is.append(max_I)
        max_I = (np.abs(sim.T) ** 2).sum(axis=1).max()
        max_Is.append(max_I)
    max_Is = np.array(max_Is)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the histogram
    ax.hist(max_Is, bins=100, edgecolor='black', alpha=0.7)
    ax.set_title('Histogram of max norm of Ts')
    ax.set_xlabel('Intensity Values')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()


def show_effics(effics):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the histogram
    ax.hist(effics, bins=50, edgecolor='black', alpha=0.7)
    ax.set_title('Histogram of efficiency (I_out/I_out.sum()')
    ax.set_xlabel('Intensity Values')
    ax.set_ylabel('Frequency')

    # Add statistical lines
    ax.axvline(np.mean(effics), color='red', linestyle='dashed', linewidth=2,
               label=f'Mean: {np.mean(effics):.2f}')
    ax.axvline(np.median(effics), color='green', linestyle='dashed', linewidth=2,
               label=f'Median: {np.median(effics):.2f}')

    ax.text(0.02, 0.98, 'Sample Histogram\nMean: {:.3f}\nStd Dev: {:.3f}'.format(np.mean(effics), np.std(effics)),
            transform=ax.transAxes,  # Use axes coordinates
            verticalalignment='top',  # Align to the top
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()