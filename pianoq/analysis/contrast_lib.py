import numpy as np
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.special import gamma
# import warnings
# warnings.filterwarnings("ignore")


def calc_contrast(V):
    contrast = V.std() / V.mean()
    N = V.size
    contrast_err = contrast * np.sqrt(1/(2*N-2) + (contrast**2)/N)
    return ufloat(contrast, contrast_err)

def contrast_to_N_modes(C):
    return 1/C**2

def calc_N_from_distribution(V):
    def gamma_pdf(I, N, I_0):
        ans = ((I**(N-1)) / (gamma(N)*I_0**N)) * np.exp(-I/I_0)
        ans[I < 0] = 0
        return ans

    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(V, bins=80, density=True)
    X = bins[:-1] + np.diff(bins) / 2  # middle of bins

    approx_c = calc_contrast(V)
    approx_N = contrast_to_N_modes(approx_c).nominal_value
    print(f'approx N: {approx_N}')
    popt, pcov = curve_fit(gamma_pdf, X, counts, bounds=(0.01, np.inf), p0=(approx_N, V.mean()))
    N, I_0 = popt
    N_err, I_0_err = np.sqrt(pcov.diagonal())

    XX = np.linspace(X[0], X[-1], 500)
    ax.plot(XX, gamma_pdf(XX, N, I_0), 'r-', label=f'fit to gamma: N={N:.1f}$\pm${N_err:.1f}, $ I_0 $={I_0:.1f}$\pm${I_0_err:.1f}')
    ax.fill_between(XX, gamma_pdf(XX, N+N_err, I_0), gamma_pdf(XX, N-N_err, I_0), color='r', alpha=0.3)
    ax.legend()
    ax.set_xlabel('coincidence counts')
    ax.set_ylabel('Probability')
    fig.show()
