"""
This should be run from within the Analysis directory of https://github.com/wavefrontshaping/article_MMF_disorder
It is mainly copying from Amalysis_Deformation.ipnb to create an object I (ronen) like working with, the
PopoffPolarizationRotationResult object. Then from here on I'm independant of their code.
"""

import matplotlib

import numpy as np
import json
import os
from functions import colorize, complex_correlation, fidelity

# Make sure this import works... 
from PopoffPolarizationRotationResult import PopoffPolarizationRotationResult

data_folder = '../Data'

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

function = fidelity
# function = complex_correlation

# %% md

# 1. First load the params and data

# %% md

## Load parameters

# %%

path = os.path.sep.join([data_folder, 'param.json'])
with open(path, 'r') as f:
    param = json.load(f)
    print('Succesfully loaded parameters from .json file')

# %% md

## Load theoretical modes

# %%

path = os.path.sep.join([data_folder, 'conversion_matrices.npz'])

data_dict = np.load(path)
# list of the radial momenta of the modes
L = data_dict['L_modes']
# list of the angular orbital momebta of the modes
M = data_dict['M_modes']
modes_in = data_dict.f.modes_in
# create mode matrix for two polarizations
modes_in_full = np.kron(np.array([[1, 0], [0, 1]]), modes_in)
# number of input pixels
m = int(np.sqrt(modes_in.shape[1]))
modes_out = data_dict.f.modes_out
modes_out_full = np.kron(np.array([[1, 0], [0, 1]]), modes_out)
# number of output pixels
n = int(np.sqrt(modes_out.shape[1]))

# %% md

## Load the TMs

# %%

TM_modes = []
for i, pos in enumerate(param['Translation']['steps_pressure']):
    path = os.path.sep.join([data_folder, f'TM_modes_{str(pos)}.npz'])
    data_dict = np.load(path)
    TM_modes.append(data_dict.f.TM_corr_modes)
TM_modes = np.array(TM_modes)
Nmodes = TM_modes.shape[-1] // 2
print(Nmodes)

# %% md

# 2. Decorrelation of the TM

# %%

index = -1
index_dx0 = 17
index_reference = 5
TM_perturb = TM_modes[index]
TM_ref = TM_modes[index_reference]
TM_correlation = []

dx = (np.array(param['Translation']['steps_pressure']) - param['Translation']['steps_pressure'][index_dx0]) * 1e3
to_img = lambda x: colorize(x, beta=1.8, max_threshold=0.8)


# %%
def get_WS(H0, H1, H2):
    '''
    Get the Wigner-Smith operator from 3 matrices at Dx-dx, Dx and Dx+dx
    '''
    dH = (H2 - H0) * 0.5
    WS = -1j * np.linalg.pinv(H1) @ dH
    return WS


def get_WS_symm(H0, H1, H2):
    '''
    Get the Hermitian part of the Wigner-Smith operator from 3 matrices at Dx-dx, Dx and Dx+dx
    '''
    WS = get_WS(H0, H1, H2)
    WS_symm = WS + WS.T.conj()
    return WS_symm


# central deformation to estimate the WS operator
center_TM_index = 24
# step to evaluate the derivative
step = 4

WS_symm = get_WS_symm(TM_modes[center_TM_index - step],
                      TM_modes[center_TM_index],
                      TM_modes[center_TM_index + step])


s, U = np.linalg.eig(WS_symm)
# sort the eigenvalues
s = s[np.argsort(s)]
U = U[np.argsort(s), :]

# %%

def get_out_states(in_modes, i0, i1):
    '''
    Get the output states corresponding to the input state 'in_modes'
    for the deformations indexed by 'i0' and 'i1'.
    Return the output fields for the two polarizations for each deformation.
    '''
    output_state = TM_modes[i0] @ in_modes
    out_pix = modes_out_full.transpose() @ output_state
    out_pix_p1_0 = out_pix[:n ** 2].reshape([n] * 2)
    out_pix_p2_0 = out_pix[n ** 2:].reshape([n] * 2)

    output_state = TM_modes[i1] @ in_modes
    out_pix = modes_out_full.transpose() @ output_state
    out_pix_p1_f = out_pix[:n ** 2].reshape([n] * 2)
    out_pix_p2_f = out_pix[n ** 2:].reshape([n] * 2)

    return out_pix_p1_0, out_pix_p2_0, out_pix_p1_f, out_pix_p2_f


def get_correlation_out(in_modes, indexes, index_reference):
    '''
    Take as argument an input wavefront 'in_modes' (in the mode basis),
    a list of indexes 'indexes' and the reference index 'index_reference'.
    Returns a list of values representing the
    correlation of the output intensity
    for each deformation indexed by 'indexes'
    with the output intensity for the reference deformation
    indexed by 'index_reference'
    '''
    corr_vector = []
    f = lambda x: np.abs(x) ** 2
    for ind in indexes:
        out_pix_p1_0, out_pix_p2_0, out_pix_p1_f, out_pix_p2_f = get_out_states(in_modes, i0=index_reference, i1=ind)
        out_0 = np.concatenate([out_pix_p1_0, out_pix_p2_0])
        out_f = np.concatenate([out_pix_p1_f, out_pix_p2_f])
        corr_vector.append(complex_correlation(f(out_0), f(out_f)))
    return corr_vector

Nmodes = TM_modes[0].shape[0]


def create_ronen_object_with_data():
    res = PopoffPolarizationRotationResult()
    res.TM_modes = TM_modes
    res.dxs = dx
    res.index_dx0 = index_dx0
    res.modes_out = modes_out
    res.L = L
    res.M = M
    path = "C:\\temp\\popoff_polarization_data5.npz"
    res.saveto(path)


if __name__ == "__main__":
    print('foo')
    create_ronen_object_with_data()
