import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

# Parameters to play
Nx = 1024
Ny = 1024
Dx = 3e-3
Dy = 3e-3
dx = Dx / Nx
dy = Dy / Ny
wl = 532e-9
sig = 50e-6
L = 50e-3
## grids

# X, Y range centered around 0
X = (np.arange(1, Nx + 1) - (Nx / 2 + 0.5)) * dx
Y = (np.arange(1, Ny + 1) - (Ny / 2 + 0.5)) * dy
XX, YY = np.meshgrid(X, Y)

# k-space grid
fs = 1 / (XX.max() - XX.min())
freq_x = fs * np.arange(-Nx // 2, Nx // 2)
fs = 1 / (YY.max() - YY.min())
freq_y = fs * np.arange(-Ny // 2, Ny // 2)
freq_XXs, freq_YYs = np.meshgrid(freq_x, freq_y)
light_k = 2 * np.pi / wl
k_xx = freq_XXs * 2 * np.pi
k_yy = freq_YYs * 2 * np.pi

k_z_sqr = light_k ** 2 - (k_xx ** 2 + k_yy ** 2)
# Remove all the negative component, as they represent evanescent waves, see Fourier Optics page 58
np.maximum(k_z_sqr, 0, out=k_z_sqr)
k_z = np.sqrt(k_z_sqr)

# Initial
E_gaus_init = np.exp(-(XX**2 + YY**2)/sig**2)

E_K = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_gaus_init)))
# Apply the transfer function of free-space, see Fourier Optics page 74
# normal forward motion is with + in the exponent
prop_mat = np.exp(+1j * k_z * L)
E_K *= prop_mat
E_out = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E_K)))

fig, axes = plt.subplots(1, 2)
imm = axes[0].imshow(np.abs(E_gaus_init)**2)
fig.colorbar(imm, ax=axes[0])
imm = axes[1].imshow(np.abs(E_out)**2)
fig.colorbar(imm, ax=axes[1])

plt.show()