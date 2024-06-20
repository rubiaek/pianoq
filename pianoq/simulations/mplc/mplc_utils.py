import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def get_lens_mask_conf(conf, f):
    return get_lens_mask(Nx=conf['Nx'] * conf['size_factor'], Ny=conf['Ny'] * conf['size_factor'],
                         dx=conf['dx'], dy=conf['dy'], wl=conf['wavelength'], f=f)


def get_lens_mask(Nx, Ny, dx, dy, wl, f):
    X = np.arange(-Nx / 2, Nx / 2) * dx
    Y = np.arange(-Ny / 2, Ny / 2) * dy
    XX, YY = np.meshgrid(X, Y)
    k = 2 * np.pi / wl
    # important -i, assuming freespace is with +i
    mask = np.exp(-1j * (XX**2 + YY**2) * k / (2 * f))

    return mask


def show_field(E, figshow=True, active_slice=None, title=''):
    fig, ax = plt.subplots()
    imm = ax.imshow(np.abs(E)**2)
    ax.set_title(title)

    if active_slice:
        rows = active_slice[0]
        cols = active_slice[1]
        ax.add_patch(Rectangle((cols.start, rows.start), cols.stop - cols.start, rows.stop-rows.start,
                               facecolor="none", ec='k', lw=2))

    fig.colorbar(imm, ax=ax)
    if figshow:
        fig.show()


def downsample_with_mean(data, block_size):
    # Ugly code by chatgpt
    i, j = block_size
    N, M = data.shape

    # Calculate the number of full blocks in each dimension
    num_full_blocks_x = N // i
    num_full_blocks_y = M // j

    # Initialize the downsampled array
    downsampled = np.zeros((N, M))

    # Handle the main body with full blocks
    main_body = data[:num_full_blocks_x * i, :num_full_blocks_y * j]
    reshaped = main_body.reshape(num_full_blocks_x, i, num_full_blocks_y, j)
    block_means = reshaped.mean(axis=(1, 3))

    # Fill the main body of the downsampled array
    downsampled[:num_full_blocks_x * i, :num_full_blocks_y * j] = np.kron(block_means, np.ones((i, j)))

    # Handle the right edge
    if M % j != 0:
        right_edge_start_y = num_full_blocks_y * j
        for bx in range(num_full_blocks_x):
            start_x = bx * i
            end_x = start_x + i
            block = data[start_x:end_x, right_edge_start_y:M]
            block_mean = block.mean()
            downsampled[start_x:end_x, right_edge_start_y:M] = block_mean

    # Handle the bottom edge
    if N % i != 0:
        bottom_edge_start_x = num_full_blocks_x * i
        for by in range(num_full_blocks_y):
            start_y = by * j
            end_y = start_y + j
            block = data[bottom_edge_start_x:N, start_y:end_y]
            block_mean = block.mean()
            downsampled[bottom_edge_start_x:N, start_y:end_y] = block_mean

    # Handle the bottom-right corner
    if N % i != 0 and M % j != 0:
        bottom_right_block = data[bottom_edge_start_x:N, right_edge_start_y:M]
        block_mean = bottom_right_block.mean()
        downsampled[bottom_edge_start_x:N, right_edge_start_y:M] = block_mean

    return downsampled


def corr(A, B):
    return (np.abs(A.conj() * B)**2).sum()
