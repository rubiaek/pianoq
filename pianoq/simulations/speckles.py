import numpy as np
import cv2
from numpy import fft


def get_speckles(N=2**11, n=20):
    X = np.arange(-N//2, N//2)
    Y = np.arange(-N // 2, N // 2)
    xx, yy = np.meshgrid(X, Y)
    g = np.exp(-(xx**2+yy**2)/(2*(N//30)**2))

    A = np.random.rand(N//n, N//n)*2*np.pi
    A = cv2.resize(A, (N,N), interpolation=cv2.INTER_AREA)
    A = np.exp(1j*A)

    g2 = g*A

    speckles = fft.fftshift(fft.fft2(fft.fftshift(g2)))

    return np.abs(speckles)**2


def contrast(im):
    return im.std() / im.mean()


def contrast_different_modes():
    b = get_speckles()
    for i in range(1, 20):
        cont = contrast(b[980:1060, 960:1060])
        print(f'i: {i}, expected: {1/np.sqrt(i):3f}, actuall: {cont:3f}')
        b += get_speckles()
