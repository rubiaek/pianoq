import numpy as np
import matplotlib.pyplot as plt
import colorsys


def colorize(z, beta=1.4, sat=1.0):
    """Return RGB where hue = phase, lightness = amplitude."""
    r   = np.abs(z)
    r  /= r.max() + 1e-12
    arg = np.angle(z)
    h   = (arg + np.pi) / (2*np.pi)      # 0…1
    l   = 1.0 - 1.0/(1.0 + r**beta)      # dark bg
    s   = sat
    rgb = np.vectorize(colorsys.hls_to_rgb)(h, l, s)
    return np.transpose(rgb, (1, 2, 0))


def make_oam_donut(Nx=512, Ny=512, ell=1, r0=0.35, sigma=0.06):
    x = np.linspace(-1, 1, Nx, endpoint=False)
    y = np.linspace(-1, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X, Y)

    amplitude = np.exp(- (R - r0)**2 / (2*sigma**2))
    phase     = np.mod(np.arctan2(Y, X) * ell, 2*np.pi)
    return amplitude * np.exp(1j * phase)


def orthogonal_lee(field,
                   carrier_f=(1/8, 0),  # (ν_x, ν_y) [cycles/pixel]
                   duty_cycle=0.5,
                   renorm=True):
    if renorm:
        field = field / (np.abs(field).max() + 1e-12)

    A   = np.abs(field)       # amplitude 0–1
    phi = np.angle(field)     # phase –π…π

    Ny, Nx = field.shape
    x_idx, y_idx = np.meshgrid(np.arange(Nx), np.arange(Ny))

    # --- Helpful shorthands ---
    fx, fy = carrier_f
    period_px = 1 / np.hypot(fx, fy)    # scalar period magnitude (pixels)

    # Shift (Lee 1978) – moves +1 order to Fourier centre after ramp
    norm_factor = 2 * np.hypot(fx, fy)
    x_shift = x_idx - fy / norm_factor
    y_shift = y_idx + fx / norm_factor

    # ---------------- Mask 1 : Phase (shift along ν⃗) ----------------
    # grating phase in *cycles* at every pixel
    grating_cycles = fx * x_shift + fy * y_shift          # ν⃗·r
    desired_cycles = phi / (2*np.pi)

    # distance from nearest transition of square wave (0.0…0.5)
    dist_to_edge = np.abs(
        np.mod(grating_cycles - desired_cycles - 0.5, 1.0) - 0.5
    )
    mask1 = dist_to_edge < duty_cycle / 2    # ON if inside bright bar

    # ---------------- Mask 2 : Amplitude (width across ν⃗) -------------
    # orthogonal coordinate (because  (-ν_y, ν_x)·ν⃗ = 0)
    orth_cycles = -fy * x_shift + fx * y_shift
    mask2 = np.mod(orth_cycles, 1.0) < A     # width = A·period

    hologram = np.logical_and(mask1, mask2).astype(float)
    return hologram, period_px


# ------------------------------------------------------------------
# 4.  4‑f propagation with centred pinhole
# ------------------------------------------------------------------
def fourier_filter(mask, carrier_f, aperture_radius):
    """FFT → crop circular aperture → IFFT."""
    Ny, Nx = mask.shape
    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    fx, fy = carrier_f

    # Centre the +1 order via a phase ramp
    ramp = np.exp(1j * 2*np.pi * (fx * x + fy * y))
    FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask * ramp)))

    # Circular low‑pass
    fx_ax = np.fft.fftshift(np.fft.fftfreq(Nx))
    fy_ax = np.fft.fftshift(np.fft.fftfreq(Ny))
    FX, FY = np.meshgrid(fx_ax, fy_ax)
    aperture = (FX**2 + FY**2) < aperture_radius**2

    shaped = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(FT * aperture)))
    return shaped, FT, aperture


def main(grating_period=8):
    N                 = 512          # simulation grid
    ELL               = 1            # OAM charge
    GRATING_PERIOD_PX = grating_period  # p (pixels)  ⇒ ν = 1/p cycles/pixel
    CARRIER_F         = (1/GRATING_PERIOD_PX, 0)  # along +x by default
    APERTURE_RADIUS   = 0.04         # fraction of freq. space to keep

    # ---- Physical parameters (SET THESE TO YOUR LAB) ----
    DMD_PIXEL_PITCH_M = 4.8e-6     # your DMD pixel = 4.8 µm
    WAVELENGTH_M      = 633e-9     # e.g., 532 nm (set to your laser)
    FOCAL_LENGTH_M    = 0.100      # e.g., f = 200 mm (set to your lens)


    # ------------------------------------------------------------------
    # 6.  Build hologram & propagate
    # ------------------------------------------------------------------
    target   = make_oam_donut(Nx=N, Ny=N, ell=ELL)
    hologram, _ = orthogonal_lee(target,
                                carrier_f=CARRIER_F,
                                duty_cycle=0.5)

    shaped, FT_holo, aperture = fourier_filter(hologram,
                                            CARRIER_F,
                                            APERTURE_RADIUS)

    fc_cycpx = float(np.hypot(*CARRIER_F))     # distance to shifted DC in cycles/pixel
    fr_cycpx = float(APERTURE_RADIUS)          # passband radius in cycles/pixel

    r_sep_mm = 1e3 * WAVELENGTH_M * FOCAL_LENGTH_M * (fc_cycpx / DMD_PIXEL_PITCH_M)
    r_pin_mm = 1e3 * WAVELENGTH_M * FOCAL_LENGTH_M * (fr_cycpx / DMD_PIXEL_PITCH_M)
    d_pin_mm = 2.0 * r_pin_mm
    margin_mm = r_sep_mm - r_pin_mm              # geometric clearance

    print("\n---- Fourier-plane (zeroth-order trick) ----")
    print(f"Pixel pitch p                : {DMD_PIXEL_PITCH_M*1e6:.2f} µm")
    print(f"Wavelength λ                 : {WAVELENGTH_M*1e9:.0f} nm")
    print(f"Fourier lens f               : {FOCAL_LENGTH_M*1e3:.0f} mm")
    print(f"|carrier|  f_c               : {fc_cycpx:.4f} cycles/pixel")
    print(f"aperture  f_r                : {fr_cycpx:.4f} cycles/pixel")
    print(f"Nearest contaminant radius   : r_sep ≈ {r_sep_mm:.3f} mm (shifted DC)")
    print(f"Pinhole radius at DC         : r_pin ≈ {r_pin_mm:.3f} mm  ⇒  diameter ≈ {d_pin_mm:.3f} mm")
    print(f"Centricity slack (geom.)     : ≤ {margin_mm:.3f} mm before overlap")
    if fr_cycpx >= 0.5*fc_cycpx:
        print("WARNING: aperture too large; will overlap shifted DC. Reduce APERTURE_RADIUS.")
    elif fr_cycpx >= 0.4*fc_cycpx:
        print("Note: aperture fairly large; consider APERTURE_RADIUS ≤ 0.3–0.4·|carrier| for margin.")
    print("----------------------------------------------------------------\n")

    # ------------------------------------------------------------------
    # 7.  Visualisation
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].set_title('Target field')
    ax[0, 0].imshow(colorize(target))
    ax[0, 0].axis('off')

    ax[0, 1].set_title('Binary Lee hologram')
    ax[0, 1].imshow(hologram, cmap='gray')
    ax[0, 1].axis('off')

    ax[1, 0].set_title('|FT(hologram)|    (cyan = aperture)')
    ax[1, 0].imshow(np.log10(np.abs(FT_holo) + 1e-3), cmap='inferno')
    ax[1, 0].contour(aperture, levels=[0.5], colors='cyan', linewidths=0.6)
    ax[1, 0].axis('off')

    ax[1, 1].set_title('Shaped field')
    ax[1, 1].imshow(colorize(shaped))
    ax[1, 1].axis('off')

    fig.tight_layout()
    plt.show()


    # ------------------------------------------------------------------
    # 8.  Fidelity metric
    # ------------------------------------------------------------------
    corr = np.vdot(target.ravel(), shaped.ravel()) /            np.sqrt(np.vdot(target, target) * np.vdot(shaped, shaped))
    print(f'Complex correlation |⟨target|shaped⟩| = {abs(corr):.3f}')
