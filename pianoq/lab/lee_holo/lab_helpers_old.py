import numpy as np
from pianoq.lab.lee_holo.generate_lee import orthogonal_lee
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LabResults:
    def __init__(self, target, hologram, camera_image, background_image, fourier_crop_slice):
        self.target = target
        self.hologram = hologram
        self.camera_image = camera_image
        self.background_image = background_image
        self.fourier_crop_slice = fourier_crop_slice

    def get_reconstructed_field(self, image):
        E_k = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image)))
        cropped = E_k[self.fourier_crop_slice]
        final_field = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(cropped)))
        return final_field

    def get_corrected_field(self):
        signal_field = self.get_reconstructed_field(self.camera_image)
        background_field = self.get_reconstructed_field(self.background_image)
        return signal_field * np.conjugate(background_field)

    def plot_fourier_space(self, vmax=1e10):
        E_k = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.camera_image)))
        
        fig, ax = plt.subplots()
        ax.imshow(np.abs(E_k)**2, vmax=vmax)
        
        y_slice, x_slice = self.fourier_crop_slice
        rect = patches.Rectangle(
            (x_slice.start, y_slice.start), 
            x_slice.stop - x_slice.start, 
            y_slice.stop - y_slice.start, 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.set_title("Fourier Space with Crop Region")
        plt.show()

    def save(self, filepath):
        y_slice, x_slice = self.fourier_crop_slice
        np.savez(
            filepath,
            target=self.target,
            hologram=self.hologram,
            camera_image=self.camera_image,
            background_image=self.background_image,
            fourier_crop_y_start=y_slice.start,
            fourier_crop_y_stop=y_slice.stop,
            fourier_crop_x_start=x_slice.start,
            fourier_crop_x_stop=x_slice.stop,
        )

    @classmethod
    def load(cls, filepath):
        data = np.load(filepath, allow_pickle=True)
        fourier_crop_slice = np.s_[
            int(data['fourier_crop_y_start']):int(data['fourier_crop_y_stop']),
            int(data['fourier_crop_x_start']):int(data['fourier_crop_x_stop'])
        ]
        return cls(
            target=data['target'],
            hologram=data['hologram'],
            camera_image=data['camera_image'],
            background_image=data['background_image'],
            fourier_crop_slice=fourier_crop_slice
        )

class LabMeasure:
    def __init__(self, dmd, cam, carrier_f, duty_cycle, fourier_crop_slice):
        self.dmd = dmd
        self.cam = cam
        self.carrier_f = carrier_f
        self.duty_cycle = duty_cycle
        self.fourier_crop_slice = fourier_crop_slice
        self.background_measurement_image = None

    def measure_background(self):
        target = np.ones((self.dmd.Ny, self.dmd.Nx))
        hologram, _ = orthogonal_lee(target, carrier_f=self.carrier_f, duty_cycle=self.duty_cycle)
        self.dmd.set_image(hologram.astype(bool))
        self.background_measurement_image = self.cam.get_image()

    def run_experiment(self, targets):
        if self.background_measurement_image is None:
            self.measure_background()

        results = []
        for target in targets:
            hologram, _ = orthogonal_lee(target, carrier_f=self.carrier_f, duty_cycle=self.duty_cycle)
            self.dmd.set_image(hologram.astype(bool))
            im = self.cam.get_image()
            
            result = LabResults(
                target=target,
                hologram=hologram,
                camera_image=im,
                background_image=self.background_measurement_image,
                fourier_crop_slice=self.fourier_crop_slice
            )
            results.append(result)
        return results
