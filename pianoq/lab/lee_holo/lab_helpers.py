import numpy as np
from pianoq.lab.lee_holo.generate_lee import orthogonal_lee
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from datetime import datetime
from matplotlib.animation import FuncAnimation

class LabResults:
    def __init__(self, target, hologram, camera_image, background_image, fourier_crop_slice, metadata=None):
        self.target = target
        self.hologram = hologram
        self.camera_image = camera_image
        self.background_image = background_image
        self.fourier_crop_slice = fourier_crop_slice
        self.metadata = metadata if metadata is not None else {}

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
            metadata=self.metadata,
        )

    @classmethod
    def load(cls, filepath):
        data = np.load(filepath, allow_pickle=True)
        fourier_crop_slice = np.s_[
            int(data['fourier_crop_y_start']):int(data['fourier_crop_y_stop']),
            int(data['fourier_crop_x_start']):int(data['fourier_crop_x_stop'])
        ]
        # Metadata is saved as a 0-d array, so we extract it with .item()
        metadata = data.get('metadata', None)
        if metadata is not None:
            metadata = metadata.item()
            
        return cls(
            target=data['target'],
            hologram=data['hologram'],
            camera_image=data['camera_image'],
            background_image=data['background_image'],
            fourier_crop_slice=fourier_crop_slice,
            metadata=metadata,
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
        print("Background measured.")

    def run_experiment(self, targets_dict, experiment_metadata=None, use_lee_cancellation=True):
        if self.background_measurement_image is None:
            self.measure_background()

        results = {}
        for name, target in targets_dict.items():
            if use_lee_cancellation:
                hologram, _ = orthogonal_lee(target*np.conjugate(self.background_measurement_image), carrier_f=self.carrier_f, duty_cycle=self.duty_cycle)
            else:
                hologram, _ = orthogonal_lee(target, carrier_f=self.carrier_f, duty_cycle=self.duty_cycle)
            self.dmd.set_image(hologram.astype(bool))
            im = self.cam.get_image()
            
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "name": name
            }
            if experiment_metadata:
                metadata.update(experiment_metadata)

            result = LabResults(
                target=target,
                hologram=hologram,
                camera_image=im,
                background_image=self.background_measurement_image,
                fourier_crop_slice=self.fourier_crop_slice,
                metadata=metadata
            )
            results[name] = result
        return results
    
    def save_results(self, results_dict, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        for name, result in results_dict.items():
            filepath = os.path.join(folder_path, f"{name}.npz")
            result.save(filepath)
            print(f"Saved result to {filepath}")

    def live_view(self, interval=100):
        fig, ax = plt.subplots()
        imm = self.cam.get_image()
        im = ax.imshow(imm)
        title = fig.suptitle(f'Max pixel: {imm.max():.3f}', fontsize=16)

        def update(i):
            imm = self.cam.get_image()
            im.set_data(imm)
            im.set_clim(imm.min(), imm.max()) # necessary to update color scale
            title.set_text(f'Max pixel: {imm.max():.3f}')
            ax.set_title('%03d' % i)

        ani = FuncAnimation(fig, update, interval=interval, cache_frame_data=False)

        def close(event):
            if event.key == 'q':
                plt.close(event.canvas.figure)
        
        fig.canvas.mpl_connect("key_press_event", close)
        plt.show()
