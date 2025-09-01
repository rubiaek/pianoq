import matplotlib.pyplot as plt
import numpy as np
import requests
from functools import wraps
from matplotlib.animation import FuncAnimation
from colorsys import hls_to_rgb
import threading
import io
import sys
import time
from scipy import ndimage
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit

from pianoq.misc.mplt import my_mesh
from pianoq_results.scan_result import ScanResult


# https://stackoverflow.com/questions/44985966/managing-dynamic-plotting-in-matplotlib-animation-module
class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min=mini
        self.max=maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self,self.fig, self.func, frames=self.play(),
                               init_func=init_func, fargs=fargs,
                               save_count=save_count, **kwargs )

    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()
    def backward(self, event=None):
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i+=1
        elif self.i == self.max and not self.forwards:
            self.i-=1
        self.func(self.i)
        self.fig.canvas.draw_idle()


def colorize(z, theme='dark', saturation=1., beta=1.4, transparent=False, alpha=1., max_threshold=1.):
    r = np.abs(z)
    r /= max_threshold * np.max(np.abs(r))
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1. / (1. + r ** beta) if theme == 'white' else 1. - 1. / (1. + r ** beta)
    s = saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    if transparent:
        a = 1. - np.sum(c ** 2, axis=-1) / 3
        alpha_channel = a[..., None] ** alpha
        return np.concatenate([c, alpha_channel], axis=-1)
    else:
        return c


def retry_if_exception(ex=Exception, max_retries=3):
    def outer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert max_retries > 0
            x = max_retries
            while x:
                try:
                    return func(*args, **kwargs)
                except ex:
                    print(f'Failed. {x} tries remain')
                    x -= 1
                    if x < 1:
                        raise
        return wrapper
    return outer


from IPython.display import display, HTML


class ThreadWithPrintCapture(threading.Thread):
    def __init__(self, target, args=(), kwargs=None):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.output = io.StringIO()

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = self.output
        try:
            self.target(*self.args, **self.kwargs)
        finally:
            sys.stdout = old_stdout


def display_thread_output(thread):
    while thread.is_alive():
        output = thread.output.getvalue()
        if output:
            # display(HTML(f"<pre>{output}</pre>"))
            thread.output.truncate(0)
            thread.output.seek(0)
        time.sleep(0.1)

    # Display any remaining output
    output = thread.output.getvalue()
    if output:
        # display(HTML(f"<pre>{output}</pre>"))
        pass


def run_in_thread(func, *args, **kwargs):
    thread = ThreadWithPrintCapture(target=func, args=args, kwargs=kwargs)
    thread.start()
    display_thread = threading.Thread(target=display_thread_output, args=(thread,))
    display_thread.start()
    return thread


def run_in_thread_simple(func, *args, **kwargs):
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.start()
    return thread


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y):
    x, y = xy
    return amplitude * np.exp(-(((x-x0)/sigma_x)**2 + ((y-y0)/sigma_y)**2)/2)


def detect_gaussian_spots_subpixel(scan, X, Y, num_spots=5, min_distance=5, window_size=4, sort_top_to_bottom=True, get_amps=False):
    """
    Claude code
    Detect Gaussian spots in a 2D scan and estimate their coordinates with sub-pixel resolution.

    Parameters:
    scan (numpy.ndarray): 2D array representing the scan
    X (numpy.ndarray): 1D array of x-coordinates (assumed to be equally spaced)
    Y (numpy.ndarray): 1D array of y-coordinates (assumed to be equally spaced)
    num_spots (int): Number of spots to detect (default: 5)
    min_distance (int): Minimum distance between spots in pixels (default: 5)
    window_size (int): Size of the window for center of mass calculation (default: 4)

    Returns:
    numpy.ndarray: Array of shape (num_spots, 2) containing the (x, y) coordinates of detected spots
    """
    # Apply Gaussian filter to reduce noise
    smoothed_scan = ndimage.gaussian_filter(scan, sigma=1)
    # smoothed_scan = scan

    # Find local maxima
    coordinates = peak_local_max(smoothed_scan, num_peaks=num_spots, min_distance=min_distance, exclude_border=False)

    # Refine coordinates with sub-pixel resolution
    refined_coordinates = []
    half_window = window_size // 2

    assert (Y[:-1] < Y[1:]).all(), 'We assume Y to be ascending'
    assert (X[:-1] < X[1:]).all(), 'We assume X to be ascending'

    amps = []
    for y, x in coordinates:
        y_start = max(0, y - half_window)
        y_end = min(scan.shape[0], y + half_window + 1)
        x_start = max(0, x - half_window)
        x_end = min(scan.shape[1], x + half_window + 1)

        # Use the original (unfiltered) scan data for fitting
        window = scan[y_start:y_end, x_start:x_end]

        x_window, y_window = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))

        initial_guess = [window.max(), x, y, 1, 1]
        try:
            popt, _ = curve_fit(gaussian_2d, (x_window.ravel(), y_window.ravel()), window.ravel(), p0=initial_guess)
            ampl, x0, y0, sig_x, sig_y = popt
            refined_x = X[0] + x0 * (X[1] - X[0])
            refined_y = Y[0] + y0 * (Y[1] - Y[0])
            refined_coordinates.append((refined_x, refined_y))
        except RuntimeError:
            print(f"Fitting failed for spot at ({x}, {y}). Using original coordinates.")
            refined_coordinates.append((X[x], Y[y]))
            ampl = window.max()
        amps.append(ampl)

    sorted_coordinates = sorted(refined_coordinates, key=lambda c: c[1], reverse=not sort_top_to_bottom)

    if get_amps:
        return np.array(amps)

    return np.array(sorted_coordinates)


def get_locs_from_scan(scan_path, single_num=1, num_spots=5, show=False):
    scan = ScanResult(scan_path)
    assert single_num in [1, 2]
    s = scan.single1s if single_num == 1 else scan.single2s
    locs = detect_gaussian_spots_subpixel(s, scan.X, scan.Y[::-1], num_spots=num_spots,
                                          sort_top_to_bottom=True if single_num == 1 else False,
                                          get_amps=False)
    if show:
        fig, ax = plt.subplots()
        my_mesh(scan.X, scan.Y, s, ax=ax)
        ax.plot(locs[:, 0], locs[:, 1], marker='+', linestyle='none')
        fig.show()

    return locs

def figshow(fig):
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return fig
        elif shell == 'TerminalInteractiveShell':  # IPython terminal
            return fig.show()
    except NameError:  # Regular Python shell
        return fig.show()


BOT_TOKEN = '7897393437:AAGlhCVyo_aawMfXGaI-qatAwZab6Nr-RUU'
CHAT_ID = '476169345'
from io import BytesIO

def send_telegram_message(message, bot_token=BOT_TOKEN, chat_id=CHAT_ID):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    response = requests.post(url, json=payload)
    return response.json()


def send_telegram_photo(photo, caption=None, bot_token=BOT_TOKEN, chat_id=CHAT_ID):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"

    files = {'photo': photo}
    data = {'chat_id': chat_id}
    if caption:
        data['caption'] = caption

    response = requests.post(url, data=data, files=files)
    return response.json()


def send_telegram_fig(fig, caption=None, bot_token=BOT_TOKEN, chat_id=CHAT_ID):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return send_telegram_photo(('figure.png', buf, 'image/png'), caption)


import inspect
from functools import wraps

def set_last_called_str(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # call first; record only on success
        result = func(self, *args, **kwargs)

        sig   = inspect.signature(func)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        parts = [f"{k}={v!r}" for k, v in bound.arguments.items() if k != "self"]

        self.last_called_setter = f"{self.__class__.__name__}.{func.__name__}({', '.join(parts)})"
        return result
    return wrapper


def track_setters(cls):
    for name, attr in list(cls.__dict__.items()):
        if name.startswith("set_") and not name.startswith("set_image") and callable(attr):
            setattr(cls, name, set_last_called_str(attr))
    return cls