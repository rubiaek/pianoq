# 2025-06-09 after some work by Elad. Putting here for ref.
# dmd_driver.py
import os
import numpy as np
import ajiledriver as aj
from PIL import Image
import cv2
from pianoq.misc.misc import track_setters


@track_setters
class DMD:
    """
    Minimal DMD driver for interactive work.

    Typical session
    ---------------
    >>> from dmd_driver import DMD
    >>> d = DMD(serial=123)           # auto-starts the 'live' sequence
    >>> d.set_grating(16)             # 16-pixel binary grating
    >>> d.set_image(np.random.randint(0, 2, d.shape, np.uint8))  # any 0/1 array
    >>> d.close()
    """

    # ---------------------------------------- setup
    def __init__(self, serial=None, frame_time_ms=1,              # 1 ms -> 1 kHz max if the board allows it
        ip="192.168.200.1", subnet="255.255.255.0", gateway="0.0.0.0", port=5005,
        interface=aj.USB3_INTERFACE_TYPE, timeout_ms=5_000, verbose=False):
        self.timeout_ms = timeout_ms
        self.seq_id = 1               # fixed, never recreated
        self.verbose = verbose 

        # --- open link -----------------------------------------------------
        self.hs = aj.HostSystem()
        self.hs.SetConnectionSettingsStr(ip, subnet, gateway, port)
        self.hs.SetCommunicationInterface(interface)
        if serial is not None:
            self.hs.SetUSB3DeviceNumber(serial)

        ret = self.hs.StartSystem()
        if ret != aj.ERROR_NONE:
            raise RuntimeError(f"HostSystem.StartSystem() failed → {ret}")
        self.log('Connected!')

        # --- hardware info -------------------------------------------------
        self.comp_idx = self._find_dmd_component()
        comp        = self.hs.GetProject().Components()[self.comp_idx]
        self.hwtype = comp.DeviceType().HardwareType()

        self.shape = (comp.NumRows(), comp.NumColumns())
        # self.shape = (1140, 912)      # (rows, cols) # Typical 
        
        # For convenience 
        self.Ny, self.Nx = self.shape
        self.X = np.arange(-self.Nx//2, self.Nx//2)
        self.Y = np.arange(-self.Ny//2, self.Ny//2)
        self.XX, self.YY = np.meshgrid(self.X, self.Y)
        self.R = np.sqrt(self.XX**2 + self.YY**2)

        self.cur_image = np.zeros(self.shape, np.uint8)
        self.last_called_setter = ''

        # --- build once-off project ---------------------------------------
        self._build_static_project(frame_time_ms)
        self.log('Built project!')

        # smaller of the two SDK options for “images only”
        self._loadopt_images_only = (
            getattr(aj, "LOAD_OPT_IMAGES_ONLY",
                    getattr(aj, "LOAD_OPT_UPDATE_IMAGES", 2))
        )

    def set_image_from_path(self, path):
        img = self._read_image(path)
        self._update_image(img)
        """
        Load any image from disk, threshold it to a 0/1 mask, resize
        to the DMD's resolution, and display it.
        """

    def set_image(self, bitmap):
        """
        Display a NumPy array (dtype uint8/bool, values 0 | 1) **or**
        an `aj.Image` already matching the DMD geometry.
        """
        self.cur_image = bitmap
        
        if isinstance(bitmap, np.ndarray):
            if bitmap.shape != self.shape:
                raise ValueError(f"bitmap shape {bitmap.shape} "
                                 f"≠ DMD shape {self.shape}")
            self._np_to_aj_image(bitmap, self._img)   # in-place overwrite
            self._update_image(self._img)
        elif isinstance(bitmap, aj.Image):
            # replace image ID 1 in the project
            bitmap.SetID(1)
            self.proj.AddImage(bitmap)                # overwrites same ID
            self._update_image(bitmap)
        else:
            raise TypeError("bitmap must be numpy array or aj.Image")

    def set_white(self):
        self.set_image(np.ones(self.shape, dtype=np.uint8))

    def set_black(self):
        self.set_image(np.zeros(self.shape, dtype=np.uint8))

    def set_pinhole(self, r=100, X0=0, Y0=0):
        """r in pixels"""
        mask = np.zeros(self.shape, dtype=np.uint8)
        mask[np.sqrt((self.XX-X0)**2+(self.YY-Y0)**2) < r] = 1 
        self.set_image(mask)

    def get_grating(self, m, phase=0):
        rows, cols = self.shape
        col_vec = ((np.arange(cols) + phase) // m) % 2  # [1,2,3,4...] -> [000,111,222,333] -> [000,111,000,111]
        bitmap = np.broadcast_to(col_vec, (rows, cols)).astype(np.uint8)
        return bitmap
    
    def get_checkerboard(self, period, phase_row=0, phase_col=0):
        rows, cols = self.shape
        row_idx = (np.arange(rows)[:, None] + phase_row) // period
        col_idx = (np.arange(cols)[None, :] + phase_col) // period
        bitmap = ((row_idx + col_idx) % 2).astype(np.uint8)
        return bitmap
    
    def get_1d_macrosteps(self, amplitudes, step_w=100, tile=8):
        rows, cols = self.shape
        assert all(amplitudes < 1.001) and all(amplitudes > -0.001), "amplitudes must be between 0 and 1"

        # Build Bayer tiles recursively (2X2 -> 4X4 -> 8X8)
        B = np.array([[0,2],[3,1]], int)
        while B.shape[0] < tile:
            B = np.block([[4*B+0, 4*B+2],
                        [4*B+3, 4*B+1]])
        T = (B[:tile, :tile] + 0.5) / (tile*tile)

        bitmap = np.zeros((rows, cols), np.uint8)
        for x0 in range(0, cols, step_w):
            a = amplitudes[(x0 // step_w) % len(amplitudes)]  # a is the Desired amplitude for this macro-step, cyclicly 
            w = min(step_w, cols - x0)  # For last column, which may be shorter than step_w 
            # Repeat tile until size rowsXw. "+tile-1//tile" is integer ceiling, which may overshoot, fixed with exact croping
            Th = np.tile(T, ((rows + tile - 1)//tile, (w + tile - 1)//tile))[:rows, :w]  
            bitmap[:, x0:x0+w] = (Th < a).astype(np.uint8)
        return bitmap

    def set_grating(self, m, phase=0):
        self.set_image(self.get_grating(m ,phase))
    
    def set_checkerboard(self, period, phase_row=0, phase_col=0):
        self.set_image(self.get_checkerboard(period, phase_row, phase_col))
    
    def set_1d_macrosteps(self, amplitudes, step_w=100, tile=8):
        self.set_image(self.get_1d_macrosteps(amplitudes, step_w, tile))

    def close(self):
        try:
            self.hs.GetDriver().StopSequence(self.comp_idx)
        except Exception:
            pass

        for fn in ("StopSystem", "Disconnect"):
            if hasattr(self.hs, fn):
                getattr(self.hs, fn)()
                break

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ======================================================================
    # internals
    # ======================================================================

    # ---------- one-shot project & sequence --------------------------------
    def log(self, txt):
        print(txt)

    def _build_static_project(self, frame_time_ms):
        """Create a 'live' project with one blank image ID 1 and start it."""
        rows, cols = self.shape

        blank = np.ones((rows, cols), np.uint8)
        
        img = aj.Image(1)

        self._np_to_aj_image(blank, img)

        self.proj = aj.Project("py_dmd_live")
        self.proj.SetComponents(self.hs.GetProject().Components())
        self.proj.AddImage(img)

        seq = aj.Sequence(self.seq_id, "live", self.hwtype, aj.SEQ_TYPE_PRELOAD, 0)
        self.proj.AddSequence(seq)
        self.proj.AddSequenceItem(aj.SequenceItem(self.seq_id, 1))

        frame = aj.Frame()
        frame.SetSequenceID(self.seq_id)
        frame.SetImageID(1)
        frame.SetFrameTimeMSec(frame_time_ms)
        self.proj.AddFrame(frame)

        drv = self.hs.GetDriver()
        drv.LoadProject(self.proj)
        drv.WaitForLoadComplete(self.timeout_ms)
        drv.StartSequence(self.seq_id, self.comp_idx)

        # remember the mutable image object for quick updates
        self._img = img

    # ---------- fast update -------------------------------------------------
    def _update_image(self, img):
        drv = self.hs.GetDriver()
        self.proj.AddImage(img)

        drv.StopSequence(self.comp_idx)
        self.proj.RemoveFrame(self.seq_id)

        frame = aj.Frame()
        frame.SetSequenceID(self.seq_id)
        frame.SetImageID(1)
        frame.SetFrameTimeMSec(1)
        self.proj.AddFrame(frame)
        
        drv.LoadProject(self.proj)
        drv.WaitForLoadComplete(self.timeout_ms)
        drv.StartSequence(self.seq_id, self.comp_idx)
        # two-arg overload: False → load *only* the image store (no XML/sequences)
        # drv.LoadProject(self.proj, False)
        # drv.WaitForLoadComplete(self.timeout_ms)
        # drv.StartSequence(self.seq_id, self.comp_idx)

    # ---------- helpers ----------------------------------------------------
    def _np_to_aj_image(self, arr2d, img):
        """
        Overwrite an aj.Image in place from a numpy 0/1 array,
        ensuring it's C-contiguous and shaped (rows, cols, channels).
        """
        # 1) scale 0/1 → 0/255 so we can use an 8-bit load
        gray = (arr2d * 255).astype(np.uint8)
        # 2) add a singleton channel axis → shape (rows, cols, 1)
        buf = gray[:, :, None]
        # 3) ensure it's contiguous
        buf = np.ascontiguousarray(buf)

        # 4) load as an 8-bit image so “255” ⇒ full-on micromirrors
        img.ReadFromMemory(
            buf,
            8,                        # bits per pixel
            aj.ROW_MAJOR_ORDER, 
            self.hwtype,
        )

        # debug: width/height nonzero, bitDepth=8, channels=1
        if self.verbose:
            print(f"Loaded image → {img.Width()}×{img.Height()}, "
                f"bitDepth={img.BitDepth()}, channels={img.NumChannels()}")

    def _read_image(self, path, img = aj.Image(1)):
        img.ReadFromFile(os.fspath(path), self.hwtype)
        return img

    def _find_dmd_component(self):
        prj = self.hs.GetProject()
        for dt in (aj.DMD_4500_DEVICE_TYPE, aj.DMD_3000_DEVICE_TYPE):
            idx = prj.GetComponentIndexWithDeviceType(dt)
            if idx >= 0:
                return idx
        raise RuntimeError("No DMD component found")
