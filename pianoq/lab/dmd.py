# 2025-06-09 after some work by Elad. Putting here for ref.
# dmd_driver.py
import os
import numpy as np
import ajiledriver as aj
from PIL import Image
import cv2


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
        try:                              # Ajile v1.5+
            self.shape = (comp.Rows(), comp.Columns())
        except AttributeError:            # fallback: common 4500 size
            self.shape = (1140, 912)      # (rows, cols)
        
        # For convenience 
        self.Ny, self.Nx = self.shape
        self.X = np.arange(-self.Nx//2, self.Nx//2)
        self.Y = np.arange(-self.Ny//2, self.Ny//2)
        self.XX, self.YY = np.meshgrid(self.X, self.Y)

        # --- build once-off project ---------------------------------------
        self._build_static_project(frame_time_ms)
        self.log('Built project!')

        # smaller of the two SDK options for “images only”
        self._loadopt_images_only = (
            getattr(aj, "LOAD_OPT_IMAGES_ONLY",
                    getattr(aj, "LOAD_OPT_UPDATE_IMAGES", 2))
        )

    # ---------------------------------------- public API
    def set_image_from_path(self, path):
        img = self._read_image(path)
        self._update_image(img)
        """
        Load any image from disk, threshold it to a 0/1 mask, resize
        to the DMD’s resolution, and display it.
        """

        # load as 8-bit grayscale
        # pil = Image.open(path).convert("L")
        # rows, cols = self.shape
        # pil = pil.resize((cols, rows), resample=Image.NEAREST)

        # # convert to array [0..255]
        # gray = np.array(pil, dtype=np.uint8)
        # # threshold at midpoint → mask of 0|1
        # mask = (gray >= 128).astype(np.uint8)

        # # debug: make sure we actually got both 0s and 1s
        # if self.verbose:
        #     u = np.unique(mask)
        #     print("set_image_from_path: unique values →", u)
        # rows, cols = self.shape
        # mask = np.zeros((rows, cols), np.uint8)
        # self.set_image(mask)

    def set_image(self, bitmap):
        """
        Display a NumPy array (dtype uint8/bool, values 0 | 1) **or**
        an `aj.Image` already matching the DMD geometry.
        """
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

    # --- one-liner helper --------------------------------------------------
    def set_grating(self, m, phase=0):
        """
        Show a binary grating with period *m* pixels (along columns).

        Parameters
        ----------
        m     : int   – grating period in pixels
        phase : int   – shift (0…m-1) in pixels
        """
        rows, cols = self.shape
        col_vec = ((np.arange(cols) + phase) // m) & 1
        bitmap = np.broadcast_to(col_vec, (rows, cols)).astype(np.uint8)

        # debug: ensure both 0s and 1s are present
        if self.verbose:
            u = np.unique(bitmap)
            print(f"set_grating({m}): unique →", u)

        self.set_image(bitmap)

    # ---------------------------------------- context / clean-up
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
        ensuring it’s C-contiguous and shaped (rows, cols, channels).
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
