"""Windows capture resources for one browser video track; no desktop capture."""

from contextlib import contextmanager
import ctypes as c
from ctypes import wintypes
from datetime import timedelta
from fractions import Fraction
import logging
from threading import Condition
from time import monotonic
from uuid import UUID

import numpy as np


def _call(obj, index, result, *arguments):
    address = c.cast(obj, c.POINTER(c.POINTER(c.c_void_p))).contents[index]
    return c.WINFUNCTYPE(result, c.c_void_p, *(kind for kind, _ in arguments))(
        address)(obj, *(value for _, value in arguments))


def _check(result):
    if result < 0:
        raise OSError(result, f"Windows graphics operation failed: HRESULT 0x{result & 0xffffffff:08X}")


def _query(obj, interface):
    output = c.c_void_p()
    guid = c.create_string_buffer(UUID(interface).bytes_le)
    _check(_call(obj, 0, c.c_long, (c.c_void_p, c.addressof(guid)),
                 (c.POINTER(c.c_void_p), c.byref(output))))
    return output


def _release(obj):
    if obj:
        _call(obj, 2, c.c_ulong)


class _VideoStream(c.Structure):
    """D3D11_VIDEO_PROCESSOR_STREAM from the Windows SDK d3d11.h."""

    _fields_ = [("enabled", c.c_int), ("output_index", c.c_uint),
                ("input_frame", c.c_uint), ("past_count", c.c_uint),
                ("future_count", c.c_uint), ("past", c.c_void_p),
                ("input", c.c_void_p), ("future", c.c_void_p),
                ("past_right", c.c_void_p), ("input_right", c.c_void_p),
                ("future_right", c.c_void_p)]


class WindowCapture:
    """Retain the D3D device while switching only the authorized window session.

    Construct, select, read and close on the media worker. Frame-arrived callbacks
    retain only the latest native frame; mapped pixels live inside frame().
    """

    def __init__(self, fps=60):
        from winrt.runtime import init_apartment, ApartmentType
        from winrt.windows.graphics.directx.direct3d11.interop import create_direct3d11_device_from_dxgi_device
        import winrt.windows.foundation
        import winrt.windows.graphics
        import winrt.windows.graphics.directx

        init_apartment(ApartmentType.MULTI_THREADED)
        self.apartment = True
        self.device, self.context = c.c_void_p(), c.c_void_p()
        self.pool = self.session = self.token = self.latest = None
        self.staging = self.staging_size = self.failure = None
        self.video_device = self.video_context = self.enumerator = self.processor = None
        self.target = self.output_view = self.video_size = self.wrapped = None
        self.interrupted = False
        self.fps = fps
        self.items = {}
        self.serial = 0
        self.condition = Condition()
        self.api = c.WinDLL("user32", use_last_error=True)
        self.api.GetClientRect.argtypes = (wintypes.HWND, c.POINTER(wintypes.RECT))
        self.api.GetWindowRect.argtypes = (wintypes.HWND, c.POINTER(wintypes.RECT))
        self.api.ClientToScreen.argtypes = (wintypes.HWND, c.POINTER(wintypes.POINT))
        self.api.IsWindowVisible.argtypes = (wintypes.HWND,)
        self.api.IsIconic.argtypes = (wintypes.HWND,)
        self.api.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
        self.api.GetAncestor.restype = wintypes.HWND
        self.api.GetWindowDisplayAffinity.argtypes = (wintypes.HWND, c.POINTER(wintypes.DWORD))
        self.dwm = c.WinDLL("dwmapi")
        self.dwm.DwmGetWindowAttribute.argtypes = (wintypes.HWND, wintypes.DWORD, c.c_void_p, wintypes.DWORD)
        create = c.WinDLL("d3d11").D3D11CreateDevice
        create.argtypes = (c.c_void_p, c.c_uint, c.c_void_p, c.c_uint, c.c_void_p,
                           c.c_uint, c.c_uint, c.POINTER(c.c_void_p), c.c_void_p, c.POINTER(c.c_void_p))
        create.restype = c.c_long
        try:
            _check(create(None, 1, None, 0x20, None, 0, 7, c.byref(self.device), None, c.byref(self.context)))
            dxgi = _query(self.device, "54ec77fa-1377-44e6-8c32-88fd5f44c84c")
            try:
                self.wrapped = create_direct3d11_device_from_dxgi_device(dxgi.value)
            finally:
                _release(dxgi)
            multithread = _query(self.context, "9b7e4e00-342c-4106-a19f-4f2704f689f0")
            try:
                _call(multithread, 5, c.c_int, (c.c_int, 1))
            finally:
                _release(multithread)
            try:
                self.video_device = _query(self.device, "10ec4d5b-975a-4689-b9e4-d0aac30fe333")
                self.video_context = _query(self.context, "61f21c45-3c0e-4a74-9cea-67100d9ad5e4")
            except OSError as error:
                if error.errno & 0xffffffff != 0x80004002:  # E_NOINTERFACE
                    raise
                _release(self.video_device)
                self.video_device = None
        except BaseException:
            self.close()
            raise

    def _geometry(self, hwnd):
        client, origin, border, window = wintypes.RECT(), wintypes.POINT(), wintypes.RECT(), wintypes.RECT()
        if (not self.api.GetClientRect(hwnd, c.byref(client))
                or not self.api.ClientToScreen(hwnd, c.byref(origin))
                or not self.api.GetWindowRect(hwnd, c.byref(window))):
            raise c.WinError(c.get_last_error())
        bounds = {}
        if self.dwm.DwmGetWindowAttribute(hwnd, 9, c.byref(border), c.sizeof(border)) >= 0:
            bounds[(border.right - border.left, border.bottom - border.top)] = (border.left, border.top)
        bounds.setdefault((window.right - window.left, window.bottom - window.top), (window.left, window.top))
        return origin, (client.right, client.bottom), bounds

    def _place_crop(self, geometry):
        origin, client_size, bounds = geometry
        if self.frame_size not in bounds:
            raise OSError(f"Capture size {self.frame_size} does not match native window bounds {tuple(bounds)}")
        frame_left, frame_top = bounds[self.frame_size]
        left, top, width, height = self.client_crop
        self.crop = (origin.x - frame_left + left, origin.y - frame_top + top, width, height)
        if min(left, top) < 0 or left + width > client_size[0] or top + height > client_size[1]:
            self.crop = None

    def select(self, identity, hwnd, crop, available, size):
        from winrt.windows.graphics.capture import Direct3D11CaptureFramePool
        from winrt.windows.graphics.capture.interop import create_for_window

        with self.condition:
            self.interrupted = False
        self.size = tuple(map(int, size))
        if min(self.size) <= 0 or any(value % 2 for value in self.size):
            raise ValueError("video output dimensions must be positive even pixels")
        self.items = {key: value for key, value in self.items.items() if key[0] in available}
        self.hwnd, self.client_crop = hwnd, tuple(crop)
        geometry = self._geometry(hwnd)
        self.clear()
        self.frame_size = None
        serial = self.serial

        def arrived(pool, _args):
            with self.condition:
                if serial != self.serial:
                    return
                try:
                    frame = pool.try_get_next_frame()
                    if frame is None:
                        return
                    if self.latest is not None:
                        self.latest.close()
                    self.latest = frame
                except OSError as error:
                    error.add_note(f"TryGetNextFrame: HWND=0x{hwnd:x}")
                    self.failure = error
                self.condition.notify()

        phase = "GraphicsCaptureItem.Size"
        try:
            item_key = (identity, hwnd)
            item = self.items.get(item_key)
            if item is None or (item.size.width, item.size.height) not in geometry[2]:
                phase = "CreateForWindow"
                try:
                    item = self.items[item_key] = create_for_window(hwnd)
                except OSError as error:
                    affinity = wintypes.DWORD()
                    code = getattr(error, "winerror", error.errno)
                    if (code is not None and code & 0xffffffff == 0x80070057
                            and self.api.IsWindowVisible(hwnd) and not self.api.IsIconic(hwnd)
                            and self.api.GetAncestor(hwnd, 2) == hwnd
                            and self.api.GetWindowDisplayAffinity(hwnd, c.byref(affinity)) and not affinity.value):
                        logging.getLogger(__name__).warning(
                            "CreateForWindow rejected HWND=0x%x; using the authorized window pixel capture: %s", hwnd, error)
                        return False
                    raise
            phase = "GraphicsCaptureItem.Size"
            size = item.size
            self.frame_size = (size.width, size.height)
            phase = "Native capture geometry"
            self._place_crop(geometry)
            phase = "FramePool.CreateFreeThreaded"
            self.pool = Direct3D11CaptureFramePool.create_free_threaded(self.wrapped, 87, 3, size)
            phase = "AddFrameArrived"
            self.token = self.pool.add_frame_arrived(arrived)
            phase = "CreateCaptureSession"
            self.session = self.pool.create_capture_session(item)
            for name, value in (("is_cursor_capture_enabled", False), ("is_border_required", False),
                                ("min_update_interval", timedelta(seconds=1 / (2 * self.fps)))):
                try:
                    setattr(self.session, name, value)
                except (AttributeError, OSError) as error:
                    # PyWinRT reports an absent optional property as AttributeError.
                    logging.getLogger(__name__).warning("Windows capture option %s is unavailable: %s", name, error)
            phase = "StartCapture"
            self.session.start_capture()
            return True
        except BaseException as error:
            error.add_note(f"{phase}: HWND=0x{hwnd:x}, capture={self.frame_size}, client_crop={self.client_crop}, output={self.size}")
            self.clear()
            raise

    def interrupt(self):
        """Wake the media worker without touching its native resources."""
        with self.condition:
            self.interrupted = True
            self.condition.notify()

    def clear(self):
        with self.condition:
            self.serial += 1
            if self.latest is not None:
                self.latest.close()
                self.latest = None
            self.failure = None
        session, pool, token = self.session, self.pool, self.token
        self.session = self.pool = self.token = None
        try:
            if session is not None:
                session.close()
        finally:
            if pool is not None:
                try:
                    if token is not None:
                        pool.remove_frame_arrived(token)
                finally:
                    pool.close()

    def _texture(self, width, height, pixel_format, staging=False):
        desc = (c.c_uint * 11)(width, height, 1, 1, pixel_format, 1, 0,
                               3 if staging else 0, 0 if staging else 0x20,
                               0x20000 if staging else 0, 0)
        texture = c.c_void_p()
        _check(_call(self.device, 5, c.c_long, (c.c_void_p, c.addressof(desc)),
                     (c.c_void_p, None), (c.POINTER(c.c_void_p), c.byref(texture))))
        return texture

    def _clear_video(self):
        for name in ("output_view", "processor", "enumerator", "target"):
            _release(getattr(self, name))
            setattr(self, name, None)
        self.video_size = None

    def _prepare_video(self, source_size):
        if self.video_device is None:
            return False
        key = (*source_size, *self.size)
        if self.video_size == key:
            return self.processor is not None
        self._clear_video()
        width, height = self.size
        try:
            rate = Fraction(str(self.fps))
            desc = (c.c_uint * 10)(0, rate.numerator, rate.denominator, *source_size,
                                   rate.numerator, rate.denominator, width, height, 1)
            self.enumerator = c.c_void_p()
            _check(_call(self.video_device, 10, c.c_long, (c.c_void_p, c.addressof(desc)),
                         (c.POINTER(c.c_void_p), c.byref(self.enumerator))))
            for pixel_format, required in ((87, 1), (103, 2)):
                flags = c.c_uint()
                _check(_call(self.enumerator, 8, c.c_long, (c.c_uint, pixel_format),
                             (c.POINTER(c.c_uint), c.byref(flags))))
                if not flags.value & required:
                    self._clear_video()
                    self.video_size = key
                    return False
            self.processor = c.c_void_p()
            _check(_call(self.video_device, 4, c.c_long, (c.c_void_p, self.enumerator),
                         (c.c_uint, 0), (c.POINTER(c.c_void_p), c.byref(self.processor))))
            self.target = self._texture(width, height, 103)
            output_desc = (c.c_uint * 4)(1, 0, 0, 0)
            self.output_view = c.c_void_p()
            _check(_call(self.video_device, 9, c.c_long, (c.c_void_p, self.target),
                         (c.c_void_p, self.enumerator), (c.c_void_p, c.addressof(output_desc)),
                         (c.POINTER(c.c_void_p), c.byref(self.output_view))))
        except OSError as error:
            self._clear_video()
            if error.errno & 0xffffffff not in (0x80004001, 0x887A0004):
                raise
            self.video_size = key
            return False
        # Progressive RGB full range to BT.709 studio-range NV12; no enhancement.
        _call(self.video_context, 27, None, (c.c_void_p, self.processor), (c.c_uint, 0), (c.c_uint, 0))
        _call(self.video_context, 37, None, (c.c_void_p, self.processor), (c.c_uint, 0), (c.c_int, 0))
        input_color, output_color = c.c_uint(2 << 4), c.c_uint((1 << 2) | (1 << 4))
        _call(self.video_context, 28, None, (c.c_void_p, self.processor), (c.c_uint, 0),
              (c.c_void_p, c.addressof(input_color)))
        _call(self.video_context, 15, None, (c.c_void_p, self.processor),
              (c.c_void_p, c.addressof(output_color)))
        black = (c.c_float * 4)(0, 0, 0, 1)
        _call(self.video_context, 14, None, (c.c_void_p, self.processor), (c.c_int, 0),
              (c.c_void_p, c.addressof(black)))
        self.video_size = key
        return True

    @contextmanager
    def frame(self, timeout=.05):
        from winrt.windows.graphics.directx.direct3d11.interop import get_dxgi_surface_from_object

        deadline = monotonic() + timeout
        with self.condition:
            while self.latest is None and self.failure is None and not self.interrupted:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            if self.failure is not None:
                raise self.failure
            if self.interrupted:
                self.interrupted = False
                frame = None
            else:
                frame, self.latest = self.latest, None
        if frame is None:
            yield None
            return
        dxgi = texture = input_view = None
        mapped = False
        phase = "Frame.ContentSize"
        try:
            content_size = frame.content_size
            if (content_size.width, content_size.height) != self.frame_size:
                frame.close()
                frame = None
                with self.condition:
                    if self.latest is not None:
                        self.latest.close()
                        self.latest = None
                phase = "FramePool.Recreate"
                self.pool.recreate(self.wrapped, 87, 3, content_size)
                with self.condition:
                    if self.latest is not None:
                        self.latest.close()
                        self.latest = None
                self.frame_size = (content_size.width, content_size.height)
                self._place_crop(self._geometry(self.hwnd))
                yield None
                return
            if self.crop is None:
                yield None
                return
            phase = "Frame.Surface / GetDXGISurface"
            dxgi = c.c_void_p(get_dxgi_surface_from_object(frame.surface))
            phase = "D3D texture conversion / Map"
            texture = _query(dxgi, "6f15aaf2-d208-4e89-9ab4-489535d34f9c")
            desc = (c.c_uint * 11)()
            _call(texture, 10, None, (c.c_void_p, c.addressof(desc)))
            left, top, width, height = self.crop
            if min(left, top) < 0 or min(width, height) <= 0 or left + width > self.frame_size[0] or top + height > self.frame_size[1]:
                yield None
                return
            gpu = self._prepare_video((desc[0], desc[1]))
            if gpu:
                output_width, output_height = self.size
                ratio = min(output_width / width, output_height / height)
                scaled_width = max(2, int(width * ratio) // 2 * 2)
                scaled_height = max(2, int(height * ratio) // 2 * 2)
                x, y = (output_width - scaled_width) // 4 * 2, (output_height - scaled_height) // 4 * 2
                source_rect = wintypes.RECT(left, top, left + width, top + height)
                destination = wintypes.RECT(x, y, x + scaled_width, y + scaled_height)
                output_rect = wintypes.RECT(0, 0, output_width, output_height)
                _call(self.video_context, 13, None, (c.c_void_p, self.processor),
                      (c.c_int, 1), (c.c_void_p, c.addressof(output_rect)))
                _call(self.video_context, 30, None, (c.c_void_p, self.processor),
                      (c.c_uint, 0), (c.c_int, 1), (c.c_void_p, c.addressof(source_rect)))
                _call(self.video_context, 31, None, (c.c_void_p, self.processor),
                      (c.c_uint, 0), (c.c_int, 1), (c.c_void_p, c.addressof(destination)))
                input_desc, input_view = (c.c_uint * 4)(0, 1, 0, 0), c.c_void_p()
                try:
                    _check(_call(self.video_device, 8, c.c_long, (c.c_void_p, texture),
                                 (c.c_void_p, self.enumerator), (c.c_void_p, c.addressof(input_desc)),
                                 (c.POINTER(c.c_void_p), c.byref(input_view))))
                    stream = _VideoStream(enabled=1, input=input_view)
                    _check(_call(self.video_context, 53, c.c_long, (c.c_void_p, self.processor),
                                 (c.c_void_p, self.output_view), (c.c_uint, 0), (c.c_uint, 1),
                                 (c.c_void_p, c.addressof(stream))))
                except OSError as error:
                    if error.errno & 0xffffffff not in (0x80004001, 0x887A0004):
                        raise
                    self._clear_video()
                    self.video_size = (desc[0], desc[1], *self.size)
                    gpu = False
            pixel_format = 103 if gpu else 87
            if gpu:
                width, height = self.size
            if self.staging_size != (width, height, pixel_format):
                _release(self.staging)
                self.staging = self.staging_size = None
                self.staging = self._texture(width, height, pixel_format, staging=True)
                self.staging_size = (width, height, pixel_format)
            if gpu:
                _call(self.context, 47, None, (c.c_void_p, self.staging), (c.c_void_p, self.target))
            else:
                box = (c.c_uint * 6)(left, top, 0, left + width, top + height, 1)
                _call(self.context, 46, None, (c.c_void_p, self.staging), (c.c_uint, 0),
                      (c.c_uint, 0), (c.c_uint, 0), (c.c_uint, 0), (c.c_void_p, texture),
                      (c.c_uint, 0), (c.c_void_p, c.addressof(box)))
            region = c.create_string_buffer(16)
            _check(_call(self.context, 14, c.c_long, (c.c_void_p, self.staging),
                         (c.c_uint, 0), (c.c_uint, 1), (c.c_uint, 0), (c.c_void_p, c.addressof(region))))
            mapped = True
            address = c.cast(region, c.POINTER(c.c_void_p))[0]
            stride = c.c_uint.from_buffer(region, 8).value
            rows = height * 3 // 2 if gpu else height
            shape, strides = ((rows, width), (stride, 1)) if gpu else ((height, width, 4), (stride, 4, 1))
            pixels = np.ndarray(shape, np.uint8, buffer=(c.c_ubyte * (stride * rows)).from_address(address),
                                strides=strides)
            phase = "Frame.SystemRelativeTime"
            pts = int(frame.system_relative_time.total_seconds() * 90000)
            # Map has completed the GPU copy; encoding needs only our staging texture.
            _release(input_view)
            _release(texture)
            _release(dxgi)
            input_view = texture = dxgi = None
            frame.close()
            frame = None
            phase = "Video frame consumer"
            yield pixels, pts, "nv12" if gpu else "bgra"
        except BaseException as error:
            error.add_note(f"{phase}: HWND=0x{self.hwnd:x}, capture={self.frame_size}, crop={self.crop}, output={self.size}")
            raise
        finally:
            if mapped:
                _call(self.context, 15, None, (c.c_void_p, self.staging), (c.c_uint, 0))
            _release(input_view)
            _release(texture)
            _release(dxgi)
            if frame is not None:
                frame.close()

    def close(self):
        from winrt.runtime import uninit_apartment

        if not self.apartment:
            return
        try:
            self.clear()
        finally:
            self.items.clear()
            self.wrapped = None
            self._clear_video()
            for name in ("staging", "video_context", "video_device", "context", "device"):
                _release(getattr(self, name))
                setattr(self, name, None)
            uninit_apartment()
            self.apartment = False
