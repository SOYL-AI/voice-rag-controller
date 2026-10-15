"""Read Intel RAPL energy counters on Windows through the Energy Meter Interface.

Windows has no user-space /sys/class/powercap. It does expose the CPU's RAPL
accumulators through the kernel's Energy Meter Interface (EMI), and those
surface as the "Energy Meter" performance-counter set:

    \\Energy Meter(RAPL_Package0_PKG)\\Energy   package (cores + uncore + iGPU)
    \\Energy Meter(RAPL_Package0_PP0)\\Energy   core power plane
    \\Energy Meter(RAPL_Package0_PP1)\\Energy   graphics power plane
    \\Energy Meter(RAPL_Package0_DRAM)\\Energy  DRAM (reads 0 on this laptop)

No administrator rights or third-party driver are required. Units were
calibrated on this machine (see energy/REPORT.md, "CPU energy"): the raw
Energy value is a cumulative count in picowatt-hours and the Time counter is
milliseconds. 1 pWh = 3.6e-9 J.

This module is a minimal ctypes wrapper over pdh.dll. It is importable on any
OS; `available()` returns False off Windows or when the counter set is absent.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

PWH_TO_J = 3.6e-9  # 1 picowatt-hour = 1e-12 W * 3600 s

DEFAULT_INSTANCES = ("RAPL_Package0_PKG", "RAPL_Package0_PP0", "RAPL_Package0_DRAM")


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class _PDH_RAW_COUNTER(ctypes.Structure):
    _fields_ = [
        ("CStatus", wintypes.DWORD),
        ("TimeStamp", _FILETIME),
        ("FirstValue", ctypes.c_longlong),
        ("SecondValue", ctypes.c_longlong),
        ("MultiCount", wintypes.DWORD),
    ]


class RaplPdhReader:
    """Cumulative RAPL energy readings (joules) for a set of EMI instances."""

    def __init__(self, instances=DEFAULT_INSTANCES):
        if sys.platform != "win32":
            raise OSError("Windows Energy Meter counters are only available on Windows")
        self._pdh = ctypes.WinDLL("pdh.dll")
        self._query = ctypes.c_void_p()
        status = self._pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query))
        if status != 0:
            raise OSError(f"PdhOpenQueryW failed: 0x{status & 0xFFFFFFFF:08x}")
        self.instances = []
        self._counters = {}
        for inst in instances:
            handle = ctypes.c_void_p()
            path = f"\\Energy Meter({inst})\\Energy"
            status = self._pdh.PdhAddEnglishCounterW(
                self._query, ctypes.c_wchar_p(path), None, ctypes.byref(handle)
            )
            if status == 0:
                self._counters[inst] = handle
                self.instances.append(inst)
        if not self._counters:
            self.close()
            raise OSError("no Energy Meter instances could be added")

    def read_j(self) -> dict:
        """Collect once and return {instance: cumulative joules}."""
        status = self._pdh.PdhCollectQueryData(self._query)
        if status != 0:
            raise OSError(f"PdhCollectQueryData failed: 0x{status & 0xFFFFFFFF:08x}")
        out = {}
        raw = _PDH_RAW_COUNTER()
        ctype = wintypes.DWORD()
        for inst, handle in self._counters.items():
            status = self._pdh.PdhGetRawCounterValue(
                handle, ctypes.byref(ctype), ctypes.byref(raw)
            )
            if status == 0:
                out[inst] = raw.FirstValue * PWH_TO_J
        return out

    def close(self):
        if getattr(self, "_query", None):
            self._pdh.PdhCloseQuery(self._query)
            self._query = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


def available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        r = RaplPdhReader()
        ok = "RAPL_Package0_PKG" in r.read_j()
        r.close()
        return ok
    except OSError:
        return False
