"""How much memory this machine can actually spare, right now.

Both heavy stages — the NAFNet clean-up and the tiled upscale — can ask for
more than the machine has, and torch's response to that is to page rather than
fail. A job that would have errored in seconds instead crawls for hours with
the whole system in swap. So both size themselves against these numbers first.

Sizing against installed RAM alone isn't enough: the same image finished in 31s
in a fresh process and thrashed for minutes in a long-lived one on the same
8 GB machine. What matters is what's free *now*.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

# Fraction of installed RAM a stage may plan to occupy. The rest is the OS, the
# app itself (~700 MB with weights loaded), and the margin that keeps estimates
# this rough from tipping the machine into swap.
RAM_BUDGET = 0.5

# Fraction of *currently free* RAM it may take.
AVAILABLE_BUDGET = 0.8

if sys.platform == "win32":
    import ctypes

    class _MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def total_ram_bytes() -> Optional[int]:
    """Physical RAM, or None if the platform won't say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    try:  # Windows has no sysconf
        import ctypes

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullTotalPhys) or None
    except Exception:
        return None


def available_ram_bytes() -> Optional[int]:
    """RAM that could be handed out right now, or None if the platform won't say.

    Deliberately counts only what the OS considers reclaimable without paging:
    MemAvailable on Linux, ullAvailPhys on Windows, and free + inactive +
    purgeable pages on macOS (its "free" alone is near zero on a warm machine
    and would refuse everything).
    """
    try:  # Linux
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    if sys.platform == "darwin":
        try:
            import re
            import subprocess

            out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                                 timeout=5).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            free = 0
            for label in ("Pages free", "Pages inactive", "Pages purgeable"):
                m = re.search(rf"{label}:\s+(\d+)", out)
                if m:
                    free += int(m.group(1))
            return free * page or None
        except Exception:
            return None

    try:  # Windows
        import ctypes

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullAvailPhys) or None
    except Exception:
        return None


def budget_from(total: Optional[int], available: Optional[int]) -> Optional[int]:
    """The tighter of "a share of the machine" and "a share of what's free".

    None when neither is known — callers treat that as "don't block".
    """
    limits = []
    if total is not None:
        limits.append(int(total * RAM_BUDGET))
    if available is not None:
        limits.append(int(available * AVAILABLE_BUDGET))
    return min(limits) if limits else None


def device_budget_bytes(device: torch.device) -> Optional[int]:
    """What the accelerator says it can spare, or None if it won't say.

    MPS reports a recommended working-set size (5.33 GiB on an 8 GB M2) — but
    that figure is static, and on unified memory the GPU draws from the same
    pool as everything else, so it's capped by what's actually free. CUDA
    already reports live free memory.
    """
    try:
        if device.type == "mps" and torch.backends.mps.is_available():
            budget = int(torch.mps.recommended_max_memory())
            available = available_ram_bytes()
            if available is not None:
                budget = min(budget, int(available * AVAILABLE_BUDGET))
            return budget or None
        if device.type == "cuda" and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info(device)
            return int(free) or None
    except (RuntimeError, AttributeError, AssertionError):
        return None
    return None


def describe_shortfall(need: int, total: Optional[int],
                       available: Optional[int]) -> str:
    """Phrase *why* there isn't room — "buy more RAM" and "close some tabs"
    are very different pieces of advice."""
    gb = 1 << 30
    if (total is not None and available is not None
            and available * AVAILABLE_BUDGET < total * RAM_BUDGET):
        return (f"only {available / gb:.1f} GB is free right now (of "
                f"{total / gb:.0f} GB). Close some apps and try again")
    if total is not None:
        return f"this machine has {total / gb:.0f} GB of RAM"
    return "there isn't enough free memory"
