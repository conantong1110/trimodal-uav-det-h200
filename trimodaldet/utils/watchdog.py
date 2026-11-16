"""
Log watchdog: end the process when nothing has been logged for too long.

A "log" is any non-blank write to stdout or stderr. If none arrives within the
limit (default 40 minutes) the run is treated as hung: the watchdog prints where
every thread is stuck and exits the process with code 124.

The limit comes from, in order: the `minutes` argument, the environment variable
TRIMODAL_WATCHDOG_MINUTES, then the 40-minute default. A value of 0 disables it.

Two mechanisms do the killing:
- A Python thread, which handles every hang where the interpreter can still run
  (blocked DataLoader, stuck GPU call, deadlock, ...). Exit code 124.
- A faulthandler timer as a backstop for the rare case where the interpreter is
  frozen inside C code that never releases the GIL. It fires 60 seconds later
  and exits with code 1. This backstop is dependable on Linux; on Windows with
  PyTorch loaded it prints the traceback but the process can fail to exit.
"""
import faulthandler
import os
import sys
import threading
import time

DEFAULT_MINUTES = 40.0
ENV_VAR = "TRIMODAL_WATCHDOG_MINUTES"
EXIT_CODE = 124          # same code the `timeout` command uses
_BACKSTOP_GRACE = 60.0   # extra seconds before the C-level backstop fires

_active = None


def _terminate():
    """End this process immediately, skipping all cleanup."""
    if os.name == 'nt':
        # os._exit goes through ExitProcess, which can deadlock while unloading
        # PyTorch's DLLs. TerminateProcess ends the process unconditionally.
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess(ctypes.c_void_p(kernel32.GetCurrentProcess()), EXIT_CODE)
    os._exit(EXIT_CODE)


class _WatchedStream:
    """Passes everything through to the real stream and reports each log write."""

    def __init__(self, stream, watchdog):
        self._stream = stream
        self._watchdog = watchdog

    def write(self, text):
        if text and not text.isspace():
            self._watchdog.touch()
        return self._stream.write(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


class LogWatchdog:
    """Kills the process once `timeout_seconds` pass without a log write."""

    def __init__(self, timeout_seconds):
        self.timeout = float(timeout_seconds)
        self._last = time.monotonic()
        self._real_stderr = sys.__stderr__ or sys.stderr
        self._thread = threading.Thread(target=self._run, name="log-watchdog", daemon=True)

    def touch(self):
        self._last = time.monotonic()

    def start(self):
        sys.stdout = _WatchedStream(sys.stdout, self)
        sys.stderr = _WatchedStream(sys.stderr, self)
        self._arm_backstop(self.timeout)
        self._thread.start()

    def _arm_backstop(self, remaining):
        # faulthandler's timer runs in C without the GIL, so it still fires when
        # the interpreter itself is frozen and the Python thread below cannot run.
        try:
            faulthandler.dump_traceback_later(
                remaining + _BACKSTOP_GRACE, exit=True, file=self._real_stderr)
        except (AttributeError, ValueError, OSError):
            pass  # stderr has no file descriptor; the Python thread still works

    def _run(self):
        interval = max(0.5, min(30.0, self.timeout / 4))
        while True:
            time.sleep(interval)
            idle = time.monotonic() - self._last
            if idle >= self.timeout:
                self._kill(idle)
            self._arm_backstop(self.timeout - idle)

    def _kill(self, idle):
        err = self._real_stderr
        try:
            err.write(f"\n[watchdog] No log output for {idle / 60:.1f} min "
                      f"(limit {self.timeout / 60:g} min). Killing the process.\n")
            err.flush()
            faulthandler.dump_traceback(file=err, all_threads=True)
            err.flush()
        finally:
            _terminate()


def install_log_watchdog(minutes=None):
    """
    Start the watchdog for this process (only the first call has an effect).

    Args:
        minutes: Allowed silence in minutes. None reads TRIMODAL_WATCHDOG_MINUTES
                 and falls back to 40. 0 or less disables the watchdog.
    """
    global _active
    if _active is not None:
        return _active
    if minutes is None:
        minutes = float(os.environ.get(ENV_VAR, DEFAULT_MINUTES))
    if minutes <= 0:
        print("[watchdog] Disabled.")
        return None

    _active = LogWatchdog(minutes * 60)
    _active.start()
    print(f"[watchdog] The process will be killed if no log appears for {minutes:g} minutes.")
    return _active


__all__ = ['LogWatchdog', 'install_log_watchdog', 'DEFAULT_MINUTES', 'ENV_VAR', 'EXIT_CODE']
