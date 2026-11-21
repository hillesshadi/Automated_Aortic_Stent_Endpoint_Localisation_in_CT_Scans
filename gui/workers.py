#!/usr/bin/env python3
"""
Background execution and log capture.

Two rules drive everything here:

  1. Widgets may only be touched from the GUI thread. Workers therefore never
     hold a widget reference; they emit signals and Qt queues the delivery.

  2. The pipeline reports progress through print() and the logging module, not
     through callbacks. Redirecting those is process-wide state, so it is
     installed once and routed to the GUI by signal -- again never by touching
     a widget from whichever thread happened to emit the line.
"""

import logging
import sys
import threading
import traceback

from .qt_compat import QtCore, Signal


# --------------------------------------------------------------------------- #
#  Log capture
# --------------------------------------------------------------------------- #
class LogBridge(QtCore.QObject):
    """Process-wide stdout/stderr/logging capture, delivered as a Qt signal.

    A single instance is installed by the main window. Emission is thread-safe
    because a queued signal is the handoff; the receiving slot runs on the GUI
    thread no matter which thread wrote the line.
    """

    message = Signal(str, str)          # (level, text)

    def __init__(self):
        super().__init__()
        self._stdout = None
        self._stderr = None
        self._handler = None
        self._installed = False
        self._lock = threading.Lock()

    # -- stream shim ------------------------------------------------------- #
    class _Stream:
        def __init__(self, bridge, level, original):
            self._bridge = bridge
            self._level = level
            self._original = original
            self._buf = ""

        def write(self, text):
            # Keep the real stream working so a terminal launch still shows
            # output, and so a crash during GUI teardown is not swallowed.
            if self._original is not None:
                try:
                    self._original.write(text)
                except UnicodeEncodeError:
                    # A legacy console (cp1252 on Windows) cannot encode the
                    # characters the pipeline reports diameters with. Degrade
                    # that line rather than losing it -- the GUI log below
                    # still receives the original text either way.
                    enc = getattr(self._original, "encoding", None) or "ascii"
                    try:
                        self._original.write(
                            text.encode(enc, "replace").decode(enc, "replace"))
                    except Exception:                          # noqa: BLE001
                        pass
                except Exception:                              # noqa: BLE001
                    pass
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    self._bridge.message.emit(self._level, line)

        def flush(self):
            if self._original is not None:
                try:
                    self._original.flush()
                except Exception:                              # noqa: BLE001
                    pass

        def isatty(self):
            return False

    # -- logging shim ------------------------------------------------------ #
    class _Handler(logging.Handler):
        def __init__(self, bridge):
            super().__init__()
            self._bridge = bridge

        def emit(self, record):
            try:
                level = ("error" if record.levelno >= logging.ERROR
                         else "warn" if record.levelno >= logging.WARNING
                         else "info")
                self._bridge.message.emit(level, self.format(record))
            except Exception:                                  # noqa: BLE001
                pass

    def install(self):
        with self._lock:
            if self._installed:
                return
            self._stdout, self._stderr = sys.stdout, sys.stderr
            sys.stdout = self._Stream(self, "info", self._stdout)
            sys.stderr = self._Stream(self, "error", self._stderr)

            self._handler = self._Handler(self)
            self._handler.setFormatter(
                logging.Formatter("%(name)s: %(message)s"))
            root = logging.getLogger()
            root.addHandler(self._handler)
            if root.level > logging.INFO:
                root.setLevel(logging.INFO)
            # The pipeline logs under this name (stentlib, process_aorta_stents).
            logging.getLogger("stent_sop").setLevel(logging.INFO)
            self._installed = True

    def uninstall(self):
        with self._lock:
            if not self._installed:
                return
            sys.stdout, sys.stderr = self._stdout, self._stderr
            if self._handler is not None:
                logging.getLogger().removeHandler(self._handler)
            self._installed = False


# --------------------------------------------------------------------------- #
#  Worker
# --------------------------------------------------------------------------- #
class PipelineWorker(QtCore.QThread):
    """Run one callable off the GUI thread.

    The callable is passed `progress` and `should_stop` keyword arguments when
    it accepts them, so a long stage can report percentages and honour Cancel
    without knowing anything about Qt.
    """

    progress = Signal(int, str)         # (percent, message)
    finished_ok = Signal(str, object)   # (task_name, result)
    failed = Signal(str, str, str)      # (task_name, message, traceback)

    def __init__(self, name, fn, *args, **kwargs):
        super().__init__()
        self.name = name
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._stop = threading.Event()

    def request_stop(self):
        self._stop.set()

    def should_stop(self):
        return self._stop.is_set()

    def run(self):
        try:
            import inspect
            kwargs = dict(self._kwargs)
            try:
                sig = inspect.signature(self._fn)
                if "progress" in sig.parameters:
                    kwargs["progress"] = self._emit_progress
                if "should_stop" in sig.parameters:
                    kwargs["should_stop"] = self.should_stop
            except (TypeError, ValueError):
                # Builtins and C functions have no introspectable signature;
                # call them plainly rather than failing the whole task.
                pass

            result = self._fn(*self._args, **kwargs)
            if self._stop.is_set():
                self.failed.emit(self.name, "Cancelled.", "")
            else:
                self.finished_ok.emit(self.name, result)
        except Exception as exc:                               # noqa: BLE001
            self.failed.emit(self.name, str(exc), traceback.format_exc())

    def _emit_progress(self, pct, message=""):
        try:
            self.progress.emit(int(pct), str(message))
        except Exception:                                      # noqa: BLE001
            pass
