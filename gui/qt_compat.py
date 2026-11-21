#!/usr/bin/env python3
"""
Qt binding selection, done once and before anything else imports Qt.

Three libraries in this app each pick a Qt binding independently, and they
will disagree unless told: Qt itself, VTK's QVTKRenderWindowInteractor
(vtkmodules.qt.PyQtImpl, auto-detected in the order PySide6 -> PyQt6 -> PyQt5),
and matplotlib (QT_API). Loading two different bindings into one process
crashes at widget-parenting time with errors that point nowhere near the
cause, so this module resolves the binding first and pins all three to it.

Import this before PyQt6/PySide6/vtk/matplotlib anywhere else in the app.

    from gui.qt_compat import QtCore, QtGui, QtWidgets, Signal, Slot, QT_API
"""

import os
import sys

QT_API = None
_ERRORS = {}

# Honour an explicit choice first so a user can force a binding when both are
# installed, e.g. AORTA_QT_API=PySide6 python app.py
_preferred = os.environ.get("AORTA_QT_API", "").strip()
_candidates = [_preferred] if _preferred else ["PyQt6", "PySide6"]

for _name in _candidates:
    try:
        if _name == "PyQt6":
            from PyQt6 import QtCore, QtGui, QtWidgets       # noqa: F401
            from PyQt6.QtCore import pyqtSignal as Signal     # noqa: F401
            from PyQt6.QtCore import pyqtSlot as Slot         # noqa: F401
            QT_API = "PyQt6"
            break
        if _name == "PySide6":
            from PySide6 import QtCore, QtGui, QtWidgets      # noqa: F401
            from PySide6.QtCore import Signal                 # noqa: F401
            from PySide6.QtCore import Slot                   # noqa: F401
            QT_API = "PySide6"
            break
    except ImportError as exc:                                # noqa: PERF203
        _ERRORS[_name] = str(exc)

if QT_API is None:
    raise ImportError(
        "No supported Qt binding found. Install one of:\n"
        "    pip install PyQt6\n"
        "    pip install PySide6\n"
        "Tried: %s" % ", ".join("%s (%s)" % (k, v) for k, v in _ERRORS.items()))

# VTK: pin before vtkmodules.qt.QVTKRenderWindowInteractor is imported. Its
# autodetection tries PySide6 first, so on a machine with both installed it
# would silently pick the one we did not.
try:
    import vtkmodules.qt as _vtk_qt
    _vtk_qt.PyQtImpl = QT_API
except ImportError:
    pass

# matplotlib: QT_API is what backend_qtagg reads to choose its binding.
os.environ.setdefault("QT_API", {"PyQt6": "pyqt6", "PySide6": "pyside6"}[QT_API])


# --------------------------------------------------------------------------- #
#  Enum shims
#
#  PyQt6/PySide6 scope enums (Qt.Orientation.Horizontal) where PyQt5 had them
#  flat (Qt.Horizontal). Both modern bindings agree, so these aliases exist
#  only to keep call sites short and readable rather than to bridge versions.
# --------------------------------------------------------------------------- #
Qt = QtCore.Qt
HORIZONTAL = Qt.Orientation.Horizontal
VERTICAL = Qt.Orientation.Vertical
ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
ALIGN_LEFT = Qt.AlignmentFlag.AlignLeft
ALIGN_RIGHT = Qt.AlignmentFlag.AlignRight
ALIGN_TOP = Qt.AlignmentFlag.AlignTop


def exec_app(app):
    """QApplication.exec() on both bindings (PyQt5's exec_ is long gone)."""
    return app.exec()


def qt_version_string():
    return "%s (Qt %s)" % (QT_API, QtCore.QT_VERSION_STR
                           if hasattr(QtCore, "QT_VERSION_STR")
                           else QtCore.__version__)


# --------------------------------------------------------------------------- #
#  Display and platform-plugin resilience
#
#  On a Mogon compute node reached through MobaXterm the desktop app can fail
#  three different ways before a single widget exists, and Qt reports all three
#  by aborting the process rather than raising:
#
#    * libxcb-cursor.so.0 is absent -- Qt 6.5+ requires it for the xcb plugin
#      and prints "Could not load the Qt platform plugin xcb" then calls
#      abort();
#    * $DISPLAY is set but the forwarded socket is dead (the login shell that
#      opened it has gone), giving BadWindow / X_ChangeWindowAttributes;
#    * the node has no GLX-capable driver, so the plugin loads but the first
#      OpenGL surface fails.
#
#  A Python try/except around QApplication() cannot catch an abort(), so the
#  checks below run *before* it is constructed. Only if they pass is the xcb
#  plugin attempted at all.
# --------------------------------------------------------------------------- #

#: Shared objects the xcb platform plugin needs at load time. Qt 6.5 added the
#: libxcb-cursor dependency, which is what breaks on otherwise-working nodes.
XCB_LIBRARIES = (
    "libxcb-cursor.so.0",
    "libxcb-xinerama.so.0",
    "libxcb-icccm.so.4",
    "libxcb-image.so.0",
    "libxcb-keysyms.so.1",
    "libxcb-randr.so.0",
    "libxcb-render-util.so.0",
    "libxcb-shape.so.0",
)
#: Only these actually prevent the plugin from loading; the rest are advisory.
XCB_REQUIRED = ("libxcb-cursor.so.0",)

#: Extra directories searched when a library is missing from the default path.
#: HPC images routinely have the library present but outside the loader's view.
FALLBACK_LIB_DIRS = ("/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib")

#: Platform plugins that start successfully but never put a window on screen.
#: "minimal" is included deliberately: it is the documented last-resort
#: fallback, but it is not a usable interactive session and must never be
#: reported as one.
HEADLESS_PLATFORMS = ("offscreen", "minimal", "vnc", "eglfs", "linuxfb")

#: Environment variables that force headless rendering. hpc/run_headless.slurm
#: exports several of these, and a shell that has sourced it -- or a user who
#: exported them while debugging a batch run -- would otherwise start the
#: desktop GUI with no visible window and no error, because Qt and PyVista
#: both treat them as a deliberate instruction.
HEADLESS_ENV_VARS = (
    "QT_QPA_PLATFORM",
    "PYVISTA_OFF_SCREEN",
    "PYVISTA_KILL_DISPLAY",
    "VTK_DEFAULT_RENDER_WINDOW_HEADLESS",
    "VTK_DEFAULT_OPENGL_WINDOW",
)


def clear_headless_environment(keep_platform=False):
    """Unset inherited headless settings before an interactive session.

    Returns {var: old_value} for what was actually cleared, so the caller can
    report it -- silently undoing something the user exported on purpose would
    be its own kind of surprise.

    QT_QPA_PLATFORM is only cleared when it names a plugin that cannot show a
    window; an explicit "wayland" or "xcb" is a real choice and is preserved.
    """
    cleared = {}
    for var in HEADLESS_ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            continue
        if var == "QT_QPA_PLATFORM":
            if keep_platform or value.split(":")[0] not in HEADLESS_PLATFORMS:
                continue
        elif var == "VTK_DEFAULT_OPENGL_WINDOW":
            # Only the offscreen render-window classes are a problem here.
            if "OSOpenGL" not in value and "EGL" not in value:
                continue
        cleared[var] = value
        del os.environ[var]

    # matplotlib is separate: Agg is correct for batch, wrong for the Qt
    # canvases the MPR panel embeds.
    backend = os.environ.get("MPLBACKEND", "")
    if backend.lower() in ("agg", "pdf", "ps", "svg", "template"):
        cleared["MPLBACKEND"] = backend
        del os.environ["MPLBACKEND"]
    return cleared


def _can_load(soname):
    """True if the loader can find and open `soname`."""
    import ctypes
    try:
        ctypes.CDLL(soname)
        return True
    except OSError:
        return False


def missing_xcb_libraries():
    """Which xcb shared objects the platform plugin needs but cannot load."""
    if os.name == "nt" or sys.platform == "darwin":
        return []
    return [lib for lib in XCB_LIBRARIES if not _can_load(lib)]


def add_fallback_library_paths(dirs=FALLBACK_LIB_DIRS):
    """Append known library directories to LD_LIBRARY_PATH.

    IMPORTANT: this does *not* help the current process find a library.
    glibc's dynamic linker reads LD_LIBRARY_PATH once, at exec time, and
    caches it; a later dlopen() -- which is how Qt loads its platform plugin
    -- uses that cached value and never re-reads the environment. Mutating
    os.environ here only affects child processes, and a re-exec of ourselves.

    preload_libraries() is what actually makes a missing SONAME resolvable in
    this process. This function exists for the child-process probe and for
    re_exec_with_library_path().
    """
    if os.name == "nt":
        return []
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    added = []
    for d in dirs:
        if os.path.isdir(d) and d not in parts:
            parts.append(d)
            added.append(d)
    if added:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return added


# --------------------------------------------------------------------------- #
#  Finding and preloading a missing xcb library, without root
#
#  Qt 6.5 made libxcb-cursor.so.0 a hard dependency of the xcb platform
#  plugin. On a compute node whose base image predates that -- and where
#  neither `dnf install` nor an outbound HTTP fetch is available -- the
#  library still often exists somewhere the loader is not looking: vendored
#  inside a Qt wheel, inside an auditwheel `.libs` directory, in a conda
#  prefix, or under an EasyBuild module root.
#
#  The fix is to load it by absolute path with RTLD_GLOBAL *before* Qt
#  dlopens the plugin. A shared object loaded that way is registered under
#  its SONAME, so when the loader later resolves the plugin's
#  "DT_NEEDED libxcb-cursor.so.0" it finds the already-open object and does
#  not search the filesystem at all. That is the one mechanism that works
#  from inside a running process.
# --------------------------------------------------------------------------- #
def qt_library_dirs():
    """Directories the installed Qt wheel ships shared objects in.

    Both bindings are inspected, not just the active one: PySide6 vendors the
    xcb libraries even when PyQt6 is the binding in use, so an unused PySide6
    install sitting in the same environment is a perfectly good source for the
    library PyQt6's plugin is missing.
    """
    dirs = []
    # dict.fromkeys de-duplicates while preserving order -- QT_API is usually
    # already one of the two names below.
    for mod_name in dict.fromkeys([QT_API, "PyQt6", "PySide6"]):
        if not mod_name:
            continue
        try:
            mod = __import__(mod_name)
        except Exception:                                      # noqa: BLE001
            continue
        root = os.path.dirname(os.path.abspath(mod.__file__))
        for rel in ("Qt6/lib", "Qt/lib", "Qt6/plugins/platforms",
                    "Qt/plugins/platforms", "Qt6/bin", "Qt/bin"):
            candidate = os.path.join(root, *rel.split("/"))
            if os.path.isdir(candidate):
                dirs.append(candidate)
        # auditwheel vendors dependencies into a sibling "<dist>.libs" dir.
        parent = os.path.dirname(root)
        for name in os.listdir(parent) if os.path.isdir(parent) else []:
            if name.endswith(".libs") and os.path.isdir(
                    os.path.join(parent, name)):
                dirs.append(os.path.join(parent, name))

    seen, ordered = set(), []
    for d in dirs:
        real = os.path.realpath(d)
        if real not in seen:
            seen.add(real)
            ordered.append(d)
    return ordered


def candidate_library_dirs():
    """Every directory worth searching for a missing shared object.

    Ordered cheapest/most-likely first. Includes the EasyBuild module roots
    HPC sites expose as $EBROOT*, because on Mogon a `module load` often puts
    exactly this kind of library somewhere the default loader path misses.
    """
    dirs = list(qt_library_dirs())
    dirs += [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
             if p]
    for prefix in (sys.prefix, getattr(sys, "base_prefix", sys.prefix),
                   os.environ.get("CONDA_PREFIX", "")):
        if prefix:
            for rel in ("lib", "lib64"):
                d = os.path.join(prefix, rel)
                if os.path.isdir(d):
                    dirs.append(d)
    for var, value in os.environ.items():
        if var.startswith("EBROOT") and value:
            for rel in ("lib", "lib64"):
                d = os.path.join(value, rel)
                if os.path.isdir(d):
                    dirs.append(d)
    dirs += [d for d in FALLBACK_LIB_DIRS if os.path.isdir(d)]

    seen, ordered = set(), []
    for d in dirs:
        real = os.path.realpath(d)
        if real not in seen:
            seen.add(real)
            ordered.append(d)
    return ordered


def find_library_file(soname, dirs=None):
    """Absolute path to a file providing `soname`, or None.

    Accepts the exact SONAME, the unversioned development symlink, and the
    fully-versioned real file (libxcb-cursor.so.0.0.0), because a node may
    ship any one of the three.
    """
    import glob as _glob

    stem = soname.split(".so")[0]
    patterns = [soname, stem + ".so", stem + ".so.*"]
    for directory in (dirs if dirs is not None else candidate_library_dirs()):
        for pattern in patterns:
            for hit in sorted(_glob.glob(os.path.join(directory, pattern))):
                if os.path.isfile(hit) or os.path.islink(hit):
                    return os.path.realpath(hit)
    return None


def preload_libraries(sonames, verbose=False):
    """dlopen each SONAME globally so a later plugin load can resolve it.

    Returns {soname: path-or-None}. RTLD_GLOBAL is the point: without it the
    symbols stay private to this handle and the plugin's own dependency is
    not satisfied.
    """
    import ctypes

    if os.name == "nt":
        return {}
    results = {}
    for soname in sonames:
        if _can_load(soname):
            results[soname] = "already resolvable"
            continue
        path = find_library_file(soname)
        if path is None:
            results[soname] = None
            continue
        try:
            ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            results[soname] = path
            if verbose:
                print("preloaded %s from %s" % (soname, path))
        except OSError as exc:
            results[soname] = None
            if verbose:
                print("could not preload %s (%s): %s" % (soname, path, exc))
    return results


def ensure_xcb_loadable(verbose=False):
    """Try to make every required xcb library resolvable in this process.

    Returns (ok, notes). `ok` means nothing required is still missing --
    either it was always there, or preloading satisfied it.
    """
    notes = []
    missing = missing_xcb_libraries()
    if not [lib for lib in missing if lib in XCB_REQUIRED]:
        return True, notes

    # Preload everything that is missing, not only the gating library: the
    # advisory ones are advisory about *whether they abort the load*, not
    # about whether Qt wants them, and satisfying one from a wheel costs
    # nothing. Only XCB_REQUIRED decides success.
    notes.append("missing before preload: %s" % ", ".join(missing))
    results = preload_libraries(missing, verbose=verbose)
    for soname in missing:
        path = results.get(soname)
        if path is None:
            notes.append("%s: not found in any searchable directory" % soname)
        elif path != "already resolvable":
            notes.append("%s: preloaded from %s" % (soname, path))

    remaining = [lib for lib in missing_xcb_libraries() if lib in XCB_REQUIRED]
    # _can_load() re-checks by SONAME; a successful RTLD_GLOBAL preload makes
    # that succeed, which is exactly the condition Qt's plugin load needs.
    if remaining:
        notes.append("still unresolved: %s" % ", ".join(remaining))
        return False, notes
    notes.append("all required xcb libraries now resolve")
    return True, notes


def probe_xcb_plugin(timeout=30):
    """(ok, detail) -- can Qt actually bring up the xcb platform plugin?

    Run in a child process for two reasons: Qt calls abort() rather than
    raising when a platform plugin fails to load, which no except clause can
    catch; and a QApplication cannot be created twice in one process, so a
    failed attempt here would poison the real one.

    This is the authority. The static library check is only a hint -- it
    cannot know which Qt version needs which SONAME, nor whether a preload
    already satisfied it.
    """
    import subprocess

    if os.name == "nt" or sys.platform == "darwin":
        return True, "native platform (no xcb)"

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Built by substitution rather than %-formatting: the program itself
    # contains a %s, and mixing the two silently turns the repo path into a
    # second format argument (TypeError: not enough arguments for format
    # string) every time this runs.
    code = "\n".join((
        "import os, sys",
        "sys.path.insert(0, __ROOT__)",
        "os.environ['QT_QPA_PLATFORM'] = 'xcb'",
        "from gui import qt_compat",
        "qt_compat.preload_libraries(qt_compat.XCB_LIBRARIES)",
        "from gui.qt_compat import QtGui",
        "a = QtGui.QGuiApplication([])",
        "s = a.primaryScreen()",
        "w = s.geometry().width() if s else 0",
        "sys.stdout.write('XCB_OK ' + str(w) + chr(10))",
    )).replace("__ROOT__", repr(root))
    try:
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=timeout,
                              env=dict(os.environ))
    except subprocess.TimeoutExpired:
        return False, "xcb probe timed out after %ds" % timeout
    except Exception as exc:                                   # noqa: BLE001
        return False, "xcb probe could not run: %s" % exc

    out = (proc.stdout or "").strip()
    if proc.returncode == 0 and "XCB_OK" in out:
        return True, "xcb started in a child process (%s)" % out.strip()
    if proc.returncode < 0:
        return False, ("xcb probe killed by signal %d -- the plugin aborted"
                       % -proc.returncode)
    err = (proc.stderr or "").strip().splitlines()
    return False, ("xcb probe failed (exit %d): %s"
                   % (proc.returncode, err[-1] if err else "no output"))


def re_exec_with_library_path(extra_dirs):
    """Restart this process with `extra_dirs` on LD_LIBRARY_PATH.

    The last resort when a library exists but preloading cannot satisfy the
    plugin -- typically because a *transitive* dependency also lives in the
    unusual directory. Re-exec is the only way to give the dynamic linker a
    new search path, since it reads it once at startup.

    Guarded by an environment marker so it can never loop.
    """
    if os.name == "nt" or os.environ.get("_AORTA_REEXEC"):
        return False
    dirs = [d for d in extra_dirs if d and os.path.isdir(d)]
    if not dirs:
        return False
    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
               if p]
    new = [d for d in dirs if d not in current]
    if not new:
        return False
    env_path = os.pathsep.join(current + new)
    os.environ["LD_LIBRARY_PATH"] = env_path
    os.environ["_AORTA_REEXEC"] = "1"
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
    return True                                    # not reached


def probe_display(timeout=5):
    """(ok, detail) for the X display named by $DISPLAY.

    Actually opens a connection rather than trusting the variable: a forwarded
    DISPLAY left over from a dead session is set, looks plausible, and fails
    only when Qt tries to use it.
    """
    display = os.environ.get("DISPLAY", "")
    if not display:
        return False, "$DISPLAY is not set"
    if os.name == "nt":
        return False, "not applicable on Windows"

    import shutil
    import subprocess
    probe = shutil.which("xdpyinfo") or shutil.which("xset")
    if probe:
        args = [probe, "-display", display] if probe.endswith("xdpyinfo") \
            else [probe, "q"]
        try:
            r = subprocess.run(args, capture_output=True, timeout=timeout)
            if r.returncode == 0:
                return True, "%s reachable (%s)" % (display,
                                                    os.path.basename(probe))
            err = (r.stderr or b"").decode("utf-8", "replace").strip()
            return False, "%s unreachable: %s" % (display,
                                                  err.splitlines()[-1] if err
                                                  else "exit %d" % r.returncode)
        except subprocess.TimeoutExpired:
            return False, "%s did not answer within %ds" % (display, timeout)
        except Exception as exc:                               # noqa: BLE001
            return False, "%s probe failed: %s" % (display, exc)

    # No X utilities installed: fall back to opening the socket ourselves.
    try:
        import socket
        host, _, rest = display.partition(":")
        num = rest.split(".")[0] or "0"
        if not host:
            path = "/tmp/.X11-unix/X%s" % num
            if not os.path.exists(path):
                return False, "%s has no socket at %s" % (display, path)
            sock = socket.socket(socket.AF_UNIX)
            sock.settimeout(timeout)
            sock.connect(path)
            sock.close()
            return True, "%s socket accepted a connection" % display
        sock = socket.create_connection((host, 6000 + int(num)), timeout)
        sock.close()
        return True, "%s TCP port open" % display
    except Exception as exc:                                   # noqa: BLE001
        return False, "%s not connectable: %s" % (display, exc)


def display_report():
    """Everything known about this machine's ability to show a Qt window."""
    ok, detail = probe_display()
    missing = missing_xcb_libraries()
    return {
        "display": os.environ.get("DISPLAY", ""),
        "display_ok": ok,
        "display_detail": detail,
        "missing_xcb": missing,
        "missing_required": [m for m in missing if m in XCB_REQUIRED],
        "platform_override": os.environ.get("QT_QPA_PLATFORM", ""),
        "qt_library_dirs": qt_library_dirs(),
    }


def choose_platform(report=None, allow_offscreen=True, interactive=False,
                    ignore_missing_xcb=False):
    """(plugin, reasons) -- which Qt platform plugin can actually start.

    Returns None for the plugin when nothing can produce a visible window, so
    the caller can offer the web viewer instead of starting a GUI no one can
    see.

    `interactive=True` is the desktop-GUI contract: the goal is a real window
    on the user's screen, so xcb is the only acceptable outcome. An inherited
    QT_QPA_PLATFORM naming a headless plugin is *refused* rather than honoured
    (hpc/run_headless.slurm exports exactly that, and inheriting it silently
    produced a GUI with no window), and the last-resort fallback is "minimal"
    rather than "offscreen".
    """
    report = report or display_report()
    reasons = []
    fallback = "minimal" if interactive else "offscreen"

    forced = report.get("platform_override")
    if forced:
        if interactive and forced.split(":")[0] in HEADLESS_PLATFORMS:
            reasons.append(
                "ignoring inherited QT_QPA_PLATFORM=%r: it cannot show a "
                "window, and this is an interactive session" % forced)
            forced = None
        else:
            return forced, ["QT_QPA_PLATFORM is set to %r; honouring it"
                            % forced]

    if os.name == "nt":
        return "windows", ["native Windows platform plugin (not X11)"]
    if sys.platform == "darwin":
        return "cocoa", ["native macOS platform plugin (not X11)"]

    if report["missing_required"]:
        reasons.append(
            "xcb plugin reports missing %s"
            % ", ".join(report["missing_required"]))

        # Try to satisfy it from a copy already on this filesystem -- a Qt
        # wheel's own lib dir, an auditwheel .libs dir, a conda prefix, an
        # EasyBuild module root. Needs no root and no network.
        resolved, notes = ensure_xcb_loadable()
        reasons.extend("  " + n for n in notes)
        if resolved:
            report["missing_required"] = []

    if report["missing_required"]:
        # The static list is a hint, not a verdict: it cannot know which Qt
        # version needs which SONAME. If a display is reachable, let Qt itself
        # decide -- in a child process, because a failed plugin load aborts
        # rather than raising.
        if report["display_ok"] or ignore_missing_xcb:
            ok, detail = probe_xcb_plugin()
            reasons.append("  empirical check: %s" % detail)
            if ok:
                reasons.append(
                    "  xcb works despite the missing-library report; using it")
                report["missing_required"] = []

    if report["missing_required"]:
        reasons.append(
            "no way to load the xcb plugin was found. Options, in order of "
            "least effort:")
        reasons.append(
            "  1. pip install --user PySide6   (its wheel vendors the xcb "
            "libraries, and this module will preload them)")
        reasons.append(
            "  2. module avail xcb / module load ...   (site modules are "
            "searched automatically once loaded)")
        reasons.append(
            "  3. python app.py --web --meshes <dir>   (browser viewer, no "
            "X11 at all)")
        reasons.append(
            "  4. ask the site admin for xcb-util-cursor")
        if not ignore_missing_xcb:
            reasons.append(
                "  --ignore-missing-xcb forces the attempt anyway")
            return (fallback if allow_offscreen else None), reasons
        reasons.append("--ignore-missing-xcb given: attempting xcb regardless")
        return "xcb", reasons

    if not report["display_ok"]:
        reasons.append("no usable X display: %s" % report["display_detail"])
        reasons.append("run with --web to view in a browser over an SSH tunnel")
        return (fallback if allow_offscreen else None), reasons

    reasons.append("xcb usable: %s" % report["display_detail"])
    return "xcb", reasons


def create_application(argv=None, allow_offscreen=False, interactive=False,
                       ignore_missing_xcb=False):
    """(app, info) -- a QApplication that will not abort the process.

    Always returns a 2-tuple: the QApplication (or None when none could be
    created) and an info dict carrying `plugin`, `started_with`, `headless`,
    `report`, `reasons`, and `cleared_env`.

    The platform plugin is decided before construction, because Qt aborts
    rather than raising when a plugin cannot load, and an abort is not
    catchable from Python. The try/except below is still there for the errors
    Qt *does* raise, but it is the second line of defence, not the first.

    `interactive=True` (the desktop GUI) additionally clears inherited
    headless environment variables first, refuses a headless
    QT_QPA_PLATFORM, and falls back to "minimal" rather than "offscreen".
    Neither fallback shows a window -- `info["headless"]` says so, and the
    caller is expected to tell the user rather than pretend the session is
    usable.
    """
    cleared = clear_headless_environment() if interactive else {}

    report = display_report()
    plugin, reasons = choose_platform(report, allow_offscreen=allow_offscreen,
                                      interactive=interactive,
                                      ignore_missing_xcb=ignore_missing_xcb)
    if cleared:
        reasons.insert(0, "cleared inherited headless settings: %s"
                       % ", ".join("%s=%s" % kv for kv in sorted(cleared.items())))

    info = {"report": report, "reasons": list(reasons), "plugin": plugin,
            "headless": plugin in HEADLESS_PLATFORMS,
            "cleared_env": cleared, "interactive": bool(interactive)}

    if plugin is None and not allow_offscreen:
        info["error"] = "no usable Qt platform plugin"
        return None, info

    if plugin:
        os.environ["QT_QPA_PLATFORM"] = plugin

    argv = list(sys.argv if argv is None else argv)
    try:
        app = QtWidgets.QApplication(argv)
        info["started_with"] = os.environ.get("QT_QPA_PLATFORM", "default")
        return app, info
    except Exception as exc:                                   # noqa: BLE001
        info["reasons"].append("QApplication(%s) raised: %s"
                               % (plugin or "default", exc))

    # Interactive sessions never silently degrade to offscreen: "minimal" is
    # the only fallback, and it is reported as headless.
    chain = ("minimal",) if interactive else ("offscreen", "minimal")
    if allow_offscreen:
        for name in chain:
            if name == plugin:
                continue
            os.environ["QT_QPA_PLATFORM"] = name
            try:
                app = QtWidgets.QApplication(argv)
                info["plugin"] = name
                info["headless"] = True
                info["started_with"] = name
                info["reasons"].append(
                    "fell back to the %s plugin (no visible window)" % name)
                return app, info
            except Exception as exc:                           # noqa: BLE001
                info["reasons"].append("%s plugin failed: %s" % (name, exc))

    info["error"] = "QApplication could not be created"
    return None, info
