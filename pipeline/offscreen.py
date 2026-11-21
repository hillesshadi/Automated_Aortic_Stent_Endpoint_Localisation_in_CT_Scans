#!/usr/bin/env python3
"""
Headless rendering: environment setup, backend selection, offscreen renderer.

Why this is not just "set two environment variables"
----------------------------------------------------
On a Slurm compute node there is no X server and $DISPLAY is either unset or
points at a forwarded socket that dies with the login shell. VTK's default
render window then tries, in order, to open an X display and to bring up EGL,
and reports the failures the user sees:

    vtkEGLRenderWindow: Failed to initialize OpenGL functions!
    Could not create shader object
    bad X server connection

Environment variables alone cannot fix that, because whether a backend works
depends on what the *wheel* was compiled with and what the *driver* exposes,
neither of which a variable changes:

  * the PyPI ``vtk`` wheel is built with X11/GLX (and, since 9.2, EGL);
  * the ``vtk-osmesa`` wheel is built for software rendering with no display;
  * an A100 in a Slurm cgroup exposes /dev/dri only if the job requested a GPU
    and the container/driver stack passes it through.

So this module *probes*: it builds each candidate render window, renders a
known scene, reads the pixels back, and accepts the backend only if the image
is actually non-blank. That is the same "smoke test rather than trust a flag"
approach that was needed to pick the working Slicer binary earlier in this
project -- a render window that constructs without error but produces a black
image is the exact failure mode being guarded against.

Degradation is deliberate. Every mesh artifact (.stl/.ply/.vtk) and every 2-D
figure comes from CPU-side code in pipeline/surfaces.py and pipeline/report.py,
neither of which needs a GL context. If no backend works the pipeline still
produces all of those and reports "screenshots skipped" -- it does not abort a
batch job over a picture.
"""

import os
import shutil
import subprocess
import sys
import time

# Order matters: OSMesa is pure software and cannot fail for driver reasons;
# EGL is the right choice on a GPU node with no X; the generic window is the
# X11/GLX path and only works if a display (real or Xvfb) exists.
BACKENDS = ("osmesa", "egl", "generic")

_STATE = {"configured": False, "backend": None, "xvfb": None, "notes": [],
          "windows": [], "window_created": False}


# --------------------------------------------------------------------------- #
#  Environment
# --------------------------------------------------------------------------- #
def configure_headless(verbose=True):
    """Set the offscreen environment. MUST run before VTK/PyVista are imported.

    Importing vtkRenderingOpenGL2 is what creates the render-window factory, so
    variables set after that import are read too late to matter.
    """
    if _STATE["configured"]:
        return _STATE

    for module in ("vtk", "vtkmodules.vtkRenderingOpenGL2", "pyvista"):
        if module in sys.modules:
            _note("WARNING: %s was already imported before configure_headless();"
                  " offscreen settings may not take effect." % module)

    # PyVista's own switch. Note the spelling: PyVista reads PYVISTA_OFF_SCREEN
    # (all caps). Linux environment variables are case-sensitive, so a variant
    # such as "PyVISTA_OFF_SCREEN" is silently ignored.
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    os.environ.setdefault("PYVISTA_KILL_DISPLAY", "true")

    # Requested in the issue report. VTK's documented selector is
    # VTK_DEFAULT_OPENGL_WINDOW (it names a render-window class); the
    # HEADLESS variant is carried too because it is harmless and some builds
    # and downstream wrappers honour it.
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_HEADLESS", "1")

    # Matplotlib must not try to load a Qt backend on a node with no display.
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Qt is never constructed in headless mode, but if some dependency imports
    # it anyway, offscreen keeps it from touching the display.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Stdout under Slurm is a pipe, not a terminal, so Python picks the locale
    # encoding -- ASCII under LANG=C, cp1252 on Windows. The pipeline's progress
    # messages contain "->" arrows and ellipses as real Unicode, and writing one
    # to an ASCII stream raises UnicodeEncodeError and kills the job. Degrade
    # those characters instead of losing the run over a log line.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:                                      # noqa: BLE001
            pass

    # Last-resort teardown: if the run exits by a path that misses the
    # explicit cleanup, this still releases the windows before the
    # interpreter starts finalizing objects that hold GL handles.
    import atexit
    atexit.register(safe_cleanup)

    _STATE["configured"] = True
    if verbose:
        for line in _STATE["notes"]:
            print(line)
    return _STATE


def _note(text):
    _STATE["notes"].append(text)
    return text


def notes():
    return list(_STATE["notes"])


# --------------------------------------------------------------------------- #
#  Xvfb
# --------------------------------------------------------------------------- #
def start_xvfb(display=99, width=1920, height=1080, depth=24, wait=2.0):
    """Start a virtual framebuffer and point $DISPLAY at it.

    pyvista.start_xvfb() is used when the installed PyVista still provides it.
    It was removed in PyVista 0.44, so on current versions calling it raises
    AttributeError -- hence the getattr guard and the manual fallback.
    """
    if os.name == "nt":
        return _note("Xvfb: not applicable on Windows.") and False

    try:
        import pyvista
        starter = getattr(pyvista, "start_xvfb", None)
        if callable(starter):
            starter(wait=wait)
            _note("Xvfb: started via pyvista.start_xvfb().")
            return True
    except Exception as exc:                                   # noqa: BLE001
        _note("Xvfb: pyvista.start_xvfb() unavailable (%s); starting manually."
              % exc)

    if shutil.which("Xvfb") is None:
        _note("Xvfb: binary not found on PATH.")
        return False

    try:
        proc = subprocess.Popen(
            ["Xvfb", ":%d" % display, "-screen", "0",
             "%dx%dx%d" % (width, height, depth), "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:                                   # noqa: BLE001
        _note("Xvfb: could not start (%s)." % exc)
        return False

    time.sleep(wait)
    if proc.poll() is not None:
        _note("Xvfb: exited immediately (display :%d may be in use)." % display)
        return False

    _STATE["xvfb"] = proc
    os.environ["DISPLAY"] = ":%d" % display
    _note("Xvfb: running on :%d (%dx%d), DISPLAY set." % (display, width, height))
    return True


def stop_xvfb():
    proc = _STATE.get("xvfb")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:                                      # noqa: BLE001
            proc.kill()
    _STATE["xvfb"] = None


def in_xvfb_run():
    """True when the process was launched under xvfb-run."""
    return bool(os.environ.get("XVFB_RUN") or
                (os.environ.get("DISPLAY") and
                 os.environ.get("XAUTHORITY", "").find("xvfb") >= 0))


# --------------------------------------------------------------------------- #
#  Backend probing
# --------------------------------------------------------------------------- #
def has_osmesa():
    """True when libOSMesa can actually be dlopen'd.

    The class vtkOSOpenGLRenderWindow can be present in a VTK build whose
    libOSMesa shared object is not installed on the node. Constructing and
    rendering into that window is not a catchable Python error -- the loader
    aborts the process, which is the exit crash this guard exists to avoid.
    Checking first means the backend is skipped rather than survived.
    """
    if os.name == "nt":
        return False
    import ctypes
    import ctypes.util
    name = ctypes.util.find_library("OSMesa")
    candidates = [name] if name else []
    candidates += ["libOSMesa.so.8", "libOSMesa.so.6", "libOSMesa.so"]
    for lib in candidates:
        try:
            ctypes.CDLL(lib)
            return True
        except OSError:
            continue
    return False


def _make_window(kind):
    """Construct an offscreen render window of the requested kind, or None."""
    import vtk
    if kind == "osmesa":
        if not has_osmesa():
            _note("osmesa: libOSMesa not found on this node; skipping that "
                  "backend rather than risking a loader abort.")
            return None
        cls = getattr(vtk, "vtkOSOpenGLRenderWindow", None)
    elif kind == "egl":
        cls = getattr(vtk, "vtkEGLRenderWindow", None)
    else:
        cls = vtk.vtkRenderWindow
    if cls is None:
        return None
    window = cls()
    window.SetOffScreenRendering(1)
    window.SetSize(300, 300)
    try:
        window.SetShowWindow(False)
    except AttributeError:
        pass          # not present on every render-window subclass
    _STATE.setdefault("windows", []).append(window)
    _STATE["window_created"] = True
    return window


def safe_finalize(window):
    """Release one render window without letting teardown raise.

    SystemError is called out because that is what a VTK Python wrapper
    raises when its C++ object has already been destroyed -- the usual shape
    of a failure during interpreter shutdown.
    """
    if window is None:
        return False
    for method in ("Finalize", "ReleaseGraphicsResources"):
        try:
            fn = getattr(window, method, None)
            if fn is None:
                continue
            fn(window) if method == "ReleaseGraphicsResources" else fn()
        except (SystemError, RuntimeError, AttributeError, TypeError):
            pass
        except Exception:                                      # noqa: BLE001
            pass
    return True


def safe_cleanup(verbose=False):
    """Close PyVista plotters and VTK render windows, swallowing teardown errors.

    Called from a finally block and again from atexit. It catches Python-level
    failures only: a genuine SIGSEGV from a half-initialised GL driver is a
    native crash that no except clause can intercept, which is why
    has_osmesa() avoids creating such a window in the first place and why
    hard_exit() exists for the shutdown path.
    """
    problems = []

    try:
        import pyvista
    except Exception:                                          # noqa: BLE001
        pass
    else:
        try:
            pyvista.close_all()
        except (SystemError, RuntimeError, AttributeError) as exc:
            problems.append("pyvista.close_all(): %s" % exc)
        except Exception as exc:                               # noqa: BLE001
            problems.append("pyvista.close_all(): %s" % exc)

    for window in list(_STATE.get("windows") or []):
        safe_finalize(window)
    _STATE["windows"] = []

    try:
        stop_xvfb()
    except Exception as exc:                                   # noqa: BLE001
        problems.append("stop_xvfb(): %s" % exc)

    for line in problems:
        _note("cleanup: " + line)
        if verbose:
            print("cleanup: " + line)
    return not problems


def hard_exit(code):
    """Exit without running interpreter finalization.

    A VTK render window whose GL context was never fully brought up can crash
    while its destructor runs at interpreter shutdown -- after every output
    file has already been written. Slurm then records the job as failed (exit
    139) even though the run succeeded. Flushing explicitly and leaving via
    os._exit skips the finalizers that would crash.
    """
    try:
        sys.stdout.flush()
    except Exception:                                          # noqa: BLE001
        pass
    try:
        sys.stderr.flush()
    except Exception:                                          # noqa: BLE001
        pass
    os._exit(int(code))


def created_window():
    """Whether a GL render window was ever constructed in this process.

    Deliberately a sticky flag rather than a check of the live window list:
    cleanup empties that list, and the exit decision is made *after* cleanup.
    What matters is whether a GL context existed at any point, because that is
    what leaves objects whose destructors can crash at interpreter shutdown.
    """
    return bool(_STATE.get("window_created"))


def _smoke_test(window):
    """Render a known sphere and confirm the pixels are not blank.

    A render window that constructs and renders without raising can still hand
    back an all-black image when shader compilation failed -- which is exactly
    what "Could not create shader object" produces. Counting lit pixels is the
    only reliable check.
    """
    import vtk
    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy

    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(24)
    sphere.SetPhiResolution(24)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 1.0, 1.0)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.AddActor(actor)
    renderer.ResetCamera()
    window.AddRenderer(renderer)
    window.Render()

    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    grab.ReadFrontBufferOff()
    grab.Update()
    image = grab.GetOutput()
    dims = image.GetDimensions()
    arr = vtk_to_numpy(image.GetPointData().GetScalars())
    lit = int((arr.max(axis=1) > 20).sum()) if arr.ndim == 2 else 0
    window.RemoveRenderer(renderer)
    return lit, dims


def _probe_main(kind):
    """Child-process entry point: smoke-test one backend, report by exit code.

    Run via `python -m pipeline.offscreen --probe <kind>`. Living in a child
    is the point: if the backend aborts the process rather than raising, only
    the child dies and the parent reads the non-zero status.
    """
    configure_headless(verbose=False)
    try:
        window = _make_window(kind)
        if window is None:
            sys.stdout.write("SKIP unavailable\n")
            return 3
        lit, dims = _smoke_test(window)
        safe_finalize(window)
        sys.stdout.write("LIT %d %d %d\n" % (lit, dims[0], dims[1]))
        return 0 if lit > 100 else 4
    except Exception as exc:                                   # noqa: BLE001
        first = str(exc).strip().splitlines()
        sys.stdout.write("ERR %s\n" % (first[0] if first else type(exc).__name__))
        return 5


def _probe_in_subprocess(kind, timeout=120):
    """(ok, detail) for one backend, tested out-of-process."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-c",
           "import sys; sys.path.insert(0, %r);"
           "from pipeline import offscreen;"
           "sys.exit(offscreen._probe_main(%r))" % (root, kind)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=dict(os.environ))
    except subprocess.TimeoutExpired:
        return False, "%s: probe timed out after %ds" % (kind, timeout)
    except Exception as exc:                                   # noqa: BLE001
        return False, "%s: probe could not run (%s)" % (kind, exc)

    out = (proc.stdout or "").strip().splitlines()
    last = out[-1] if out else ""
    if proc.returncode == 0 and last.startswith("LIT"):
        return True, "%s: %s" % (kind, last)
    if proc.returncode < 0:
        # Killed by a signal -- exactly the libOSMesa / bad-driver crash. The
        # parent survives because the crash happened over there.
        return False, ("%s: probe crashed with signal %d (this is the crash "
                       "that would otherwise kill the job)"
                       % (kind, -proc.returncode))
    if last.startswith(("SKIP", "ERR")):
        return False, "%s: %s" % (kind, last)
    err = (proc.stderr or "").strip().splitlines()
    return False, "%s: exit %d %s" % (kind, proc.returncode,
                                      err[-1] if err else "")


def select_backend(allow_xvfb=True, verbose=True, in_process=False):
    """Pick a working offscreen backend, or None if none renders.

    Each candidate is smoke-tested in a child process by default. A backend
    whose driver aborts instead of raising then kills only the child, and this
    process carries on to the next candidate -- the alternative is losing a
    batch job that had already produced every mesh and figure. Pass
    in_process=True to test inline (faster, but a bad backend is fatal).

    Returns the backend name. Call configure_headless() first.
    """
    if _STATE["backend"] is not None:
        return _STATE["backend"]

    try:
        import vtk  # noqa: F401
    except ImportError as exc:
        _note("VTK is not importable (%s); 3-D rendering disabled." % exc)
        return None

    tried = []
    for kind in BACKENDS:
        if kind == "generic" and not os.environ.get("DISPLAY") and allow_xvfb:
            # The generic window is the X11 path; give it a display to talk to.
            start_xvfb()

        if not in_process:
            ok, detail = _probe_in_subprocess(kind)
            if not ok:
                tried.append(detail)
                continue
            _note("Render backend: %s (probed out-of-process -- %s)."
                  % (kind, detail))
            _STATE["backend"] = kind
            if verbose:
                print(_STATE["notes"][-1])
            return kind

        try:
            window = _make_window(kind)
            if window is None:
                tried.append("%s: unavailable in this VTK build" % kind)
                continue
            lit, dims = _smoke_test(window)
            safe_finalize(window)
            if lit > 100:
                _note("Render backend: %s (%dx%d, %d lit pixels)."
                      % (kind, dims[0], dims[1], lit))
                _STATE["backend"] = kind
                if verbose:
                    print(_STATE["notes"][-1])
                return kind
            tried.append("%s: rendered blank (%d lit pixels)" % (kind, lit))
        except Exception as exc:                               # noqa: BLE001
            tried.append("%s: %s" % (kind, str(exc).strip().splitlines()[0]
                                     if str(exc).strip() else type(exc).__name__))

    _note("No working offscreen render backend. Tried:")
    for line in tried:
        _note("    " + line)
    _note("Meshes and 2-D figures will still be written; 3-D screenshots will "
          "be skipped. To enable them, install the software-rendering build:")
    _note("    pip install --force-reinstall vtk-osmesa "
          "--extra-index-url https://wheels.vtk.org")
    if verbose:
        for line in _STATE["notes"][-len(tried) - 4:]:
            print(line)
    return None


def backend():
    return _STATE["backend"]


# --------------------------------------------------------------------------- #
#  Offscreen renderer
# --------------------------------------------------------------------------- #
class OffscreenRenderer:
    """Renders polydata to PNG with no display, mirroring the desktop viewer.

    Colours, opacities, depth peeling and camera framing are kept identical to
    gui/widgets/viewer_3d.py on purpose: a batch screenshot should be
    recognisably the same picture an operator would see interactively.
    """

    COLORS = {
        "aorta": (0.86, 0.26, 0.28),
        "stent": (0.85, 0.87, 0.90),
        "centerline": (1.00, 0.85, 0.10),
        "preview": (0.62, 0.66, 0.78),
    }

    def __init__(self, size=(1400, 1100), backend_name=None):
        import vtk
        self.vtk = vtk
        self.size = size
        self.actors = {}

        self.window = _make_window(backend_name or _STATE["backend"] or "generic")
        if self.window is None:
            raise RuntimeError("No offscreen render window available.")
        self.window.SetSize(int(size[0]), int(size[1]))

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.07, 0.08, 0.11)
        self.renderer.SetBackground2(0.15, 0.17, 0.23)
        self.renderer.GradientBackgroundOn()
        self.window.AddRenderer(self.renderer)

        # Order-independent transparency, as in the interactive viewer: the
        # aorta is translucent so the stent inside it stays visible, and
        # without peeling the draw order decides what appears in front.
        self.window.SetAlphaBitPlanes(1)
        self.window.SetMultiSamples(0)
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(8)
        self.renderer.SetOcclusionRatio(0.05)

    # ---------------------------------------------------------------- actors
    def add_surface(self, name, poly, opacity=1.0, color=None):
        if poly is None or poly.GetNumberOfPoints() == 0:
            return False
        vtk = self.vtk
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*(color or self.COLORS.get(name, (0.8,) * 3)))
        actor.GetProperty().SetOpacity(float(opacity))
        actor.GetProperty().SetSpecular(0.25)
        actor.GetProperty().SetSpecularPower(30)
        self.renderer.AddActor(actor)
        self.actors[name] = actor
        return True

    def set_visible(self, name, visible):
        actor = self.actors.get(name)
        if actor is not None:
            actor.SetVisibility(bool(visible))

    def reset_camera(self, azimuth=30.0, elevation=15.0, zoom=1.0):
        """Frame all actors from an anterior-oblique viewpoint.

        The interactive viewer zooms 1.35x after ResetCamera because its panel
        is taller than it is wide. A batch frame is landscape, and the aorta is
        the tall axis, so the same factor clips the vessel top and bottom --
        default to no extra zoom here and let --zoom override it.
        """
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0.0, -1.0, 0.0)
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetViewUp(0.0, 0.0, 1.0)          # patient superior is up
        cam.Azimuth(azimuth)
        cam.Elevation(elevation)
        cam.OrthogonalizeViewUp()
        self.renderer.ResetCamera()
        cam.Zoom(zoom)

    def screenshot(self, path, magnification=1):
        """Render and write a PNG. Returns the path, or None if it came out blank."""
        vtk = self.vtk
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy

        self.window.Render()
        grab = vtk.vtkWindowToImageFilter()
        grab.SetInput(self.window)
        if magnification > 1 and hasattr(grab, "SetScale"):
            grab.SetScale(int(magnification))
        # Read the back buffer: the front buffer can hold a stale frame on an
        # offscreen window, which is how corrupted screenshots arise.
        grab.ReadFrontBufferOff()
        grab.Update()

        arr = vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars())
        lit = int((arr.max(axis=1) > 30).sum()) if arr.ndim == 2 else 0
        if lit < 50:
            return None

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(grab.GetOutputPort())
        writer.Write()
        return path

    def orbit(self, path_template, n=4, elevation=15.0, zoom=1.0):
        """Screenshots around the object -- one view rarely shows a bent aorta."""
        out = []
        for i in range(n):
            self.reset_camera(azimuth=30.0 + i * (360.0 / n),
                              elevation=elevation, zoom=zoom)
            p = self.screenshot(path_template % i)
            if p:
                out.append(p)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    def close(self):
        """Release actors and the render window, in that order.

        Dropping the renderer's actors before finalizing the window means the
        mappers release their GL resources while a context still exists;
        leaving them to the garbage collector is what produces a crash during
        interpreter shutdown.
        """
        try:
            if self.renderer is not None:
                self.renderer.RemoveAllViewProps()
        except Exception:                                      # noqa: BLE001
            pass
        self.actors = {}
        try:
            if self.window is not None and self.renderer is not None:
                self.window.RemoveRenderer(self.renderer)
        except Exception:                                      # noqa: BLE001
            pass
        safe_finalize(self.window)
        try:
            windows = _STATE.get("windows") or []
            if self.window in windows:
                windows.remove(self.window)
        except Exception:                                      # noqa: BLE001
            pass
        self.renderer = None
        self.window = None


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # `python -m pipeline.offscreen --probe egl` -- the out-of-process probe.
    if len(sys.argv) >= 3 and sys.argv[1] == "--probe":
        sys.exit(_probe_main(sys.argv[2]))
    configure_headless(verbose=True)
    print("libOSMesa available:", has_osmesa())
    print("selected backend   :", select_backend(verbose=True))
    safe_cleanup(verbose=True)
