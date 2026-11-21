# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for Mini CT Viewer.

Build (onedir - default, recommended: fast startup, easy DLL troubleshooting):
    pyinstaller viewer.spec

Build onefile (single .exe, slower first-launch unpack):
    PowerShell : $env:CTVIEWER_ONEFILE = "1"; pyinstaller viewer.spec
    cmd.exe    : set CTVIEWER_ONEFILE=1 && pyinstaller viewer.spec

Output: dist/MiniCTViewer/MiniCTViewer.exe (onedir)
     or dist/MiniCTViewer.exe             (onefile)
"""
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ONEFILE = os.environ.get("CTVIEWER_ONEFILE", "0") == "1"
ROOT = os.path.abspath(os.path.dirname(SPEC))

# --------------------------------------------------------------------------- #
#  Non-code assets. None currently ship with this app (no icon/config/UI
#  files exist yet) - both are wired up so dropping an icon.ico into the
#  project root picks it up automatically, in both the exe and the running
#  app's window/taskbar icon (see ct_viewer.resource_path/app_icon).
# --------------------------------------------------------------------------- #
datas = []
datas += collect_data_files("pydicom")     # UID/charset dictionaries (package data)
datas += collect_data_files("matplotlib")  # mpl-data: fonts, style sheets

icon_path = os.path.join(ROOT, "icon.ico")
if not os.path.isfile(icon_path):
    icon_path = None

sample_data = os.path.join(ROOT, "5119278-1")
if os.path.isdir(sample_data):
    # Sample DICOM series, bundled only so `--selftest` has data to exercise
    # DICOM-read on a clean machine with no other data available.
    datas.append((sample_data, "5119278-1"))

# --------------------------------------------------------------------------- #
#  Hidden imports: dynamic/plugin-style C-extension loads that PyInstaller's
#  static import scan can't see.
#   - vtkmodules: VTK self-registers renderer/mapper/filter implementations
#     through an internal object-factory mechanism, not plain `import`
#     statements, so the whole package is pulled in rather than guessing
#     which of vtkRenderingOpenGL2/vtkRenderingVolumeOpenGL2/vtkFiltersCore/
#     etc. are needed.
#   - skimage.restoration / skimage.metrics: skimage lazy-loads its
#     submodules' compiled Cython pieces (denoise_nl_means, SSIM, ...)
#     through its own lazy-import machinery, which is invisible to
#     PyInstaller's static analysis.
# --------------------------------------------------------------------------- #
hiddenimports = []
hiddenimports += collect_submodules("vtkmodules")
hiddenimports += collect_submodules("skimage.restoration")
hiddenimports += collect_submodules("skimage.metrics")
hiddenimports += [
    "vtkmodules.util.numpy_support",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "phasepack",
    "phasepack.tools",
    "pydicom.pixel_data_handlers.util",
    "nibabel",
]

excludes = ["tkinter", "PyQt6", "PySide2", "PySide6"]

a = Analysis(
    ["ct_viewer.py"],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe_kwargs = dict(
    name="MiniCTViewer",
    debug=False,
    strip=False,
    # UPX-compressing giant sci-Python DLLs (vtk*.dll, opencv_world*.dll) is
    # a common source of "app silently fails to start" and AV false
    # positives - leave binaries uncompressed.
    upx=False,
    console=False,
    icon=icon_path,
)

if ONEFILE:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
        **exe_kwargs,
    )
else:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        **exe_kwargs,
    )
    COLLECT(
        exe, a.binaries, a.zipfiles, a.datas,
        strip=False, upx=False, name="MiniCTViewer",
    )
