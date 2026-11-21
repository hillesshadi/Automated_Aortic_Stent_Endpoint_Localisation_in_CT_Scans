#!/usr/bin/env python3
"""
Entry point for the aortic centerline & stent verification app.

Three ways to run it, in the order you should reach for them on a cluster:

    python app.py                                   # desktop GUI (needs X11)
    python app.py --web --meshes results/CASE01     # browser viewer over SSH
    python app.py --headless --volume case.nii.gz --out results/CASE01

The desktop GUI needs a working X server *and* the xcb platform libraries. On
a Mogon node reached through MobaXterm either can be missing, and Qt reports
that by aborting the process rather than raising -- so the display and the
platform plugin are probed before QApplication is constructed, and if neither
can produce a visible window the app says so and points at --web instead of
starting a GUI nobody can see. See gui/qt_compat.py.

--headless and --web never import Qt at all: the offscreen environment must be
set before VTK, PyVista or matplotlib are first imported, so those paths branch
before any of them is touched. See pipeline/offscreen.py.
"""

import argparse
import os
import sys
import traceback


def _excepthook(exc_type, exc, tb):
    """Last-resort handler.

    An uncaught exception on the GUI thread otherwise kills the event loop
    with only a console trace, which a user running a bundled build never
    sees. Show it, then let the app keep running: the failure is usually in
    one callback, not the whole session.
    """
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    sys.__stderr__.write(text)
    try:
        from gui.qt_compat import QtWidgets
        if QtWidgets.QApplication.instance() is not None:
            box = QtWidgets.QMessageBox()
            box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            box.setWindowTitle("Unexpected error")
            box.setText(str(exc))
            box.setDetailedText(text)
            box.exec()
    except Exception:                                          # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
#  Web viewer
# --------------------------------------------------------------------------- #
def _run_web(argv):
    """--web / --export-html: browser viewer, no Qt and no X11."""
    parser = argparse.ArgumentParser(
        prog="app.py --web",
        description="Serve the 3-D scene to a browser (no X11 needed).")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--meshes", metavar="DIR",
                        help="directory of exported meshes to display")
    parser.add_argument("--volume", metavar="FILE",
                        help="run the pipeline on this volume first")
    parser.add_argument("--dicom", metavar="DIR",
                        help="run the pipeline on this DICOM series first")
    parser.add_argument("--centerline", metavar="FILE")
    parser.add_argument("--reference", metavar="FILE")
    parser.add_argument("--out", metavar="DIR", default="results",
                        help="where a pipeline run writes its outputs")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--export-html", metavar="FILE",
                        help="write one self-contained file instead of serving")
    args, _unknown = parser.parse_known_args(argv)

    # Offscreen first: web_viewer imports PyVista, which reads this at import.
    from pipeline import offscreen
    offscreen.configure_headless(verbose=False)
    import web_viewer

    try:
        if args.meshes:
            meshes = web_viewer.discover_meshes(args.meshes)
            print("meshes from %s:" % args.meshes)
        elif args.volume or args.dicom:
            print("no --meshes given; running the pipeline first")
            import run_headless
            forwarded = ["--out", args.out]
            for flag, value in (("--volume", args.volume),
                                ("--dicom", args.dicom),
                                ("--centerline", args.centerline),
                                ("--reference", args.reference)):
                if value:
                    forwarded += [flag, value]
            # Screenshots need a GL context the node may not have, and the
            # browser is about to render the same geometry anyway.
            forwarded.append("--no-screenshots")
            code = run_headless.main(forwarded)
            if code not in (0, 2):
                print("pipeline failed (exit %d); nothing to display" % code)
                return code
            meshes = web_viewer.discover_meshes(args.out)
        else:
            print("error: --web needs --meshes DIR, or --volume/--dicom to "
                  "run the pipeline first", file=sys.stderr)
            return 1

        if args.export_html:
            path = web_viewer.export_html(meshes, args.export_html)
            print("wrote %s (%.1f MB) -- copy it anywhere and open it"
                  % (path, os.path.getsize(path) / 1e6))
            return 0
        web_viewer.serve(meshes, port=args.port, host=args.host)
        return 0
    except web_viewer.WebViewerError as exc:
        print("web viewer: %s" % exc, file=sys.stderr)
        return 1


# --------------------------------------------------------------------------- #
#  Desktop GUI
# --------------------------------------------------------------------------- #
#: Attributes app.py relies on. Checked by name before use so a half-synced
#: checkout reports *which file is stale* instead of raising a bare
#: AttributeError from somewhere in the middle of startup.
_REQUIRED_QT_COMPAT = ("create_application", "clear_headless_environment",
                       "choose_platform", "display_report", "exec_app")


def _check_qt_compat(module):
    """Fail loudly and specifically when gui/qt_compat.py is out of date."""
    missing = [n for n in _REQUIRED_QT_COMPAT if not hasattr(module, n)]
    if not missing:
        return True
    print("\ngui/qt_compat.py is out of date: it is missing %s.\n"
          % ", ".join(missing), file=sys.stderr)
    print("This copy of the tree is only partly synced -- app.py expects a\n"
          "newer gui/qt_compat.py than the one that got copied. Re-sync the\n"
          "whole project directory (not just app.py) and try again.\n",
          file=sys.stderr)
    print("    loaded from : %s" % getattr(module, "__file__", "?"),
          file=sys.stderr)
    return False


def _run_gui(args):
    from gui import qt_compat
    if not _check_qt_compat(qt_compat):
        return 4
    from gui.qt_compat import QtWidgets, QT_API, exec_app

    # interactive=True is the whole contract of this entry point: clear any
    # inherited headless settings, insist on xcb, and refuse to quietly start
    # a windowless session. hpc/run_headless.slurm exports QT_QPA_PLATFORM=
    # offscreen and PYVISTA_OFF_SCREEN=true; a shell that has sourced it would
    # otherwise launch this GUI with no window and no explanation.
    app, info = qt_compat.create_application(
        sys.argv, allow_offscreen=args.force_gui, interactive=True,
        ignore_missing_xcb=args.ignore_missing_xcb or args.force_gui)

    print("Qt binding      : %s" % QT_API)
    print("platform plugin : %s" % (info.get("started_with") or info["plugin"]))
    print("DISPLAY         : %s" % (os.environ.get("DISPLAY") or "(unset)"))
    for line in info["reasons"]:
        print("  %s" % line)

    if app is None:
        print("\nThe desktop GUI cannot start on this machine.\n", file=sys.stderr)
        report = info["report"]
        if report["missing_required"]:
            print("Missing Qt platform libraries: %s"
                  % ", ".join(report["missing_required"]), file=sys.stderr)
        print("Use the browser viewer instead -- it renders on your laptop and "
              "needs no X11:\n", file=sys.stderr)
        print("    python app.py --web --meshes <results-dir>\n", file=sys.stderr)
        print("or run the pipeline in batch:\n", file=sys.stderr)
        print("    python app.py --headless --volume <case.nii.gz> "
              "--out <results-dir>\n", file=sys.stderr)
        print("Pass --force-gui to start anyway on the offscreen plugin "
              "(no visible window; useful only for smoke tests).",
              file=sys.stderr)
        return 3

    if info["headless"]:
        print("\nWARNING: running on the %s plugin -- there will be no visible\n"
              "window. This is a smoke test, not a usable session. For a real\n"
              "3-D view without X11, use:  python app.py --web --meshes <dir>\n"
              % info["plugin"])

    from gui.main_window import MainWindow, DARK_QSS
    sys.excepthook = _excepthook

    app.setApplicationName("Aortic Stent GUI")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)

    window = MainWindow()
    window.show()
    window.log.info("Started with %s on the %s platform plugin."
                    % (QT_API, info.get("started_with") or "default"))
    for line in info["reasons"]:
        window.log.info("  " + line)

    # Deferred so the window is painted before a multi-second read begins;
    # loading inline here would show a blank frame until it finished.
    if args.dicom:
        window.controls.dicom_row.set_path(args.dicom)
        QtWidgets.QApplication.processEvents()
        window.on_load("dicom", args.dicom)
    elif args.volume:
        window.controls.image_row.set_path(args.volume)
        QtWidgets.QApplication.processEvents()
        window.on_load("image", args.volume)
    if args.mesh:
        window.controls.mesh_row.set_path(args.mesh)

    return exec_app(app)


# --------------------------------------------------------------------------- #
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Both alternative modes are checked before argparse and before any GUI
    # import, so neither ever touches Qt.
    if "--headless" in argv:
        argv.remove("--headless")
        import run_headless
        return run_headless.main(argv)

    if "--web" in argv or "--export-html" in argv:
        return _run_web(argv)

    parser = argparse.ArgumentParser(
        description="Aortic centerline & stent verification (desktop GUI)",
        epilog="Other modes: --headless runs the batch pipeline (see "
               "run_headless.py --help); --web serves an interactive 3-D scene "
               "to a browser over an SSH tunnel, which needs no X11.")
    parser.add_argument("--headless", action="store_true",
                        help="run the batch pipeline instead of the GUI")
    parser.add_argument("--web", action="store_true",
                        help="serve the 3-D scene to a browser instead")
    parser.add_argument("--export-html", metavar="FILE",
                        help="write a self-contained interactive HTML file")
    parser.add_argument("--force-gui", action="store_true",
                        help="start the GUI even with no usable display "
                             "(minimal plugin; no visible window)")
    parser.add_argument("--ignore-missing-xcb", action="store_true",
                        help="attempt the xcb plugin even when a required "
                             "library looks missing -- the static check is "
                             "only a hint, and a preloaded or vendored copy "
                             "often satisfies Qt anyway")
    parser.add_argument("--check-display", action="store_true",
                        help="report what this machine can show, then exit")
    parser.add_argument("--dicom", metavar="DIR",
                        help="DICOM series directory to load on startup")
    parser.add_argument("--volume", metavar="FILE",
                        help="NIfTI/NRRD volume to load on startup")
    parser.add_argument("--mesh", metavar="FILE",
                        help="STL/VTK surface mesh to load on startup")
    args = parser.parse_args(argv)

    if args.check_display:
        from gui import qt_compat
        report = qt_compat.display_report()
        plugin, reasons = qt_compat.choose_platform(
            report, interactive=True,
            ignore_missing_xcb=args.ignore_missing_xcb)
        for key, value in report.items():
            print("%-18s %s" % (key, value))
        print("%-18s %s" % ("chosen plugin", plugin))
        for line in reasons:
            print("   - %s" % line)
        return 0 if plugin not in (None, "offscreen", "minimal") else 2

    return _run_gui(args)


if __name__ == "__main__":
    sys.exit(main())
