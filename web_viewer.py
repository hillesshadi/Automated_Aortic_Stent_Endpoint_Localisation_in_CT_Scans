#!/usr/bin/env python3
"""
Browser-based 3-D viewer, for when X11 forwarding is not usable.

Two modes, both driven from app.py:

    python app.py --web  --meshes results/CASE01          # live server
    python app.py --web  --volume case.nii.gz --out out/  # run, then serve
    python app.py --export-html results/CASE01/view.html --meshes results/CASE01

`--web` starts a Trame server: the geometry is serialised to the browser and
rendered there with vtk.js, so the *client* machine does the OpenGL work. That
is the point -- a Mogon compute node with no GLX driver and a MobaXterm X
server that cannot allocate windows can still serve a fully interactive scene
to the browser on your laptop over an SSH tunnel.

`--export-html` writes one self-contained file instead. No server, no tunnel,
nothing to keep running: copy it anywhere and open it. Prefer this for sharing
a result with someone; prefer --web for looking at a case while it is on the
cluster.

Neither path needs a working render window on the node. PyVista is put in
off-screen mode before it is imported, and the geometry is handed to the
browser as data rather than as pixels.
"""

import glob
import os
import socket
import sys

# Off-screen before PyVista is imported: it reads this at import time, and on a
# node with no display an on-screen plotter would try to open a window.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

MESH_ROLES = (
    # (filename suffix, display name, colour, opacity)
    ("_aorta", "Aorta", "#db4347", 0.42),
    ("_stent", "Stent", "#d9dee6", 1.00),
    ("_centerline", "Centerline", "#ffd91a", 1.00),
    ("_preview", "Preview surface", "#9ea8c4", 0.35),
)
MESH_EXTENSIONS = (".stl", ".vtp", ".vtk", ".ply", ".obj")
DEFAULT_PORT = 8080


class WebViewerError(RuntimeError):
    pass


def _require_pyvista():
    try:
        import pyvista
        return pyvista
    except ImportError as exc:                                 # noqa: BLE001
        raise WebViewerError(
            "PyVista is required for the web viewer.\n\n"
            "    pip install 'pyvista[all]' trame trame-vtk trame-vuetify\n\n"
            "(%s)" % exc) from exc


# --------------------------------------------------------------------------- #
#  Loading geometry
# --------------------------------------------------------------------------- #
def discover_meshes(directory):
    """Map role -> mesh path for the files run_headless.py writes.

    Roles are matched on the filename suffix rather than an exact case name, so
    a directory holding several cases still resolves, and .vtk is preferred
    over .stl for the centerline because only the former keeps it a polyline.
    """
    if not os.path.isdir(directory):
        raise WebViewerError("Not a directory:\n%s" % directory)

    found = {}
    for suffix, label, _colour, _opacity in MESH_ROLES:
        for ext in MESH_EXTENSIONS:
            hits = sorted(glob.glob(os.path.join(directory, "*%s%s"
                                                 % (suffix, ext))))
            if hits:
                found[label] = hits[0]
                break
    if not found:
        raise WebViewerError(
            "No meshes found in:\n%s\n\nExpected files ending in %s with one "
            "of %s -- the outputs run_headless.py writes."
            % (directory, ", ".join(s for s, _l, _c, _o in MESH_ROLES),
               ", ".join(MESH_EXTENSIONS)))
    return found


def build_plotter(meshes, title="Aortic centerline & stent", window_size=None):
    """A PyVista plotter holding the named meshes.

    `meshes` maps display name -> path or PolyData. Colours follow the desktop
    viewer so the browser scene and the Qt scene are recognisably the same.
    """
    pv = _require_pyvista()
    palette = {label: (colour, opacity)
               for _s, label, colour, opacity in MESH_ROLES}

    plotter = pv.Plotter(off_screen=True,
                         window_size=list(window_size or (1400, 1000)))
    plotter.set_background("#0e1219", top="#232838")

    added = 0
    for label, source in meshes.items():
        try:
            mesh = pv.read(source) if isinstance(source, str) else source
        except Exception as exc:                               # noqa: BLE001
            print("  skipped %s (%s)" % (label, exc))
            continue
        if mesh is None or mesh.n_points == 0:
            print("  skipped %s (empty)" % label)
            continue
        colour, opacity = palette.get(label, ("#9aa6bd", 1.0))
        plotter.add_mesh(mesh, color=colour, opacity=opacity, smooth_shading=True,
                         specular=0.25, specular_power=30, name=label,
                         label=label)
        added += 1
        print("  %-16s %8d points  %8d cells" % (label, mesh.n_points,
                                                 mesh.n_cells))
    if not added:
        raise WebViewerError("Nothing could be loaded into the scene.")

    # Translucent surfaces need order-independent transparency here for the
    # same reason the desktop viewer does: without it the stent inside the
    # aorta is drawn in whatever order happens to win.
    try:
        plotter.enable_depth_peeling(8)
    except Exception:                                          # noqa: BLE001
        pass
    plotter.add_axes()
    plotter.view_isometric()
    plotter.add_text(title, font_size=10, color="#c8d3e6")
    return plotter


def from_context(ctx, target_triangles=150000):
    """Build the mesh set straight from an in-memory CaseContext."""
    pv = _require_pyvista()
    from pipeline import surfaces

    meshes = {}
    for name, label, mask in (("aorta", "Aorta", getattr(ctx, "aorta_mask", None)),
                              ("stent", "Stent", getattr(ctx, "stent_mask", None))):
        poly = surfaces.build_surface(mask, ctx.geom, target_triangles)
        if poly is not None:
            meshes[label] = pv.wrap(poly)
    if getattr(ctx, "resampler", None) is not None:
        tube = surfaces.tube_polydata(ctx.resampler.points)
        if tube is not None:
            meshes["Centerline"] = pv.wrap(tube)
    if not meshes:
        raise WebViewerError(
            "This case has no geometry yet -- run preprocessing first.")
    return meshes


# --------------------------------------------------------------------------- #
#  Static export
# --------------------------------------------------------------------------- #
def export_html(meshes, path, title="Aortic centerline & stent"):
    """Write one self-contained HTML file with the scene embedded."""
    pv = _require_pyvista()
    plotter = build_plotter(meshes, title=title)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        plotter.export_html(path)
    finally:
        try:
            plotter.close()
            pv.close_all()
        except Exception:                                      # noqa: BLE001
            pass
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        raise WebViewerError("export_html produced no usable file at %s" % path)
    return path


# --------------------------------------------------------------------------- #
#  Live server
# --------------------------------------------------------------------------- #
def _free_port(preferred, host="127.0.0.1", tries=20):
    """First free port at or after `preferred`.

    Several people sharing a login node will otherwise collide, and the error
    Trame gives for a taken port is not obviously about ports.
    """
    for offset in range(tries):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise WebViewerError("No free port in %d-%d." % (preferred,
                                                     preferred + tries - 1))


def tunnel_banner(port, host=None):
    """The ssh command needed to reach this server from a laptop."""
    node = host or socket.gethostname()
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "USER"
    job = os.environ.get("SLURM_JOB_ID")
    lines = [
        "=" * 68,
        "  Web viewer ready.",
        "",
        "  On the compute/login node : http://localhost:%d" % port,
        "  Node                      : %s%s" % (node,
                                                "  (job %s)" % job if job else ""),
        "",
        "  From your laptop, open a tunnel and then browse to",
        "  http://localhost:%d" % port,
        "",
        "      ssh -N -L %d:%s:%d %s@mogon-nhr01.zdv.uni-mainz.de"
        % (port, node, port, user),
        "",
        "  In MobaXterm the same thing is Tunneling -> New SSH tunnel,",
        "  Local port %d  ->  %s : %d" % (port, node, port),
        "",
        "  Rendering happens in your browser, so this needs no OpenGL on",
        "  the node and no X11 forwarding at all.",
        "",
        "  Ctrl-C to stop.",
        "=" * 68,
    ]
    return "\n".join(lines)


def serve(meshes, port=DEFAULT_PORT, host="0.0.0.0", title="Aortic stent viewer",
          open_browser=False):
    """Start a Trame server rendering `meshes` in the browser.

    Falls back to writing a static HTML file if Trame cannot start, because a
    viewer you can open is worth more than an error about a web framework.
    """
    pv = _require_pyvista()
    try:
        from trame.app import get_server
        from trame.ui.vuetify3 import SinglePageLayout
        from trame.widgets import vuetify3 as vuetify
        from pyvista.trame.ui import plotter_ui
    except Exception as exc:                                   # noqa: BLE001
        raise WebViewerError(
            "Trame is not usable (%s).\n\n"
            "    pip install trame trame-vtk trame-vuetify\n\n"
            "Or use --export-html, which needs no server." % exc) from exc

    bind_port = _free_port(port, host="127.0.0.1")
    if bind_port != port:
        print("port %d was busy; using %d" % (port, bind_port))

    plotter = build_plotter(meshes, title=title)
    server = get_server(client_type="vue3")

    with SinglePageLayout(server) as layout:
        layout.title.set_text(title)
        with layout.toolbar:
            vuetify.VSpacer()
            vuetify.VLabel("drag to rotate  ·  scroll to zoom  ·  "
                           "middle-drag to pan")
        with layout.content:
            with vuetify.VContainer(fluid=True,
                                    classes="pa-0 fill-height"):
                plotter_ui(plotter)

    print(tunnel_banner(bind_port))
    sys.stdout.flush()
    try:
        server.start(port=bind_port, host=host, open_browser=open_browser,
                     show_connection_info=False)
    except KeyboardInterrupt:
        print("\nweb viewer stopped.")
    finally:
        try:
            plotter.close()
            pv.close_all()
        except Exception:                                      # noqa: BLE001
            pass
    return bind_port


# --------------------------------------------------------------------------- #
def main(argv=None):
    """Standalone entry point: python web_viewer.py <mesh-dir> [--port N]."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Browser 3-D viewer for pipeline outputs.")
    parser.add_argument("meshes", help="directory of exported meshes")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--export-html", metavar="FILE",
                        help="write a self-contained file instead of serving")
    args = parser.parse_args(argv)

    found = discover_meshes(args.meshes)
    print("loaded from %s:" % args.meshes)
    if args.export_html:
        path = export_html(found, args.export_html)
        print("wrote %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))
        return 0
    serve(found, port=args.port, host=args.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
