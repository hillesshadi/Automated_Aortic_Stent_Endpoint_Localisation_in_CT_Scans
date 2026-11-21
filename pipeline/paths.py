#!/usr/bin/env python3
"""
sys.path bootstrap for the existing pipeline code.

Two directories have to be importable, and neither is a package:

  src/         process_aorta_stents, stent_endpoint_sop, stent_html_export,
               oblique_mpr, and the stentlib/ package
  ct_viewer/   mpr, segmentation, enhancement, metrics

ct_viewer's modules import each other flatly ("import mpr"), so their own
directory must be on sys.path -- adding only the repo root is not enough.

Import this module before importing anything from either directory.
"""

import os
import sys

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(APP_ROOT, "src")
VIEWER_DIR = os.path.join(APP_ROOT, "ct_viewer")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")


def bootstrap():
    """Put src/ and ct_viewer/ on sys.path. Idempotent."""
    for directory in (SRC_DIR, VIEWER_DIR):
        if os.path.isdir(directory) and directory not in sys.path:
            sys.path.insert(0, directory)
    return APP_ROOT


bootstrap()


def ensure_output_dir(*parts):
    path = os.path.join(OUTPUT_DIR, *parts) if parts else OUTPUT_DIR
    os.makedirs(path, exist_ok=True)
    return path
