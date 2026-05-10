# Automated_Aortic_Stent_Endpoint_Localisation_in_CT_Scans
## Overview
This project implements an automated deep learning-based pipeline for detecting and localizing aortic stent endpoints in CT (Computed Tomography) scans. The system uses nnU-Net segmentation (via TotalSegmentator) combined with 3D cylindrical consistency tracking to identify stent boundaries and generate precise world-coordinate measurements.

## Key Features
Deep Learning Segmentation: Integrates TotalSegmentator (nnU-Net) for robust aorta segmentation with threshold-based fallback
Ring Detection: Multi-modal stent detection combining contour fitting (ellipse) and radial profile analysis
3D Consistency Tracking: Enforces cylindrical geometry constraints to validate true stent regions across consecutive slices
Batch Processing: Parallel processing of multiple patient datasets with configurable worker threads
Excel Reporting: Automated generation of detailed results with world coordinates (mm) and validation metrics
Ground Truth Validation: Optional comparison against annotated endpoints with error metrics

## Pipeline Steps
DICOM Loading - Read CT volume with metadata
Aorta Segmentation - Deep learning (nnU-Net) or threshold-based segmentation
ROI Extraction - Per-slice aorta centroid and dimensions
Candidate Extraction - Adaptive high-HU stent material detection
Ring Detection - Contour + radial profile analysis per slice
3D Consistency - Cylindrical geometry validation across slices
Classification - Per-slice ring scoring and detection
Gap Bridging - Temporal smoothing and small gap interpolation
Endpoint Localization - Start/end/midpoint coordinates in world space
Excel Report - Detailed output with measurements and validation errors

##Usage

Command-line Options:

--data_root (required): Root directory containing patient DICOM folders
--output: Output Excel file path (default: stent_ring_results_v2.xlsx)
--workers: Number of parallel workers (default: CPU count)
--sequential: Run single-threaded (useful for debugging)
--annotations: CSV file with ground truth coordinates for validation
--use_builtin_annotations: Use built-in validation dataset

## Requirements
Python 3.7+
SimpleITK, pydicom, scipy, scikit-image, openpyxl, pandas
OpenCV (optional, for ellipse fitting)
TotalSegmentator (optional, for nnU-Net segmentation)
Output
Excel spreadsheet with columns:

## Case ID, CT Date
Start/End/Midpoint coordinates (X, Y, Z in mm)
Stent length, detected slices count, mean ring score
Segmentation method used, validation errors (if available)
Processing status (OK/WARNING/ERROR)
