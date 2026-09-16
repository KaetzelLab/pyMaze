#!/usr/bin/env python3
"""
Retracking Pipeline - PyQt6 High-Performance Version
Uses pipeline architecture with multiprocessing for maximum CPU/GPU utilization
"""

import os
import sys
import json
import glob
import math
import copy
import time
import queue
import atexit
import signal
import pickle
import hashlib
import platform
import threading
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import shared_memory
from functools import lru_cache

warnings.filterwarnings('ignore')

import numpy as np
import cv2

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QGroupBox, QLabel, QLineEdit, QPushButton,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget,
    QListWidgetItem, QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QScrollArea, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage, QFont

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# UI HELPERS
# =============================================================================

class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):  # pragma: no cover - GUI behavior
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):  # pragma: no cover - GUI behavior
        event.ignore()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Optimal thread/process counts
CPU_COUNT = os.cpu_count() or 4
IS_JETSON = 'aarch64' in platform.machine()
READER_BUFFER_SIZE = 64 if not IS_JETSON else 16
CORRECTION_BATCH_SIZE = 100  # Process 100 frames at a time for correction
WRITER_BUFFER_SIZE = 64
NUM_CORRECTION_WORKERS = max(2, CPU_COUNT - 2)
NUM_ANALYSIS_WORKERS = max(2, CPU_COUNT // 2)

cv2.setNumThreads(2 if IS_JETSON else min(4, CPU_COUNT // 2))
cv2.setUseOptimized(True)

# Try to import optional acceleration libraries
try:
    from numba import njit, prange
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False
    def njit(func=None, **kwargs):
        def wrapper(f): return f
        return wrapper if func is None else func
    prange = range

try:
    import yaml
    YAML_OK = True
except ImportError:
    YAML_OK = False

try:
    from shapely.geometry import Point, Polygon
    from shapely.prepared import prep as shapely_prep
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False

# Global stop flag
STOP_FLAG = mp.Event()

# =============================================================================
# NUMBA-ACCELERATED CORRECTION FUNCTIONS (Run in separate processes)
# =============================================================================

if NUMBA_OK:
    @njit(cache=True, fastmath=True, parallel=True)
    def batch_rolling_median_numba(x_arr: np.ndarray, y_arr: np.ndarray,
                                    window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Parallel rolling median for batch of trajectories"""
        n_frames, n_keypoints = x_arr.shape
        x_out = x_arr.copy()
        y_out = y_arr.copy()
        half_w = window // 2

        for k in prange(n_keypoints):
            for i in range(n_frames):
                start = max(0, i - half_w)
                end = min(n_frames, i + half_w + 1)

                # Collect valid values
                x_vals = []
                y_vals = []
                for j in range(start, end):
                    if not np.isnan(x_arr[j, k]):
                        x_vals.append(x_arr[j, k])
                    if not np.isnan(y_arr[j, k]):
                        y_vals.append(y_arr[j, k])

                if len(x_vals) > 0:
                    x_vals_arr = np.array(x_vals)
                    x_vals_arr.sort()
                    x_out[i, k] = x_vals_arr[len(x_vals_arr) // 2]
                if len(y_vals) > 0:
                    y_vals_arr = np.array(y_vals)
                    y_vals_arr.sort()
                    y_out[i, k] = y_vals_arr[len(y_vals_arr) // 2]

        return x_out, y_out

    @njit(cache=True, fastmath=True, parallel=True)
    def batch_speed_correction_numba(x_arr: np.ndarray, y_arr: np.ndarray,
                                      timestamps: np.ndarray, max_speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """Parallel speed-based outlier correction"""
        n_frames, n_keypoints = x_arr.shape
        x_out = x_arr.copy()
        y_out = y_arr.copy()

        for k in prange(n_keypoints):
            for i in range(1, n_frames):
                if np.isnan(x_arr[i, k]) or np.isnan(x_arr[i-1, k]):
                    continue

                dt = (timestamps[i] - timestamps[i-1]) / 1000.0
                if dt <= 0:
                    continue

                dx = x_arr[i, k] - x_arr[i-1, k]
                dy = y_arr[i, k] - y_arr[i-1, k]
                speed = np.sqrt(dx*dx + dy*dy) / dt

                if speed > max_speed:
                    x_out[i, k] = x_out[i-1, k]
                    y_out[i, k] = y_out[i-1, k]

        return x_out, y_out

    @njit(cache=True, fastmath=True)
    def batch_swap_detection_numba(x_arr: np.ndarray, y_arr: np.ndarray,
                                    head_idx: int, body_idx: int,
                                    threshold: float) -> np.ndarray:
        """Detect swaps in batch - returns boolean array of swap frames"""
        n_frames = x_arr.shape[0]
        swaps = np.zeros(n_frames, dtype=np.bool_)

        for i in range(1, n_frames):
            # Skip if any value is NaN
            if (np.isnan(x_arr[i, head_idx]) or np.isnan(x_arr[i, body_idx]) or
                np.isnan(x_arr[i-1, head_idx]) or np.isnan(x_arr[i-1, body_idx])):
                continue

            # Cost without swap
            cost_normal = (np.sqrt((x_arr[i, head_idx] - x_arr[i-1, head_idx])**2 +
                                   (y_arr[i, head_idx] - y_arr[i-1, head_idx])**2) +
                          np.sqrt((x_arr[i, body_idx] - x_arr[i-1, body_idx])**2 +
                                  (y_arr[i, body_idx] - y_arr[i-1, body_idx])**2))

            # Cost with swap
            cost_swap = (np.sqrt((x_arr[i, body_idx] - x_arr[i-1, head_idx])**2 +
                                 (y_arr[i, body_idx] - y_arr[i-1, head_idx])**2) +
                        np.sqrt((x_arr[i, head_idx] - x_arr[i-1, body_idx])**2 +
                                (y_arr[i, head_idx] - y_arr[i-1, body_idx])**2))

            if cost_normal > 0:
                improvement = (cost_normal - cost_swap) / cost_normal
                if improvement > threshold:
                    swaps[i] = True

        return swaps

    @njit(cache=True, fastmath=True, parallel=True)
    def batch_zone_assignment_numba(centers_x: np.ndarray, centers_y: np.ndarray,
                                     zone_bounds: np.ndarray, zone_polys: List) -> np.ndarray:
        """Fast batch zone assignment using bounding box pre-filtering"""
        n_frames = centers_x.shape[0]
        n_zones = zone_bounds.shape[0]
        assignments = np.full(n_frames, -1, dtype=np.int32)

        for i in prange(n_frames):
            x, y = centers_x[i], centers_y[i]
            if np.isnan(x) or np.isnan(y):
                continue

            for z in range(n_zones):
                # Bounding box check first (fast)
                if (zone_bounds[z, 0] <= x <= zone_bounds[z, 2] and
                    zone_bounds[z, 1] <= y <= zone_bounds[z, 3]):
                    assignments[i] = z
                    break  # Take first matching zone

        return assignments

else:
    # Fallback without numba
    def batch_rolling_median_numba(x_arr, y_arr, window=5):
        from scipy.ndimage import median_filter
        x_out = np.apply_along_axis(lambda m: median_filter(m, size=window, mode='nearest'), 0, x_arr)
        y_out = np.apply_along_axis(lambda m: median_filter(m, size=window, mode='nearest'), 0, y_arr)
        return x_out, y_out

    def batch_speed_correction_numba(x_arr, y_arr, timestamps, max_speed):
        x_out, y_out = x_arr.copy(), y_arr.copy()
        for k in range(x_arr.shape[1]):
            for i in range(1, x_arr.shape[0]):
                if np.isnan(x_arr[i, k]) or np.isnan(x_arr[i-1, k]):
                    continue
                dt = (timestamps[i] - timestamps[i-1]) / 1000.0
                if dt <= 0:
                    continue
                speed = np.sqrt((x_arr[i,k]-x_arr[i-1,k])**2 + (y_arr[i,k]-y_arr[i-1,k])**2) / dt
                if speed > max_speed:
                    x_out[i, k], y_out[i, k] = x_out[i-1, k], y_out[i-1, k]
        return x_out, y_out

    def batch_swap_detection_numba(x_arr, y_arr, head_idx, body_idx, threshold):
        n_frames = x_arr.shape[0]
        swaps = np.zeros(n_frames, dtype=bool)
        for i in range(1, n_frames):
            if np.any(np.isnan([x_arr[i, head_idx], x_arr[i, body_idx],
                                x_arr[i-1, head_idx], x_arr[i-1, body_idx]])):
                continue
            cost_normal = (np.hypot(x_arr[i, head_idx] - x_arr[i-1, head_idx],
                                    y_arr[i, head_idx] - y_arr[i-1, head_idx]) +
                          np.hypot(x_arr[i, body_idx] - x_arr[i-1, body_idx],
                                   y_arr[i, body_idx] - y_arr[i-1, body_idx]))
            cost_swap = (np.hypot(x_arr[i, body_idx] - x_arr[i-1, head_idx],
                                  y_arr[i, body_idx] - y_arr[i-1, head_idx]) +
                        np.hypot(x_arr[i, head_idx] - x_arr[i-1, body_idx],
                                 y_arr[i, head_idx] - y_arr[i-1, body_idx]))
            if cost_normal > 0 and (cost_normal - cost_swap) / cost_normal > threshold:
                swaps[i] = True
        return swaps


# =============================================================================
# MULTIPROCESSING WORKER FUNCTIONS (Top-level for pickling)
# =============================================================================

def correction_worker(batch_data: Dict) -> Dict:
    """
    Worker function for batch keypoint correction.
    Runs in separate process for true parallelism.
    """
    try:
        x_arr = batch_data['x']  # Shape: (batch_size, n_keypoints)
        y_arr = batch_data['y']
        conf_arr = batch_data['conf']
        timestamps = batch_data['timestamps']
        config = batch_data['config']
        batch_idx = batch_data['batch_idx']
        head_idx = batch_data.get('head_idx', 0)
        body_idx = batch_data.get('body_idx', 1)

        # Apply corrections in sequence

        # 1. Swap resolution
        if config.get('enable_swap_resolver', False):
            swaps = batch_swap_detection_numba(
                x_arr, y_arr, head_idx, body_idx,
                config.get('swap_cost_threshold', 0.3)
            )
            # Apply swaps
            for i in np.where(swaps)[0]:
                x_arr[i, head_idx], x_arr[i, body_idx] = x_arr[i, body_idx], x_arr[i, head_idx]
                y_arr[i, head_idx], y_arr[i, body_idx] = y_arr[i, body_idx], y_arr[i, head_idx]
                conf_arr[i, head_idx], conf_arr[i, body_idx] = conf_arr[i, body_idx], conf_arr[i, head_idx]

        # 2. Speed-based correction
        if config.get('max_speed_px_sec', 0) > 0:
            x_arr, y_arr = batch_speed_correction_numba(
                x_arr, y_arr, timestamps,
                config.get('max_speed_px_sec', 3000)
            )

        # 3. Rolling median
        if config.get('use_rolling_median', False):
            x_arr, y_arr = batch_rolling_median_numba(x_arr, y_arr, window=5)

        # 4. One-Euro filter
        if config.get('use_one_euro', False):
            x_arr, y_arr = apply_one_euro_batch(
                x_arr, y_arr, timestamps,
                config.get('euro_min_cutoff', 1.0),
                config.get('euro_beta', 0.007)
            )

        return {
            'batch_idx': batch_idx,
            'x': x_arr,
            'y': y_arr,
            'conf': conf_arr,
            'success': True
        }
    except Exception as e:
        return {
            'batch_idx': batch_data.get('batch_idx', -1),
            'error': str(e),
            'success': False
        }


def zone_assignment_worker(batch_data: Dict) -> Dict:
    """
    Worker function for batch zone assignment.
    """
    try:
        centers = batch_data['centers']  # List of (x, y) or None
        zones_data = batch_data['zones_data']
        batch_idx = batch_data['batch_idx']

        # Build zone polygons for OpenCV
        zone_names = list(zones_data.keys())
        zone_polys = []
        zone_bounds = []

        for name in zone_names:
            pts = zones_data[name].get('values', [])
            if len(pts) >= 3:
                arr = np.array(pts, dtype=np.float32)
                zone_polys.append(arr)
                zone_bounds.append([
                    float(np.min(arr[:, 0])), float(np.min(arr[:, 1])),
                    float(np.max(arr[:, 0])), float(np.max(arr[:, 1]))
                ])
            else:
                zone_polys.append(None)
                zone_bounds.append([0, 0, 0, 0])

        zone_bounds = np.array(zone_bounds, dtype=np.float32)

        # Assign zones
        assignments = []
        for center in centers:
            if center is None:
                assignments.append(None)
                continue

            x, y = center
            if np.isnan(x) or np.isnan(y):
                assignments.append(None)
                continue

            found = None
            for z_idx, (name, poly) in enumerate(zip(zone_names, zone_polys)):
                if poly is None:
                    continue
                # Bounding box check
                if not (zone_bounds[z_idx, 0] <= x <= zone_bounds[z_idx, 2] and
                        zone_bounds[z_idx, 1] <= y <= zone_bounds[z_idx, 3]):
                    continue
                # Point in polygon
                if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                    found = name
                    break

            assignments.append(found)

        return {
            'batch_idx': batch_idx,
            'assignments': assignments,
            'success': True
        }
    except Exception as e:
        return {
            'batch_idx': batch_data.get('batch_idx', -1),
            'error': str(e),
            'success': False
        }


def statistics_worker(data: Dict) -> Dict:
    """Worker for parallel statistics calculation"""
    try:
        import pandas as pd

        times = data['times']
        zones = data['zones']
        states = data['states']
        centers = data['centers']
        heads = data['heads']
        config = data['config']
        zones_full = data['zones_full']
        stem = data['stem']
        output_dir = data['output_dir']

        result = calculate_zone_statistics_full(
            stem, times, zones, states, output_dir, zones_full,
            centers, heads, config.get('min_dwell_time', 0.2),
            file_meta=data.get('file_meta'),
            angle_threshold=config.get('angle_threshold', 45.0),
            ia_confirm_ms=config.get('ia_confirm_ms', 100.0),
            ia_line_extend_px=config.get('ia_line_extend_px', 15.0),
            ia_inner_offset_px=config.get('ia_inner_offset_px', 50.0),
            ia_parallel_tol_deg=config.get('ia_parallel_tol_deg', 15.0),
            params=config
        )

        return {'success': True, 'stem': stem, 'excel_path': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}


# =============================================================================
# ONE-EURO FILTER IMPLEMENTATION
# =============================================================================

class OneEuroFilter:
    """
    One-Euro filter for position smoothing with minimal lag.
    Adapts cutoff frequency based on speed to reduce jitter while preserving quick movements.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def _exp_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev

    def filter(self, x: float, t: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        # Derivative
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exp_smoothing(a_d, dx, self.dx_prev)

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Position
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exp_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class OneEuroFilter2D:
    """2D One-Euro filter for (x, y) positions"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def filter(self, x: float, y: float, t: float) -> Tuple[float, float]:
        if np.isnan(x) or np.isnan(y):
            return x, y
        return self.filter_x.filter(x, t), self.filter_y.filter(y, t)

    def reset(self):
        self.filter_x.reset()
        self.filter_y.reset()


def apply_one_euro_batch(x_arr: np.ndarray, y_arr: np.ndarray, timestamps: np.ndarray,
                         min_cutoff: float = 1.0, beta: float = 0.007) -> Tuple[np.ndarray, np.ndarray]:
    """Apply One-Euro filter to batch of keypoint trajectories"""
    n_frames, n_keypoints = x_arr.shape
    x_out = x_arr.copy()
    y_out = y_arr.copy()

    for k in range(n_keypoints):
        filt = OneEuroFilter2D(min_cutoff, beta)
        for i in range(n_frames):
            if not np.isnan(x_arr[i, k]) and not np.isnan(y_arr[i, k]):
                t = timestamps[i] / 1000.0  # Convert to seconds
                x_out[i, k], y_out[i, k] = filt.filter(x_arr[i, k], y_arr[i, k], t)

    return x_out, y_out


# =============================================================================
# REGIONS HIERARCHY - Maps main regions to sub-zones for Excel aggregation
# =============================================================================

REGIONS_HIERARCHY = {
    'CenterZone': [
        'Social_Buffer', 'Social_Open_Corner', 'Familiar_Social_Corner', 'Open_Buffer',
        'Open_Novel_Corner', 'Novel_Buffer', 'Novel_Object1_Corner', 'Object1_Buffer',
        'Object1_Object2_Corner', 'Object2_Buffer', 'Object2_Familiar_Corner', 'Familiar_Buffer'
    ],
    'Center': ['CenterZone'],
    'Social_IA': ['Social_IA'],
    'Social_Entry': ['Social_Entry'],
    'OpenEntry': ['OpenEntry'],
    'OpenArm': ['OpenArm'],
    'NovelArm': ['NovelArm'],
    'Object1_IA': ['Object1_IA'],
    'Object1_Entry': ['Object1_Entry'],
    'Object2_IA': ['Object2_IA'],
    'Object2_Entry': ['Object2_Entry'],
    'FamiliarArm': ['FamiliarArm']
}


# =============================================================================
# METADATA PARSING FROM FILENAME
# =============================================================================

def parse_metadata_from_filename(stem: str) -> Dict[str, str]:
    """
    Parse metadata from filename.
    Expected format variations:
    - GROUP_MOUSEID_DATE_TIME or similar patterns
    - Returns dict with: group, mouseID, expt_date, time
    """
    import re

    result = {
        'group': '',
        'mouseID': '',
        'expt_date': '',
        'time': ''
    }

    # Try to parse common patterns
    parts = [p for p in stem.replace('-', '_').split('_') if p]

    # Look for date patterns (YYYYMMDD, YYYY-MM-DD, DD-MM-YYYY, etc.)
    date_pattern = re.compile(r'(\d{4}[-_]?\d{2}[-_]?\d{2})|(\d{2}[-_]?\d{2}[-_]?\d{4})')
    time_pattern = re.compile(r'(\d{2}[-_:]?\d{2}[-_:]?\d{2})')
    mouse_pattern = re.compile(r'(mouse|m|M)[-_]?(\d+)', re.IGNORECASE)

    # Prefer explicit mouse-like IDs (e.g., D853, M12)
    simple_mouse_pattern = re.compile(r'^[A-Za-z]*\d+$')

    for i, part in enumerate(parts):
        # Check for date
        if date_pattern.match(part):
            result['expt_date'] = part
        # Check for time (HH:MM:SS or HHMMSS)
        elif time_pattern.match(part) and len(part) >= 6 and not date_pattern.match(part):
            result['time'] = part
        # Check for mouse ID
        elif mouse_pattern.match(part):
            result['mouseID'] = part
        elif simple_mouse_pattern.match(part) and not result['mouseID']:
            result['mouseID'] = part

    # Detect YYYY_MM_DD pattern split across parts
    for i in range(len(parts) - 2):
        if (re.fullmatch(r'\d{4}', parts[i]) and
            re.fullmatch(r'\d{2}', parts[i + 1]) and
            re.fullmatch(r'\d{2}', parts[i + 2])):
            result['expt_date'] = f"{parts[i]}-{parts[i + 1]}-{parts[i + 2]}"
            break

    # If no structured parsing, use positional
    if not result['group'] and len(parts) >= 1:
        ignore = {'video', 'data', 'videodata'}
        for part in parts:
            if part == result['mouseID']:
                continue
            if date_pattern.match(part) or time_pattern.match(part):
                continue
            if part.lower() in ignore:
                continue
            result['group'] = part
            break
    if not result['mouseID'] and len(parts) >= 2:
        result['mouseID'] = parts[1]
    if not result['expt_date'] and len(parts) >= 3:
        result['expt_date'] = parts[2]
    if not result['time'] and len(parts) >= 4:
        result['time'] = parts[3]

    return result


def parse_metadata_from_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Parse metadata from B-line meta dict."""
    import re

    result = {
        'group': '',
        'mouseID': '',
        'expt_date': '',
        'time': '',
        'start_time': '',
        'subgroup': ''
    }
    if not meta:
        return result

    def pick(keys):
        for k in keys:
            v = meta.get(k)
            if v:
                return str(v)
        return ''

    result['mouseID'] = pick(['Subject ID', 'SubjectID', 'Mouse ID', 'MouseID', 'ID'])
    result['group'] = pick(['Group'])
    result['subgroup'] = pick(['Sub Group', 'SubGroup'])

    start_time = pick(['Start time', 'Start_time', 'StartTime', 'DateTime', 'Date'])
    result['start_time'] = start_time
    if start_time:
        date_match = re.search(r'(\d{4})[/-](\d{2})[/-](\d{2})', start_time)
        time_match = re.search(r'(\d{2})[:\-](\d{2})[:\-](\d{2})', start_time)
        if date_match:
            result['expt_date'] = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        if time_match:
            result['time'] = f"{time_match.group(1)}:{time_match.group(2)}:{time_match.group(3)}"

    return result


def merge_metadata(primary: Dict[str, str], fallback: Dict[str, str]) -> Dict[str, str]:
    """Prefer primary values, fill missing from fallback."""
    merged = dict(primary)
    for key in ('group', 'mouseID', 'expt_date', 'time', 'start_time', 'subgroup'):
        if not merged.get(key):
            merged[key] = fallback.get(key, '')
    return merged


def _pix_per_cm_from_scale_info(scale_info: Any) -> Optional[float]:
    if scale_info is None:
        return None
    if isinstance(scale_info, (int, float)):
        return float(scale_info)
    if isinstance(scale_info, dict):
        for key in ('pix_per_cm', 'pixels_per_cm', 'px_per_cm', 'pixpercm'):
            if key in scale_info:
                try:
                    return float(scale_info[key])
                except Exception:
                    pass
        if 'length' in scale_info and 'values' in scale_info:
            try:
                length_cm = float(scale_info.get('length', 0))
                vals = scale_info.get('values')
                if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                    if isinstance(vals[0], (list, tuple)) and len(vals[0]) >= 2:
                        x1, y1 = float(vals[0][0]), float(vals[0][1])
                        x2, y2 = float(vals[1][0]), float(vals[1][1])
                    elif len(vals) >= 4:
                        x1, y1, x2, y2 = map(float, vals[:4])
                    elif len(vals) == 2:
                        x1, x2 = map(float, vals[:2])
                        y1 = y2 = 0.0
                    else:
                        return None
                    pix_dist = math.hypot(x2 - x1, y2 - y1)
                    if pix_dist > 0 and length_cm > 0:
                        return pix_dist / length_cm
            except Exception:
                return None
    if isinstance(scale_info, (list, tuple)) and len(scale_info) >= 4:
        try:
            x1, y1, x2, y2 = map(float, scale_info[:4])
            length_cm = 22.0
            pix_dist = math.hypot(x2 - x1, y2 - y1)
            if pix_dist > 0:
                return pix_dist / length_cm
        except Exception:
            return None
    return None


def extract_scale_m_per_px(zones_full: Optional[Dict], file_meta: Optional[Dict[str, Any]]) -> float:
    """Return meters-per-pixel based on scale info when available."""
    pix_per_cm = None
    if zones_full:
        try:
            scale_info = zones_full.get('scale') or zones_full.get('Scale')
            pix_per_cm = _pix_per_cm_from_scale_info(scale_info)
        except Exception:
            pix_per_cm = None
    if pix_per_cm is None and file_meta:
        for key in ('scale', 'Scale', 'pix_per_cm', 'pixels_per_cm', 'px_per_cm'):
            if key in file_meta:
                pix_per_cm = _pix_per_cm_from_scale_info(file_meta.get(key))
                if pix_per_cm:
                    break
    if pix_per_cm and pix_per_cm > 0:
        return 0.01 / float(pix_per_cm)
    return 1.0

# =============================================================================
# IA ZONE GENERATION & ANGLE CALCULATION
# =============================================================================

def get_zone_center(zone_pts: List) -> Tuple[float, float]:
    """Get centroid of a zone polygon"""
    if not zone_pts or len(zone_pts) < 3:
        return (0.0, 0.0)
    pts = np.array(zone_pts, dtype=np.float32)
    return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))


def get_arena_center(zroot: Dict) -> Tuple[float, float]:
    """Estimate arena center as mean of zone centroids."""
    if not zroot:
        return (0.0, 0.0)
    centers = []
    for z in zroot.values():
        pts = z.get('values', [])
        if len(pts) >= 3:
            centers.append(get_zone_center(pts))
    if not centers:
        return (0.0, 0.0)
    arr = np.array(centers, dtype=np.float32)
    return (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))


def get_outermost_edge_line(zone_pts: List,
                            reference_center: Tuple[float, float],
                            parallel_tol_deg: float = 15.0) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Get the outermost polygon edge line relative to a reference center.
    Prefer the longest edge from a parallel-edge pair (inner/outer), then choose
    the edge whose midpoint is farthest from the reference center.
    """
    if not zone_pts or len(zone_pts) < 2:
        return None

    pts = np.array(zone_pts, dtype=np.float32)
    if len(pts) < 2:
        return None

    rx, ry = reference_center
    edges = []
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        angle = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
        mx = 0.5 * (p1[0] + p2[0])
        my = 0.5 * (p1[1] + p2[1])
        dist = float((mx - rx) ** 2 + (my - ry) ** 2)
        edges.append({
            'p1': (float(p1[0]), float(p1[1])),
            'p2': (float(p2[0]), float(p2[1])),
            'angle': angle,
            'length': length,
            'dist': dist
        })

    if not edges:
        return None

    # Group edges by parallel orientation (angle modulo 180)
    groups = []
    for e in edges:
        placed = False
        for g in groups:
            if abs(e['angle'] - g['angle']) <= parallel_tol_deg:
                g['edges'].append(e)
                g['max_len'] = max(g['max_len'], e['length'])
                placed = True
                break
        if not placed:
            groups.append({'angle': e['angle'], 'edges': [e], 'max_len': e['length']})

    # Prefer group with the longest edge (rectangle outer edge case)
    groups.sort(key=lambda g: g['max_len'], reverse=True)
    best_group = groups[0]

    # In that group, pick the edge farthest from arena center
    best_edge = max(best_group['edges'], key=lambda e: e['dist'])
    return best_edge['p1'], best_edge['p2']


def extend_line(p1: Tuple[float, float], p2: Tuple[float, float], extend_px: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Extend a line segment by extend_px on both ends."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return p1, p2
    ux = dx / length
    uy = dy / length
    return (x1 - ux * extend_px, y1 - uy * extend_px), (x2 + ux * extend_px, y2 + uy * extend_px)


def build_ia_reference_lines(zone_pts: List,
                             arena_center: Tuple[float, float],
                             line_extend_px: float = 15.0,
                             inner_offset_px: float = 50.0,
                             parallel_tol_deg: float = 15.0
                             ) -> Tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
                                        Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
                                        Optional[Tuple[float, float]]]:
    """
    Return (outer_line, inner_line, normal_unit_toward_center).
    The outer line is extended by line_extend_px; inner line is parallel and
    shifted toward arena center by inner_offset_px.
    """
    edge = get_outermost_edge_line(zone_pts, arena_center, parallel_tol_deg=parallel_tol_deg)
    if not edge:
        return None, None, None

    outer_line = extend_line(edge[0], edge[1], line_extend_px)
    (x1, y1), (x2, y2) = outer_line
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return outer_line, None, None

    # Normal direction (perpendicular) - choose the side pointing to arena center
    nx = -dy / length
    ny = dx / length
    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)
    vx = arena_center[0] - mx
    vy = arena_center[1] - my
    if nx * vx + ny * vy < 0:
        nx *= -1.0
        ny *= -1.0

    inner_line = ((x1 + nx * inner_offset_px, y1 + ny * inner_offset_px),
                  (x2 + nx * inner_offset_px, y2 + ny * inner_offset_px))
    return outer_line, inner_line, (nx, ny)


def get_zone_ia_inner_offset(zone_data: Dict, default_px: float) -> float:
    """Return per-zone IA inner offset (px), falling back to default."""
    try:
        return float(zone_data.get('ia_inner_offset_px', default_px))
    except Exception:
        return float(default_px)


def get_zone_ia_line_extend(zone_data: Dict, default_px: float) -> float:
    """Return per-zone IA line extend (px), falling back to default."""
    try:
        return float(zone_data.get('ia_line_extend_px', default_px))
    except Exception:
        return float(default_px)


def point_is_past_line(point: Tuple[float, float],
                       line_p1: Tuple[float, float],
                       normal_unit: Tuple[float, float],
                       toward_center: bool = True) -> bool:
    """Check which side of the line the point is on (relative to normal)."""
    px, py = point
    x1, y1 = line_p1
    nx, ny = normal_unit
    side = (px - x1) * nx + (py - y1) * ny
    return side >= 0.0 if toward_center else side <= 0.0


def calculate_perpendicular_facing(head_xy: Tuple[float, float], body_xy: Tuple[float, float],
                                   edge_p1: Tuple[float, float], edge_p2: Tuple[float, float],
                                   angle_tolerance_deg: float = 30.0) -> bool:
    """Check if head-body line is near perpendicular to an edge line."""
    if head_xy is None or body_xy is None or edge_p1 is None or edge_p2 is None:
        return False

    hx, hy = head_xy
    bx, by = body_xy
    x1, y1 = edge_p1
    x2, y2 = edge_p2

    if not all(np.isfinite([hx, hy, bx, by, x1, y1, x2, y2])):
        return False

    # Heading vector
    vx = hx - bx
    vy = hy - by
    vmag = np.hypot(vx, vy)
    if vmag < 1e-6:
        return False

    # Edge direction vector
    ex = x2 - x1
    ey = y2 - y1
    emag = np.hypot(ex, ey)
    if emag < 1e-6:
        return False

    vx /= vmag
    vy /= vmag
    ex /= emag
    ey /= emag

    dot = np.clip(vx * ex + vy * ey, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(dot)))

    return abs(angle - 90.0) <= angle_tolerance_deg


def compute_facing_angle_to_point(head_xy: Tuple[float, float], body_xy: Tuple[float, float],
                                  target_xy: Tuple[float, float]) -> Optional[float]:
    """Return angle (deg) between head-body vector and head->target vector."""
    if head_xy is None or body_xy is None or target_xy is None:
        return None
    hx, hy = head_xy
    bx, by = body_xy
    tx, ty = target_xy
    if not all(np.isfinite([hx, hy, bx, by, tx, ty])):
        return None

    vx = hx - bx
    vy = hy - by
    vmag = np.hypot(vx, vy)
    if vmag < 1e-6:
        return None
    vx /= vmag
    vy /= vmag

    ux = tx - hx
    uy = ty - hy
    umag = np.hypot(ux, uy)
    if umag < 1e-6:
        return None
    ux /= umag
    uy /= umag

    dot = np.clip(vx * ux + vy * uy, -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def closest_point_on_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float]]:
    """Closest point on segment to point P."""
    ex = x2 - x1
    ey = y2 - y1
    denom = ex * ex + ey * ey
    if denom < 1e-6:
        return None
    t = ((px - x1) * ex + (py - y1) * ey) / denom
    t = max(0.0, min(1.0, t))
    return (x1 + t * ex, y1 + t * ey)


def compute_facing_angle_to_edge(head_xy: Tuple[float, float], body_xy: Tuple[float, float],
                                 edge_p1: Tuple[float, float], edge_p2: Tuple[float, float]) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Return angle (deg) between head-body vector and head->closest-edge vector."""
    if head_xy is None or body_xy is None or edge_p1 is None or edge_p2 is None:
        return None, None
    hx, hy = head_xy
    if not all(np.isfinite([hx, hy, edge_p1[0], edge_p1[1], edge_p2[0], edge_p2[1]])):
        return None, None

    cp = closest_point_on_segment(hx, hy, edge_p1[0], edge_p1[1], edge_p2[0], edge_p2[1])
    if cp is None:
        return None, None

    angle = compute_facing_angle_to_point(head_xy, body_xy, cp)
    return angle, cp
def calculate_angle_time_facing(head_xy: Tuple[float, float], body_xy: Tuple[float, float],
                                 zone_centroid: Tuple[float, float], angle_tolerance_deg: float = 30.0) -> bool:
    """
    Legacy facing check (centroid-based). Prefer calculate_perpendicular_facing for _IA rules.
    """
    if head_xy is None or body_xy is None or zone_centroid is None:
        return False

    hx, hy = head_xy
    bx, by = body_xy
    zx, zy = zone_centroid

    if not all(np.isfinite([hx, hy, bx, by, zx, zy])):
        return False

    dx_head = hx - bx
    dy_head = hy - by
    norm_head = np.hypot(dx_head, dy_head)

    if norm_head < 1e-6:
        return False

    dx_head /= norm_head
    dy_head /= norm_head

    dx_zone = zx - bx
    dy_zone = zy - by
    norm_zone = np.hypot(dx_zone, dy_zone)

    if norm_zone < 1e-6:
        return False

    dx_zone /= norm_zone
    dy_zone /= norm_zone

    dot = dx_head * dx_zone + dy_head * dy_zone
    threshold = np.cos(np.radians(angle_tolerance_deg))

    return dot > threshold


def get_target_zone_name(ia_zone_name: str) -> str:
    """Get the target zone name for an IA zone (remove the _IA token if present)."""
    if '_IA' in ia_zone_name:
        return ia_zone_name.replace('_IA', '')
    return ia_zone_name


def _zone_name_tokens(name: str) -> List[str]:
    parts = [p for p in name.replace('-', '_').split('_') if p]
    return [p.lower() for p in parts if p.upper() != 'IA']


def resolve_ia_target_centroid(zroot: Dict, ia_zone_name: str) -> Optional[Tuple[float, float]]:
    """Resolve the best target centroid for an IA zone using name or proximity heuristics."""
    if not zroot or ia_zone_name not in zroot:
        return None

    ia_pts = zroot.get(ia_zone_name, {}).get('values', [])
    if len(ia_pts) < 3:
        return None
    ia_centroid = get_zone_center(ia_pts)

    target_name = get_target_zone_name(ia_zone_name)
    if target_name in zroot:
        return get_zone_center(zroot[target_name].get('values', []))

    ia_tokens = _zone_name_tokens(ia_zone_name)
    candidates = []
    for zn, zdata in zroot.items():
        if '_IA' in zn or zn == ia_zone_name:
            continue
        zn_tokens = _zone_name_tokens(zn)
        if all(t in zn_tokens for t in ia_tokens):
            candidates.append(zn)

    non_ia = [zn for zn in zroot.keys() if '_IA' not in zn and zn != ia_zone_name]
    if not candidates and not non_ia:
        return ia_centroid

    def dist_to_ia(zn: str) -> float:
        pts = zroot.get(zn, {}).get('values', [])
        if len(pts) < 3:
            return float('inf')
        c = get_zone_center(pts)
        dx = c[0] - ia_centroid[0]
        dy = c[1] - ia_centroid[1]
        return dx * dx + dy * dy

    if candidates:
        best = min(candidates, key=dist_to_ia)
        return get_zone_center(zroot[best].get('values', []))

    best = min(non_ia, key=dist_to_ia)
    return get_zone_center(zroot[best].get('values', []))


def count_zone_entries_with_dwell(in_zone_array: np.ndarray, time_intervals: np.ndarray,
                                   min_dwell_time: float = 0.2) -> int:
    """Count zone entries that meet minimum dwell time requirement."""
    if len(in_zone_array) < 2:
        return 0

    entries = 0
    in_zone = False
    dwell_time = 0.0
    entry_valid = False

    for i in range(len(in_zone_array)):
        if in_zone_array[i]:
            if not in_zone:
                # Entering zone
                in_zone = True
                dwell_time = time_intervals[i] if i < len(time_intervals) else 0.0
                entry_valid = False
            else:
                # Still in zone
                dwell_time += time_intervals[i] if i < len(time_intervals) else 0.0
                if dwell_time >= min_dwell_time and not entry_valid:
                    entries += 1
                    entry_valid = True
        else:
            # Exited zone
            in_zone = False
            dwell_time = 0.0
            entry_valid = False

    return entries


def get_ia_zones(zones_full: Dict) -> List[str]:
    """Get list of IA zones (zones containing '_IA') for angle time calculation"""
    if not zones_full:
        return []
    zroot = zones_full.get('zones', zones_full)
    return [name for name in zroot.keys() if '_IA' in name]


def generate_imaginary_ia_zones(zones_full: Dict, crop: 'CropBox', ia_zone_depth: int = 70) -> Dict:
    """
    Generate imaginary IA (Interaction Area) zones as full-width strips at frame edges.

    Each IA zone gets a corresponding strip at the edge it's closest to:
    - RIGHT edge: full-height strip on right side
    - LEFT edge: full-height strip on left side
    - TOP edge: full-width strip on top
    - BOTTOM edge: full-width strip on bottom

    The strips extend to BOTH edges (full width or full height) with user-defined depth.

    Args:
        zones_full: Zone dictionary with zone data
        crop: CropBox defining frame boundaries
        ia_zone_depth: Depth of the edge strip in pixels (default 70)

    Returns:
        Updated zones dictionary with imaginary IA zones added
    """
    if not zones_full or ia_zone_depth <= 0:
        return zones_full

    updated = copy.deepcopy(zones_full)
    zroot = zones_root(updated)

    # Frame boundaries from crop box
    x1, y1, x2, y2 = crop.x1, crop.y1, crop.x2, crop.y2
    frame_w = x2 - x1
    frame_h = y2 - y1

    # Center of frame
    frame_cx = (x1 + x2) / 2.0
    frame_cy = (y1 + y2) / 2.0

    # Find all IA zones (excluding any already generated imaginary ones)
    ia_zones = [name for name in zroot.keys()
                if '_IA' in name and '_Imaginary' not in name and '_Touch' not in name]

    for ia_zone_name in ia_zones:
        zone_data = zroot.get(ia_zone_name, {})
        pts = zone_data.get('values', [])

        if len(pts) < 3:
            continue

        # Calculate centroid of the IA zone
        pts_array = np.array(pts, dtype=np.float32)
        centroid_x = float(np.mean(pts_array[:, 0]))
        centroid_y = float(np.mean(pts_array[:, 1]))

        # Determine which edge the zone is closest to
        # Calculate distances to each edge
        dist_left = centroid_x - x1
        dist_right = x2 - centroid_x
        dist_top = centroid_y - y1
        dist_bottom = y2 - centroid_y

        # Find minimum distance to determine closest edge
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        # Create full-width/height strip at the closest edge
        if min_dist == dist_right:
            # RIGHT edge: vertical strip on right side
            strip_pts = [
                [x2 - ia_zone_depth, y1],  # top-left
                [x2, y1],                   # top-right
                [x2, y2],                   # bottom-right
                [x2 - ia_zone_depth, y2],  # bottom-left
            ]
            edge_name = "Right"
        elif min_dist == dist_left:
            # LEFT edge: vertical strip on left side
            strip_pts = [
                [x1, y1],                   # top-left
                [x1 + ia_zone_depth, y1],  # top-right
                [x1 + ia_zone_depth, y2],  # bottom-right
                [x1, y2],                   # bottom-left
            ]
            edge_name = "Left"
        elif min_dist == dist_top:
            # TOP edge: horizontal strip on top
            strip_pts = [
                [x1, y1],                   # top-left
                [x2, y1],                   # top-right
                [x2, y1 + ia_zone_depth],  # bottom-right
                [x1, y1 + ia_zone_depth],  # bottom-left
            ]
            edge_name = "Top"
        else:
            # BOTTOM edge: horizontal strip on bottom
            strip_pts = [
                [x1, y2 - ia_zone_depth],  # top-left
                [x2, y2 - ia_zone_depth],  # top-right
                [x2, y2],                   # bottom-right
                [x1, y2],                   # bottom-left
            ]
            edge_name = "Bottom"

        # Create the imaginary IA zone
        imaginary_zone_name = ia_zone_name.replace('_IA', f'_IA_Imaginary_{edge_name}')
        zroot[imaginary_zone_name] = {
            'values': strip_pts,
            'color': zone_data.get('color', [255, 200, 0]),
            'type': 'imaginary_ia',
            'source_zone': ia_zone_name,
            'edge': edge_name.lower(),
            'depth': ia_zone_depth
        }

    return updated


def calculate_facing_angle(head_x: float, head_y: float, body_x: float, body_y: float,
                           target_x: float, target_y: float) -> float:
    """
    Calculate angle between head-body vector and direction to target.
    Returns angle in degrees (0 = directly facing, 180 = facing away).
    """
    hb_x = head_x - body_x
    hb_y = head_y - body_y
    hb_mag = np.hypot(hb_x, hb_y)

    bt_x = target_x - body_x
    bt_y = target_y - body_y
    bt_mag = np.hypot(bt_x, bt_y)

    if hb_mag < 1e-6 or bt_mag < 1e-6:
        return 180.0

    hb_x /= hb_mag
    hb_y /= hb_mag
    bt_x /= bt_mag
    bt_y /= bt_mag

    dot = hb_x * bt_x + hb_y * bt_y
    dot = np.clip(dot, -1.0, 1.0)

    return float(np.degrees(np.arccos(dot)))


if NUMBA_OK:
    @njit(cache=True, fastmath=True, parallel=True)
    def batch_facing_angles_numba(head_x: np.ndarray, head_y: np.ndarray,
                                   body_x: np.ndarray, body_y: np.ndarray,
                                   target_x: float, target_y: float) -> np.ndarray:
        """Calculate facing angles for batch of frames in parallel"""
        n = len(head_x)
        angles = np.full(n, 180.0, dtype=np.float64)

        for i in prange(n):
            if np.isnan(head_x[i]) or np.isnan(head_y[i]) or np.isnan(body_x[i]) or np.isnan(body_y[i]):
                continue

            hb_x = head_x[i] - body_x[i]
            hb_y = head_y[i] - body_y[i]
            hb_mag = np.sqrt(hb_x*hb_x + hb_y*hb_y)

            bt_x = target_x - body_x[i]
            bt_y = target_y - body_y[i]
            bt_mag = np.sqrt(bt_x*bt_x + bt_y*bt_y)

            if hb_mag < 1e-6 or bt_mag < 1e-6:
                continue

            hb_x /= hb_mag
            hb_y /= hb_mag
            bt_x /= bt_mag
            bt_y /= bt_mag

            dot = hb_x * bt_x + hb_y * bt_y
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0

            angles[i] = np.degrees(np.arccos(dot))

        return angles
else:
    def batch_facing_angles_numba(head_x, head_y, body_x, body_y, target_x, target_y):
        n = len(head_x)
        angles = np.full(n, 180.0)
        for i in range(n):
            if np.isnan(head_x[i]) or np.isnan(body_x[i]):
                continue
            angles[i] = calculate_facing_angle(head_x[i], head_y[i], body_x[i], body_y[i], target_x, target_y)
        return angles


def order_zones_spatial(zones_full: Dict) -> List[str]:
    """Order zones spatially by angle from center"""
    zroot = zones_full.get('zones', zones_full)
    if not zroot:
        return []

    names, centroids = [], []
    for name, z in zroot.items():
        pts = z.get('values', [])
        if len(pts) >= 3:
            arr = np.array(pts, dtype=np.float32)
            names.append(name)
            centroids.append((float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))))

    if not names:
        return list(zroot.keys())

    centroids_arr = np.array(centroids, dtype=np.float32)
    center = np.mean(centroids_arr, axis=0)
    angles = np.arctan2(centroids_arr[:, 1] - center[1], centroids_arr[:, 0] - center[0])
    return [names[i] for i in np.argsort(angles)]


def calculate_zone_statistics_full(stem: str, times_buf: List[float],
                                    zone_buf: List[Optional[str]], state_buf: List[Optional[str]],
                                    output_dir: str, zones_full: Optional[Dict] = None,
                                    centers_buf: List = None, head_buf: List = None,
                                    min_dwell_time: float = 0.2,
                                    zone_assigner: 'FastZoneAssigner' = None,
                                    file_meta: Optional[Dict[str, Any]] = None,
                                    angle_threshold: float = 45.0,
                                    ia_confirm_ms: float = 100.0,
                                    ia_line_extend_px: float = 15.0,
                                    ia_inner_offset_px: float = 50.0,
                                    ia_parallel_tol_deg: float = 15.0,
                                    params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Calculate zone duration statistics with angle time for IA zones.
    Uses REGIONS_HIERARCHY for aggregation and dot product > 0.5 for facing check.
    Angle time = time IN zone AND facing zone centroid.
    Now includes: head entries/duration (simple zone entry like body), metadata columns.
    """
    try:
        import pandas as pd

        if not times_buf or len(times_buf) < 2:
            return None

        # Parse metadata from file meta (B-line) then fallback to filename
        metadata_from_meta = parse_metadata_from_meta(file_meta)
        metadata_from_name = parse_metadata_from_filename(stem)
        metadata = merge_metadata(metadata_from_meta, metadata_from_name)
        scale_m_per_px = extract_scale_m_per_px(zones_full, file_meta)

        n_frames = len(times_buf)
        zroot = zones_full.get('zones', zones_full) if zones_full else {}
        zone_names = list(zroot.keys()) if zroot else []

        # Build zone assigner if not provided (for head zone detection)
        if zone_assigner is None and zones_full:
            zone_assigner = FastZoneAssigner(zones_full)

        # Build frame data list with head zone
        frame_data = []
        for i in range(n_frames):
            head = head_buf[i] if head_buf and i < len(head_buf) else None
            # Determine which zone the HEAD is in (simple point-in-polygon)
            head_zone = None
            if head and zone_assigner:
                hx, hy = head[0], head[1]
                if np.isfinite(hx) and np.isfinite(hy):
                    head_zone = zone_assigner.locate(hx, hy)

            fd = {
                'timestamp': times_buf[i],
                'zone': zone_buf[i] if i < len(zone_buf) else None,
                'head_zone': head_zone,  # NEW: zone where head is located
                'state': state_buf[i] if i < len(state_buf) else None,
                'center': centers_buf[i] if centers_buf and i < len(centers_buf) else None,
                'head': head,
            }
            frame_data.append(fd)

        # Calculate time intervals (dt) in seconds from timestamps.
        # Keep input order (frame order) and unwrap timestamps on reset.
        raw_ts = []
        for f in frame_data:
            try:
                raw_ts.append(float(f.get('timestamp', 0)))
            except Exception:
                raw_ts.append(0.0)

        pos_diffs = [raw_ts[i] - raw_ts[i - 1] for i in range(1, len(raw_ts)) if raw_ts[i] > raw_ts[i - 1]]
        median_dt_ms = float(np.median(pos_diffs)) if pos_diffs else 0.0

        offset = 0.0
        prev_adj = None
        for i, f in enumerate(frame_data):
            raw = raw_ts[i]
            if prev_adj is None:
                adj = raw
                dt = 0.0
            else:
                if raw + offset < prev_adj:
                    offset += (prev_adj - (raw + offset)) + (median_dt_ms if median_dt_ms > 0 else 0.0)
                adj = raw + offset
                dt = (adj - prev_adj) / 1000.0
                if dt < 0:
                    dt = 0.0
            f['dt'] = dt
            f['t_adj'] = adj
            prev_adj = adj

        # Pre-compute zone centroids for angle time
        zone_centroids = {}
        for zn in zone_names:
            zone_centroids[zn] = get_zone_center(zroot[zn].get('values', []))
        arena_center = get_arena_center(zroot)
        ia_edge_lines = {}
        ia_inner_lines = {}
        ia_line_normals = {}
        for zn in zone_names:
            if '_IA' in zn:
                zdata = zroot.get(zn, {})
                ia_offset = get_zone_ia_inner_offset(zdata, ia_inner_offset_px)
                ia_extend = get_zone_ia_line_extend(zdata, ia_line_extend_px)
                outer_line, inner_line, normal = build_ia_reference_lines(
                    zroot[zn].get('values', []),
                    arena_center,
                    line_extend_px=ia_extend,
                    inner_offset_px=ia_offset,
                    parallel_tol_deg=ia_parallel_tol_deg
                )
                ia_edge_lines[zn] = outer_line
                ia_inner_lines[zn] = inner_line
                ia_line_normals[zn] = normal

        # Group by state
        state_groups = {}
        for fd in frame_data:
            state = fd.get('state') or 'unknown'
            if state not in state_groups:
                state_groups[state] = []
            state_groups[state].append(fd)

        rows = []

        def _sum_distance_in_mask(distances_m: np.ndarray, mask: np.ndarray) -> float:
            if distances_m.size < 2:
                return 0.0
            mask = mask.astype(bool)
            mask_shift = mask & np.roll(mask, 1)
            mask_shift[0] = False
            return float(np.nansum(distances_m[mask_shift]))

        # Process each state
        for state_name, frames in state_groups.items():
            if not frames or state_name == 'unknown':
                continue

            # Time intervals array
            time_intervals = np.array([f.get('dt', 0) for f in frames], dtype=np.float64)

            # Total duration for this state
            total_duration = float(np.sum(time_intervals))

            # Distance and speed (meters)
            centers = [f.get('center') for f in frames]
            coords_px = np.array(
                [(c[0], c[1]) if c else (np.nan, np.nan) for c in centers],
                dtype=np.float64
            )
            coords_m = coords_px * float(scale_m_per_px)
            dx = np.diff(coords_m[:, 0], prepend=np.nan)
            dy = np.diff(coords_m[:, 1], prepend=np.nan)
            distances_m = np.sqrt(dx * dx + dy * dy)
            total_distance_m = float(np.nansum(distances_m))
            speeds_mps = np.divide(
                distances_m, time_intervals,
                out=np.full_like(distances_m, np.nan),
                where=time_intervals > 0
            )
            max_speed_mps = float(np.nanmax(speeds_mps)) if np.any(np.isfinite(speeds_mps)) else 0.0
            mean_speed_mps = float(np.nanmean(speeds_mps)) if np.any(np.isfinite(speeds_mps)) else 0.0
            immobile_mask = (speeds_mps < 0.01) & np.isfinite(speeds_mps) & (time_intervals > 0)
            immobile_time_s = float(np.sum(immobile_mask * time_intervals))
            center_distance_m = 0.0
            novel_distance_m = 0.0
            familiar_distance_m = 0.0

            # Pre-compute zone membership for BODY (center) for all frames
            zone_membership_body = {}
            for zone_name in zone_names:
                zone_membership_body[zone_name] = np.array(
                    [f.get('zone') == zone_name for f in frames], dtype=bool
                )

            # Pre-compute zone membership for HEAD for all frames
            zone_membership_head = {}
            for zone_name in zone_names:
                zone_membership_head[zone_name] = np.array(
                    [f.get('head_zone') == zone_name for f in frames], dtype=bool
                )

            # Pre-compute angle times and angle entries for _IA zones
            # IMPORTANT: angle_time = time IN zone AND facing target zone
            angle_times = {}
            angle_entries = {}
            ia_confirm_s = max(ia_confirm_ms, 0.0) / 1000.0
            FACING_CONFIRM_MS = max(ia_confirm_ms, 0.0)
            zone_membership_head_confirmed = {}
            for zone_name in zone_names:
                if '_IA' not in zone_name:
                    continue

                zc = resolve_ia_target_centroid(zroot, zone_name) or zone_centroids.get(zone_name)
                if not zc:
                    continue

                angle_time = 0.0
                angle_entry_count = 0
                head_in_zone = zone_membership_head.get(zone_name, np.zeros(len(frames), dtype=bool))
                inner_line = ia_inner_lines.get(zone_name)
                line_normal = ia_line_normals.get(zone_name)
                prev_facing_confirmed = False
                facing_start_time = None
                edge_line = ia_edge_lines.get(zone_name)
                confirmed_mask = np.zeros(len(frames), dtype=bool)
                in_zone_time = 0.0
                in_zone_confirmed = False

                for i, f in enumerate(frames):
                    # Use HEAD-in-zone for IA dwell/entry; head/body define facing vector.
                    is_in_zone = head_in_zone[i]
                    head = f.get('head')
                    body = f.get('center')
                    timestamp = f.get('t_adj', f.get('timestamp', 0))

                    crossed = True
                    if is_in_zone and inner_line and line_normal and head:
                        crossed = point_is_past_line((head[0], head[1]), inner_line[0], line_normal, toward_center=False)
                    is_in_zone = is_in_zone and crossed

                    if is_in_zone:
                        in_zone_time += time_intervals[i] if i < len(time_intervals) else 0.0
                    else:
                        in_zone_time = 0.0
                        in_zone_confirmed = False
                    if not in_zone_confirmed and in_zone_time >= ia_confirm_s:
                        in_zone_confirmed = True
                    confirmed_mask[i] = in_zone_confirmed

                    is_facing = False
                    if head and body and in_zone_confirmed and edge_line:
                        ang_edge, _ = compute_facing_angle_to_edge(
                            (head[0], head[1]), (body[0], body[1]),
                            edge_line[0], edge_line[1]
                        )
                        if ang_edge is not None and ang_edge <= angle_threshold:
                            is_facing = True

                    facing_confirmed = False
                    if is_facing:
                        if facing_start_time is None:
                            facing_start_time = timestamp
                        elif timestamp - facing_start_time >= FACING_CONFIRM_MS:
                            facing_confirmed = True
                    else:
                        facing_start_time = None

                    # Count angle entries on confirmed facing transition
                    if in_zone_confirmed and facing_confirmed and not prev_facing_confirmed:
                        angle_entry_count += 1

                    if in_zone_confirmed and is_facing:
                        angle_time += time_intervals[i] if i < len(time_intervals) else 0.0

                    prev_facing_confirmed = facing_confirmed

                angle_times[zone_name] = angle_time
                angle_entries[zone_name] = angle_entry_count
                zone_membership_head_confirmed[zone_name] = confirmed_mask

            start_time = metadata.get('time', '') or metadata.get('start_time', '')
            row = {
                'MouseID': metadata.get('mouseID', ''),
                'Group': metadata.get('group', ''),
                'StartTime': start_time,
                'ExptDate': metadata.get('expt_date', ''),
                'subgroup': metadata.get('subgroup', ''),
                'State': state_name,
                'StateDur_s': int(math.ceil(total_duration)),
                'Distance_m': round(total_distance_m, 5),
                'Max_speed': round(max_speed_mps, 4),
            }

            # Use REGIONS_HIERARCHY when exact sub-zones exist
            covered_zones = set()
            for main_region, sub_zones in REGIONS_HIERARCHY.items():
                matched = [sz for sz in sub_zones if sz in zone_names]
                if not matched:
                    continue
                covered_zones.update(matched)

                combined_body = np.zeros(len(frames), dtype=bool)
                for sz in matched:
                    combined_body |= zone_membership_body.get(sz, np.zeros(len(frames), dtype=bool))
                time_body = float(np.sum(time_intervals[combined_body])) if np.any(combined_body) else 0.0
                entries_body = count_zone_entries_with_dwell(combined_body, time_intervals, min_dwell_time)
                if main_region != 'CenterZone':
                    row[f'{main_region}_time_body'] = round(time_body, 3)
                    row[f'{main_region}_entries_body'] = entries_body
                if main_region == 'Center':
                    center_distance_m = _sum_distance_in_mask(distances_m, combined_body)
                elif main_region == 'NovelArm':
                    novel_distance_m = _sum_distance_in_mask(distances_m, combined_body)
                elif main_region == 'FamiliarArm':
                    familiar_distance_m = _sum_distance_in_mask(distances_m, combined_body)

                combined_head = np.zeros(len(frames), dtype=bool)
                for sz in matched:
                    if '_IA' in sz:
                        combined_head |= zone_membership_head_confirmed.get(sz, np.zeros(len(frames), dtype=bool))
                    else:
                        combined_head |= zone_membership_head.get(sz, np.zeros(len(frames), dtype=bool))
                time_head = float(np.sum(time_intervals[combined_head])) if np.any(combined_head) else 0.0
                entries_head = count_zone_entries_with_dwell(combined_head, time_intervals, min_dwell_time)
                row[f'{main_region}_time_head'] = round(time_head, 3)
                row[f'{main_region}_entries_head'] = entries_head

                if '_IA' in main_region:
                    total_angle_time = sum(angle_times.get(sz, 0) for sz in matched)
                    total_angle_entries = sum(angle_entries.get(sz, 0) for sz in matched)
                    row[f'{main_region}_time_angle'] = round(total_angle_time, 3)
                    if main_region == 'Object1_IA':
                        row['Object1_IA-entries_angle'] = total_angle_entries
                    else:
                        row[f'{main_region}_entries_angle'] = total_angle_entries

            row['Distance_Center_m'] = round(center_distance_m, 5)
            row['Distance_NovelArm_m'] = round(novel_distance_m, 5)
            row['Distance_FamiliarArm_m'] = round(familiar_distance_m, 5)

            # Per-zone stats for zones not covered by hierarchy (dynamic/experimental)
            for zone_name in zone_names:
                if zone_name in covered_zones:
                    continue
                body_mask = zone_membership_body.get(zone_name, np.zeros(len(frames), dtype=bool))
                time_body = float(np.sum(time_intervals[body_mask])) if np.any(body_mask) else 0.0
                entries_body = count_zone_entries_with_dwell(body_mask, time_intervals, min_dwell_time)
                row[f'{zone_name}_time_body'] = round(time_body, 3)
                row[f'{zone_name}_entries_body'] = entries_body

                if '_IA' in zone_name:
                    head_mask = zone_membership_head_confirmed.get(zone_name, np.zeros(len(frames), dtype=bool))
                else:
                    head_mask = zone_membership_head.get(zone_name, np.zeros(len(frames), dtype=bool))
                time_head = float(np.sum(time_intervals[head_mask])) if np.any(head_mask) else 0.0
                entries_head = count_zone_entries_with_dwell(head_mask, time_intervals, min_dwell_time)
                row[f'{zone_name}_time_head'] = round(time_head, 3)
                row[f'{zone_name}_entries_head'] = entries_head

                if '_IA' in zone_name:
                    row[f'{zone_name}_angle_time'] = round(angle_times.get(zone_name, 0.0), 3)
                    row[f'{zone_name}_angle_entries'] = angle_entries.get(zone_name, 0)

            rows.append(row)

        if not rows:
            return None

        result_df = pd.DataFrame(rows)
        desired_cols = [
            'MouseID', 'Group', 'ExptDate', 'StartTime', 'subgroup', 'State',
            'StateDur_s', 'Distance_m', 'Max_speed',
            'OpenArm_time_body', 'OpenEntry_time_body', 'Object1_IA_time_head',
            'Object2_IA_time_head', 'Social_IA_time_head', 'FamiliarArm_time_body',
            'NovelArm_time_body', 'Object1_IA_entries_head', 'Object2_IA_entries_head',
            'Social_IA_entries_head', 'Object1_IA_time_angle', 'Object2_IA_time_angle',
            'Object1_IA-entries_angle', 'Object2_IA_entries_angle',
            'Social_IA_time_angle', 'Social_IA_entries_angle',
            'Distance_Center_m', 'Distance_NovelArm_m', 'Distance_FamiliarArm_m',
            'Center_entries_body', 'Center_time_body', 'Center_entries_head', 'Center_time_head',
            'Social_IA_entries_body', 'Social_IA_time_body',
            'Social_Entry_entries_body', 'Social_Entry_time_body', 'Social_Entry_entries_head', 'Social_Entry_time_head',
            'OpenEntry_entries_body', 'OpenEntry_entries_head', 'OpenEntry_time_head',
            'OpenArm_entries_body', 'OpenArm_entries_head', 'OpenArm_time_head',
            'NovelArm_entries_body', 'NovelArm_entries_head', 'NovelArm_time_head',
            'Object1_IA_entries_body', 'Object1_IA_time_body',
            'Object1_Entry_entries_body', 'Object1_Entry_time_body', 'Object1_Entry_entries_head', 'Object1_Entry_time_head',
            'Object2_IA_entries_body', 'Object2_IA_time_body',
            'Object2_Entry_entries_body', 'Object2_Entry_time_body', 'Object2_Entry_entries_head', 'Object2_Entry_time_head',
            'FamiliarArm_entries_body', 'FamiliarArm_entries_head', 'FamiliarArm_time_head'
        ]
        text_cols = {'MouseID', 'Group', 'ExptDate', 'StartTime', 'subgroup', 'State'}
        for col in desired_cols:
            if col not in result_df.columns:
                result_df[col] = '' if col in text_cols else 0.0
        ordered_existing = [c for c in desired_cols if c in result_df.columns]
        trailing_cols = [c for c in result_df.columns if c not in desired_cols]
        result_df = result_df[ordered_existing + trailing_cols]
        excel_path = os.path.join(output_dir, f'{stem}_zone_stats.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            result_df.to_excel(writer, index=False, sheet_name='Zone Statistics')
            if params:
                param_rows = []
                for key, val in params.items():
                    if isinstance(val, (dict, list, tuple)):
                        try:
                            val_out = json.dumps(val)
                        except Exception:
                            val_out = str(val)
                    else:
                        val_out = str(val)
                    param_rows.append({'param': str(key), 'value': val_out})
                params_df = pd.DataFrame(param_rows)
                params_df.to_excel(writer, index=False, sheet_name='params')

        return excel_path

    except Exception as e:
        print(f"Statistics error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_zone_timeline_plot(stem: str, times: List[float], old_z: List, new_z: List,
                            states: List, output_dir: str, zones_full: Dict = None) -> Optional[str]:
    """Save zone timeline plot"""
    try:
        plt.ioff()
        if not times:
            return None

        x = np.array(times, dtype=np.float64)
        z_order = order_zones_spatial(zones_full) if zones_full else sorted(set(z for z in new_z if z))

        if not z_order:
            return None

        zid = {name: i for i, name in enumerate(z_order)}

        def to_ids(zlist):
            return np.array([zid.get(z, np.nan) + 1 if z in zid else np.nan for z in zlist], dtype=np.float64)

        y_new = to_ids(new_z)

        fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
        ax.step(x, y_new, where='post', linewidth=0.9, color='0.1')
        ax.set_ylim(0.5, len(z_order) + 0.5)
        ax.set_ylabel('Zone')
        ax.set_xlabel('Time (ms)')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f'{stem}_zone_timeline.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return out_path
    except:
        plt.close('all')
        return None


def save_distance_plot(stem: str, times: List[float], centers: List,
                       states: List, output_dir: str, smooth_sec: float = 1.0) -> Optional[str]:
    """Save distance/speed plot"""
    try:
        plt.ioff()
        if not times or not centers:
            return None

        t = np.array(times, dtype=np.float64)
        coords = np.array([(c[0] if c else np.nan, c[1] if c else np.nan) for c in centers], dtype=np.float64)

        dx = np.diff(coords[:, 0], prepend=np.nan)
        dy = np.diff(coords[:, 1], prepend=np.nan)
        dt = np.diff(t, prepend=t[0] if len(t) > 0 else 0)

        distances = np.sqrt(dx*dx + dy*dy)
        speeds = np.divide(distances, dt, out=np.full_like(distances, np.nan), where=dt > 0)

        fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
        ax.plot(t, speeds, linewidth=0.9)
        ax.set_ylabel('Speed (px/ms)')
        ax.set_xlabel('Time (ms)')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f'{stem}_distance.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return out_path
    except:
        plt.close('all')
        return None


# =============================================================================
# PIPELINE COMPONENTS
# =============================================================================

class FrameBuffer:
    """Lock-free ring buffer for frames using numpy arrays"""

    def __init__(self, capacity: int, frame_shape: Tuple[int, int, int]):
        self.capacity = capacity
        self.frame_shape = frame_shape
        self.buffer = np.zeros((capacity,) + frame_shape, dtype=np.uint8)
        self.metadata = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put(self, frame: np.ndarray, meta: Any, timeout: float = 1.0) -> bool:
        with self.not_full:
            if self.size >= self.capacity:
                if not self.not_full.wait(timeout):
                    return False

            self.buffer[self.tail] = frame
            self.metadata[self.tail] = meta
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1
            self.not_empty.notify()
            return True

    def get(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, Any]]:
        with self.not_empty:
            if self.size == 0:
                if not self.not_empty.wait(timeout):
                    return None

            if self.size == 0:
                return None

            frame = self.buffer[self.head].copy()
            meta = self.metadata[self.head]
            self.metadata[self.head] = None
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            self.not_full.notify()
            return frame, meta


class FrameReaderThread(threading.Thread):
    """Reads frames ahead into buffer"""

    def __init__(self, video_path: str, output_buffer: FrameBuffer,
                 txt_iter, skip_frames: int = 1):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.output_buffer = output_buffer
        self.txt_iter = txt_iter
        self.skip_frames = skip_frames
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.total_frames = 0

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames
                if self.frame_count % self.skip_frames != 0:
                    self.frame_count += 1
                    try:
                        next(self.txt_iter)
                    except StopIteration:
                        pass
                    continue

                # Get original data
                try:
                    orig_data = next(self.txt_iter)
                except StopIteration:
                    orig_data = {}

                meta = {
                    'frame_idx': self.frame_count,
                    'orig_data': orig_data,
                    'timestamp': self._get_timestamp(orig_data, cap)
                }

                if not self.output_buffer.put(frame, meta, timeout=2.0):
                    if self.stop_event.is_set():
                        break

                self.frame_count += 1
        finally:
            cap.release()
            # Signal end
            self.output_buffer.put(np.zeros(self.output_buffer.frame_shape, dtype=np.uint8),
                                   {'end': True}, timeout=5.0)

    def _get_timestamp(self, orig_data: Dict, cap) -> float:
        for key in ('timestamp', 'timestamps', 'ts', 'time', 't'):
            if key in orig_data:
                try:
                    return float(orig_data[key])
                except:
                    pass
        return cap.get(cv2.CAP_PROP_POS_MSEC)

    def stop(self):
        self.stop_event.set()


class DLCInferenceThread(threading.Thread):
    """Runs DLC inference - uses GPU so runs in thread not process"""

    def __init__(self, input_buffer: FrameBuffer, output_queue: queue.Queue,
                 model_path: str, crop_box, resize: float = 0.4,
                 gpu_id: str = '0', batch_size: int = 1):
        super().__init__(daemon=True)
        self.input_buffer = input_buffer
        self.output_queue = output_queue
        self.model_path = model_path
        self.crop_box = crop_box
        self.resize = resize
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.stop_event = threading.Event()
        self.dlc = None

    def run(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        try:
            from dlclive import DLCLive
            self.dlc = DLCLive(self.model_path, resize=self.resize)
            initialized = False

            while not self.stop_event.is_set():
                result = self.input_buffer.get(timeout=0.5)
                if result is None:
                    continue

                frame, meta = result

                if meta.get('end'):
                    self.output_queue.put({'end': True})
                    break

                # Crop and process
                x1, y1, x2, y2 = self.crop_box
                crop = frame[y1:y2, x1:x2]
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                if not initialized:
                    self.dlc.init_inference(rgb)
                    initialized = True

                pose = self.dlc.get_pose(rgb)

                if pose is not None:
                    if not isinstance(pose, np.ndarray):
                        pose = np.array(pose, dtype=np.float32)
                    if pose.ndim == 1:
                        pose = pose.reshape(-1, 3)
                    # Offset to full frame coords
                    pose[:, 0] += x1
                    pose[:, 1] += y1
                else:
                    pose = np.zeros((0, 3), dtype=np.float32)

                self.output_queue.put({
                    'frame': frame,
                    'meta': meta,
                    'pose': pose
                })

        except Exception as e:
            self.output_queue.put({'error': str(e), 'end': True})

    def stop(self):
        self.stop_event.set()


class CorrectionManager:
    """Manages batch correction using process pool"""

    def __init__(self, config: Dict, num_workers: int = None):
        self.config = config
        self.num_workers = num_workers or NUM_CORRECTION_WORKERS
        self.pool = None
        self.pending_batches = {}
        self.batch_counter = 0

        # Accumulation buffers
        self.x_buffer = []
        self.y_buffer = []
        self.conf_buffer = []
        self.ts_buffer = []
        self.meta_buffer = []
        self.frame_buffer = []  # Store frames for synchronization!

    def start(self):
        self.pool = ProcessPoolExecutor(max_workers=self.num_workers)

    def stop(self):
        if self.pool:
            self.pool.shutdown(wait=False)

    def add_frame(self, pose: np.ndarray, timestamp: float, meta: Dict, frame: np.ndarray = None) -> Optional[List[Dict]]:
        """Add frame to batch, returns corrected results when batch is complete"""
        if pose.size == 0:
            self.x_buffer.append(np.full(10, np.nan))  # Placeholder
            self.y_buffer.append(np.full(10, np.nan))
            self.conf_buffer.append(np.zeros(10))
        else:
            self.x_buffer.append(pose[:, 0])
            self.y_buffer.append(pose[:, 1])
            self.conf_buffer.append(pose[:, 2])

        self.ts_buffer.append(timestamp)
        self.meta_buffer.append(meta)
        self.frame_buffer.append(frame.copy() if frame is not None else None)

        # Check if batch is ready
        if len(self.x_buffer) >= CORRECTION_BATCH_SIZE:
            return self._submit_and_get_results()

        return None

    def flush(self) -> Optional[List[Dict]]:
        """Flush remaining frames"""
        if self.x_buffer:
            return self._submit_and_get_results()
        return None

    def _submit_and_get_results(self) -> List[Dict]:
        """Submit batch for processing and return results"""
        # Pad arrays to same size
        max_kp = max(len(x) for x in self.x_buffer)
        x_arr = np.full((len(self.x_buffer), max_kp), np.nan, dtype=np.float32)
        y_arr = np.full((len(self.y_buffer), max_kp), np.nan, dtype=np.float32)
        conf_arr = np.zeros((len(self.conf_buffer), max_kp), dtype=np.float32)

        for i, (x, y, c) in enumerate(zip(self.x_buffer, self.y_buffer, self.conf_buffer)):
            x_arr[i, :len(x)] = x
            y_arr[i, :len(y)] = y
            conf_arr[i, :len(c)] = c

        batch_data = {
            'batch_idx': self.batch_counter,
            'x': x_arr,
            'y': y_arr,
            'conf': conf_arr,
            'timestamps': np.array(self.ts_buffer, dtype=np.float64),
            'config': self.config,
            'head_idx': self.config.get('head_idx', 0),
            'body_idx': self.config.get('body_idx', 1)
        }

        # Process synchronously for now (can be made async)
        result = correction_worker(batch_data)

        # Build output
        outputs = []
        if result['success']:
            for i, meta in enumerate(self.meta_buffer):
                kp_count = len(self.x_buffer[i])
                outputs.append({
                    'x': result['x'][i, :kp_count],
                    'y': result['y'][i, :kp_count],
                    'conf': result['conf'][i, :kp_count],
                    'meta': meta,
                    'frame': self.frame_buffer[i]  # Include corresponding frame
                })
        else:
            # Return uncorrected on error
            for i, meta in enumerate(self.meta_buffer):
                outputs.append({
                    'x': self.x_buffer[i],
                    'y': self.y_buffer[i],
                    'conf': self.conf_buffer[i],
                    'meta': meta,
                    'frame': self.frame_buffer[i]  # Include corresponding frame
                })

        # Clear buffers
        self.x_buffer.clear()
        self.y_buffer.clear()
        self.conf_buffer.clear()
        self.ts_buffer.clear()
        self.meta_buffer.clear()
        self.frame_buffer.clear()  # Clear frame buffer too
        self.batch_counter += 1

        return outputs


class VideoWriterThread(threading.Thread):
    """Async video writer with timeline remux support"""

    def __init__(self, output_path: str, fps: float, size: Tuple[int, int],
                 queue_size: int = WRITER_BUFFER_SIZE, use_timeline_remux: bool = False):
        super().__init__(daemon=True)
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.frames_written = 0
        self.use_timeline_remux = use_timeline_remux
        self.t0 = None  # First timestamp for timeline remux
        self.writer_frame_idx = -1  # Last written frame index

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.size)

        try:
            while not self.stop_event.is_set():
                try:
                    item = self.queue.get(timeout=0.5)
                    if item is None:
                        break

                    # Unpack frame and optional timestamp
                    if isinstance(item, tuple):
                        frame, timestamp = item
                    else:
                        frame = item
                        timestamp = None

                    if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
                        frame = cv2.resize(frame, self.size)

                    # Timeline remux: emit frames based on timestamp
                    if self.use_timeline_remux and timestamp is not None:
                        if self.t0 is None:
                            self.t0 = timestamp

                        emit_idx = int(round(max(0.0, (timestamp - self.t0) / 1000.0) * self.fps))
                        frames_to_write = emit_idx - self.writer_frame_idx

                        if self.writer_frame_idx < 0 and frames_to_write <= 0:
                            frames_to_write = 1

                        frames_to_write = max(0, min(frames_to_write, 10))  # Cap at 10 to avoid memory issues

                        for _ in range(int(frames_to_write)):
                            writer.write(frame)
                            self.frames_written += 1

                        self.writer_frame_idx = emit_idx
                    else:
                        writer.write(frame)
                        self.frames_written += 1

                except queue.Empty:
                    continue
        finally:
            writer.release()

    def write(self, frame: np.ndarray, timestamp: float = None):
        """Write frame to queue. If timeline remux enabled, include timestamp."""
        try:
            if self.use_timeline_remux and timestamp is not None:
                self.queue.put((frame, timestamp), timeout=0.5)
            else:
                self.queue.put(frame, timeout=0.5)
        except queue.Full:
            pass  # Drop frame if queue is full

    def stop(self):
        self.stop_event.set()
        try:
            self.queue.put(None, timeout=0.5)
        except:
            pass


# =============================================================================
# ZONE OPERATIONS (Optimized)
# =============================================================================

@dataclass
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        # Ensure correct order
        self.x1, self.x2 = min(self.x1, self.x2), max(self.x1, self.x2)
        self.y1, self.y2 = min(self.y1, self.y2), max(self.y1, self.y2)

    @property
    def w(self) -> int:
        return int(self.x2 - self.x1)

    @property
    def h(self) -> int:
        return int(self.y2 - self.y1)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def crop_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Crop a BGR frame to this box"""
        return frame_bgr[self.y1:self.y2, self.x1:self.x2, :]

    def to_full(self, xy: Tuple[float, float]) -> Tuple[float, float]:
        """Convert crop coordinates to full frame coordinates"""
        x, y = xy
        return (x + self.x1, y + self.y1)


def crop_from_V(V: List[int]) -> CropBox:
    """Create CropBox from V list. V format is [y1, y2, x1, x2]"""
    y1, y2, x1, x2 = [int(v) for v in V]
    return CropBox(x1=x1, y1=y1, x2=x2, y2=y2)


def zones_are_cropped(zones: Dict, crop: CropBox) -> bool:
    """Check if zones appear to be in cropped coordinates (need shifting)"""
    if not zones:
        return False
    if zones.get('coords') == 'full' or zones.get('_coords') == 'full' or zones.get('full_coords') is True:
        return False
    zroot = zones_root(zones)
    eps = 5.0
    max_x = float('-inf')
    max_y = float('-inf')

    for z in zroot.values():
        for p in z.get('values', []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = float(p[0]), float(p[1])
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                # If any point is near origin (less than crop offset), zones are cropped
                if x < crop.x1 + eps or y < crop.y1 + eps:
                    return True
    # If all points fit within crop dims, treat as cropped
    if max_x <= crop.w + eps and max_y <= crop.h + eps:
        return True
    return False


def shift_zones_to_full(zones: Dict, crop: CropBox) -> Dict:
    """Shift zones from crop coordinates to full frame coordinates"""
    z = copy.deepcopy(zones)
    z['coords'] = 'full'
    zroot = zones_root(z)

    dx, dy = float(crop.x1), float(crop.y1)

    for item in zroot.values():
        vals = item.get('values', [])
        if vals:
            vals_array = np.array(vals, dtype=np.float32)
            vals_array[:, 0] += dx
            vals_array[:, 1] += dy
            item['values'] = vals_array.tolist()

    # Also shift arena if present
    if 'arena' in z and isinstance(z['arena'], dict) and 'arena' in z['arena']:
        a = z['arena']['arena']
        if isinstance(a, dict) and 'values' in a:
            vals = a['values']
            if vals:
                vals_array = np.array(vals, dtype=np.float32)
                vals_array[:, 0] += dx
                vals_array[:, 1] += dy
                a['values'] = vals_array.tolist()

    return z


def zones_root(z: Dict) -> Dict:
    return z.get('zones', z)


def zones_deepcopy(z: Dict) -> Dict:
    return copy.deepcopy(z)


def precompute_zone_layer(w: int, h: int, zones_full: Dict) -> Optional[np.ndarray]:
    """Pre-compute zone overlay layer"""
    if not zones_full:
        return None

    # Create contiguous array for OpenCV
    layer = np.zeros((h, w, 3), dtype=np.uint8, order='C')
    layer = np.ascontiguousarray(layer)

    palette = [(150, 50, 255), (1, 190, 200), (255, 128, 0), (238, 130, 238),
               (0, 165, 255), (180, 130, 70), (50, 205, 154), (180, 105, 255)]

    zroot = zones_root(zones_full)
    if not zroot:
        return None

    for i, (name, z) in enumerate(zroot.items()):
        pts = z.get('values', [])
        if len(pts) >= 3:
            try:
                # Ensure contiguous int32 array with proper shape
                poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                poly = np.ascontiguousarray(poly)
                cv2.fillPoly(layer, [poly], palette[i % len(palette)])
                cv2.polylines(layer, [poly], True, (255, 255, 255), 1)
            except Exception as e:
                print(f"Warning: Could not draw zone {name}: {e}")
                continue

    return layer


class FastZoneAssigner:
    """Optimized zone assignment with spatial indexing"""

    def __init__(self, zones_full: Dict):
        self.zone_names = []
        self.zone_polys = []
        self.zone_bounds = []

        zroot = zones_root(zones_full)
        for name, z in zroot.items():
            pts = z.get('values', [])
            if len(pts) >= 3:
                arr = np.array(pts, dtype=np.float32)
                self.zone_names.append(name)
                self.zone_polys.append(arr)
                self.zone_bounds.append([
                    float(np.min(arr[:, 0])), float(np.min(arr[:, 1])),
                    float(np.max(arr[:, 0])), float(np.max(arr[:, 1]))
                ])

        self.bounds_arr = np.array(self.zone_bounds, dtype=np.float32) if self.zone_bounds else np.zeros((0, 4))

    def locate(self, x: float, y: float) -> Optional[str]:
        if np.isnan(x) or np.isnan(y):
            return None

        for i, (name, poly) in enumerate(zip(self.zone_names, self.zone_polys)):
            # Fast bounding box check
            if not (self.bounds_arr[i, 0] <= x <= self.bounds_arr[i, 2] and
                    self.bounds_arr[i, 1] <= y <= self.bounds_arr[i, 3]):
                continue

            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                return name

        return None

    def locate_batch(self, centers: List[Tuple[float, float]]) -> List[Optional[str]]:
        """Batch zone assignment"""
        return [self.locate(c[0], c[1]) if c else None for c in centers]


# =============================================================================
# FILE PARSING (Optimized)
# =============================================================================

def parse_headers(txt_path: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[List]]:
    meta = zones = V = None
    with open(txt_path, 'r', encoding='utf-8', buffering=65536) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('B '):
                meta = json.loads(s[2:])
            elif s.startswith('M '):
                zones = json.loads(s[2:])
            elif s.startswith('V '):
                raw = s[2:].strip()
                V = json.loads(raw) if raw.startswith('[') else eval(raw)
            elif s.startswith('{'):
                break
    return meta, zones, V


def iter_original_frames(txt_path: str):
    """Generator that yields frame data from txt file. Handles early termination gracefully."""
    f = None
    try:
        f = open(txt_path, 'r', encoding='utf-8', buffering=65536)
        for line in f:
            s = line.strip()
            if not s or s.startswith(('B ', 'M ', 'V ', 'P ')):
                continue
            if s.startswith('{'):
                try:
                    yield json.loads(s)
                except GeneratorExit:
                    return  # Properly stop on close
                except:
                    yield {}
    except GeneratorExit:
        return  # Properly stop on close
    finally:
        if f:
            f.close()


def _pose_looks_cropped(pose: np.ndarray, crop: 'CropBox') -> bool:
    if pose.size == 0 or crop is None:
        return False
    xs = pose[:, 0]
    ys = pose[:, 1]
    if xs.size == 0 or ys.size == 0:
        return False
    max_x = np.nanmax(xs)
    max_y = np.nanmax(ys)
    # Heuristic: values fit inside crop dims (assume cropped coords)
    if max_x <= crop.w + 5 and max_y <= crop.h + 5:
        return True
    return False


def _pose_looks_full(pose: np.ndarray, crop: 'CropBox') -> bool:
    if pose.size == 0 or crop is None:
        return False
    xs = pose[:, 0]
    ys = pose[:, 1]
    if xs.size == 0 or ys.size == 0:
        return False
    min_x = np.nanmin(xs)
    min_y = np.nanmin(ys)
    max_x = np.nanmax(xs)
    max_y = np.nanmax(ys)
    if (min_x >= crop.x1 - 5 and max_x <= crop.x2 + 5 and
        min_y >= crop.y1 - 5 and max_y <= crop.y2 + 5):
        return True
    return False


def adjust_pose_for_frame(pose: np.ndarray, crop: Optional['CropBox'],
                          frame_w: int, frame_h: int) -> np.ndarray:
    """Align pose coordinates to the actual frame size."""
    if pose.size == 0 or crop is None:
        return pose

    if frame_w == crop.w and frame_h == crop.h:
        # Frame is already cropped -> convert full coords to crop coords if needed
        if _pose_looks_full(pose, crop):
            pose[:, 0] -= float(crop.x1)
            pose[:, 1] -= float(crop.y1)
    else:
        # Frame is full -> convert crop coords to full coords if needed
        if _pose_looks_cropped(pose, crop):
            pose[:, 0] += float(crop.x1)
            pose[:, 1] += float(crop.y1)

    return pose


def extract_pose_array(orig_data: Dict, crop: Optional['CropBox'] = None,
                       frame_w: Optional[int] = None, frame_h: Optional[int] = None) -> np.ndarray:
    """Extract pose array from original data record."""
    if not orig_data:
        return np.zeros((0, 3), dtype=np.float32)

    for key in ('pose_array', 'pose', 'poses'):
        if key in orig_data:
            arr = np.array(orig_data.get(key), dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 3)
            if arr.shape[1] == 2:
                conf = np.ones((arr.shape[0], 1), dtype=np.float32)
                arr = np.concatenate([arr, conf], axis=1)
            if arr.shape[1] >= 3:
                arr = arr[:, :3]
                if crop and frame_w and frame_h:
                    arr = adjust_pose_for_frame(arr, crop, frame_w, frame_h)
                return arr
            break

    return np.zeros((0, 3), dtype=np.float32)


def parse_video_name_from_txt(txt_name: str) -> Tuple[str, str]:
    """Parse video stem name from txt filename. Handles multiple formats."""
    base = os.path.basename(txt_name)
    name_no_ext = os.path.splitext(base)[0]

    # Format 1: subject_video_data_-timestamp.txt -> subject-timestamp
    if '_video_data_-' in base:
        parts = base.split('_video_data_-')
        subject = parts[0]
        ts = parts[-1].replace('.txt', '')
        return f'{subject}-{ts}', ts

    # Format 2: subject_video_data-timestamp.txt (without underscore before dash)
    if '_video_data-' in base:
        parts = base.split('_video_data-')
        subject = parts[0]
        ts = parts[-1].replace('.txt', '')
        return f'{subject}-{ts}', ts

    # Format 3: subject-timestamp.txt
    if '-' in name_no_ext:
        return name_no_ext, name_no_ext.split('-')[-1]

    # Format 4: Just use filename without extension as stem
    return name_no_ext, name_no_ext


def find_video_path(vdir: str, stem: str) -> Optional[str]:
    """Find video file matching stem. Returns None if not found."""
    if not vdir or not os.path.isdir(vdir):
        return None

    # Direct match
    for ext in ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'):
        p = os.path.join(vdir, stem + ext)
        if os.path.exists(p):
            return p

    # Try partial match (stem contained in filename)
    for ext in ('.mp4', '.avi', '.mov', '.mkv'):
        for vf in glob.glob(os.path.join(vdir, f'*{ext}')):
            vf_base = os.path.splitext(os.path.basename(vf))[0]
            if stem in vf_base or vf_base in stem:
                return vf

    # Try matching by timestamp part
    if '-' in stem:
        ts_part = stem.split('-')[-1]
        for ext in ('.mp4', '.avi', '.mov', '.mkv'):
            for vf in glob.glob(os.path.join(vdir, f'*{ts_part}*{ext}')):
                return vf

    return None


def discover_bodyparts(model_dir: str) -> List[str]:
    if not (model_dir and YAML_OK):
        return []
    for ext in ('*.yaml', '*.yml'):
        for yp in glob.glob(os.path.join(model_dir, '**', ext), recursive=True):
            try:
                with open(yp, 'r') as f:
                    cfg = yaml.safe_load(f)
                for key in ('all_joints_names', 'bodyparts'):
                    if key in cfg:
                        return list(cfg[key])
            except:
                continue
    return []


def guess_head_body_indices(bodyparts: List[str]) -> Optional[Tuple[int, int]]:
    if not bodyparts:
        return None
    names = [str(n).lower() for n in bodyparts]
    head_keys = ['nose', 'snout', 'head', 'ear', 'left_ear', 'right_ear', 'leftear', 'rightear', 'face', 'forehead']
    body_keys = ['body', 'center', 'torso', 'mid', 'spine', 'neck', 'thorax', 'back']

    def find_idx(keys):
        for key in keys:
            for i, n in enumerate(names):
                if key in n:
                    return i
        return None

    head_idx = find_idx(head_keys)
    body_idx = find_idx(body_keys)

    if head_idx is None and body_idx is None:
        return None
    if head_idx is None:
        head_idx = 0 if body_idx != 0 else (1 if len(names) > 1 else 0)
    if body_idx is None:
        if len(names) > 1:
            body_idx = 1 if head_idx != 1 else 0
        else:
            body_idx = head_idx
    if head_idx == body_idx and len(names) > 1:
        body_idx = 0 if head_idx != 0 else 1
    return head_idx, body_idx


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

class PipelineSignals(QObject):
    progress = pyqtSignal(int, str)
    file_started = pyqtSignal(str)
    file_completed = pyqtSignal(str, bool, str)
    log = pyqtSignal(str)
    finished = pyqtSignal()


class HighPerformancePipeline(QThread):
    """
    High-performance processing pipeline using:
    - Threaded frame reading
    - GPU-based DLC inference
    - Multiprocessing for corrections
    - Threaded video writing
    - Batch processing throughout
    """

    def __init__(self, config: Dict, file_info: Dict):
        super().__init__()
        self.config = config
        self.file_info = file_info
        self.signals = PipelineSignals()
        self.is_running = True
        self.excel_paths = []  # Track Excel files for cumulative saving

    def stop(self):
        self.is_running = False
        STOP_FLAG.set()

    def run(self):
        STOP_FLAG.clear()
        self.excel_paths = []  # Reset for new run

        valid_files = {k: v for k, v in self.file_info.items() if not v.get('error')}
        total_files = len(valid_files)

        if total_files == 0:
            self.signals.log.emit("No valid files")
            self.signals.finished.emit()
            return

        for file_idx, (base, info) in enumerate(valid_files.items()):
            if not self.is_running or STOP_FLAG.is_set():
                break

            self.signals.file_started.emit(base)

            try:
                success, msg, excel_path = self._process_file(base, info)
                self.signals.file_completed.emit(base, success, msg)
                if excel_path and os.path.exists(excel_path):
                    self.excel_paths.append(excel_path)
            except Exception as e:
                self.signals.file_completed.emit(base, False, str(e))

            self.signals.progress.emit(int((file_idx + 1) / total_files * 100),
                                       f"Processed {file_idx + 1}/{total_files}")

        # Create cumulative Excel after all files processed
        if self.excel_paths and len(self.excel_paths) > 0:
            try:
                cumulative_path = self._create_cumulative_excel()
                if cumulative_path:
                    self.signals.log.emit(f"Cumulative Excel saved: {os.path.basename(cumulative_path)}")
            except Exception as e:
                self.signals.log.emit(f"Cumulative Excel error: {e}")

        self.signals.finished.emit()

    def _create_cumulative_excel(self) -> Optional[str]:
        """Concatenate all individual Excel files into one cumulative file."""
        import pandas as pd

        if not self.excel_paths:
            return None

        all_dfs = []
        for excel_path in self.excel_paths:
            try:
                df = pd.read_excel(excel_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {excel_path}: {e}")
                continue

        if not all_dfs:
            return None

        # Concatenate all DataFrames
        cumulative_df = pd.concat(all_dfs, ignore_index=True)

        # Save cumulative file
        output_dir = self.config.get('output_dir', '.')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        cumulative_path = os.path.join(output_dir, f'cumulative_zone_stats_{timestamp}.xlsx')
        cumulative_df.to_excel(cumulative_path, index=False, sheet_name='Cumulative Statistics')

        return cumulative_path

    def _process_file(self, base: str, info: Dict) -> Tuple[bool, str, Optional[str]]:
        """Process a single file. Returns (success, message, excel_path)."""
        config = self.config
        txt_path = info['txt_path']
        video_path = info.get('video_path')
        stem = info['stem']
        V = info.get('V', [0, 100, 0, 100])
        crop = info.get('crop') or crop_from_V(V)

        # Analysis-only: use existing tracking in TXT, no correction/video/output txt
        if config.get('processing_mode') == 'analysis_only':
            zones_full = info.get('zones_full') or info.get('zones', {})
            zones_orig = info.get('zones', {})
            zones_for_video = zones_full or zones_orig
            zone_assigner = FastZoneAssigner(zones_for_video) if zones_for_video else None

            # Buffers for statistics/plots
            times_buf = []
            zones_buf = []
            old_zones_buf = []
            states_buf = []
            centers_buf = []
            heads_buf = []

            head_idx = config.get('head_idx', 0)
            body_idx = config.get('body_idx', 1)
            pcut = config.get('pcut', 0.1)

            for rec in iter_original_frames(txt_path):
                if not rec:
                    continue
                pose = rec.get('pose_array') or []
                if not pose:
                    continue
                pose_arr = np.array(pose, dtype=np.float32)

                x = pose_arr[:, 0].copy() if pose_arr.size else np.array([], dtype=np.float32)
                y = pose_arr[:, 1].copy() if pose_arr.size else np.array([], dtype=np.float32)
                conf = pose_arr[:, 2].copy() if pose_arr.size and pose_arr.shape[1] > 2 else np.zeros_like(x)

                for i in range(len(x)):
                    if len(conf) > i and conf[i] < pcut:
                        x[i] = np.nan
                        y[i] = np.nan

                # Center (prefer stored center if present)
                center = None
                cx, cy = np.nan, np.nan
                if isinstance(rec.get('center'), list) and len(rec['center']) >= 2:
                    try:
                        cx, cy = float(rec['center'][0]), float(rec['center'][1])
                        center = (cx, cy)
                    except Exception:
                        center = None
                if center is None:
                    if config.get('infer_center_from_all', True):
                        valid = ~(np.isnan(x) | np.isnan(y))
                        if np.any(valid):
                            cx = float(np.mean(x[valid]))
                            cy = float(np.mean(y[valid]))
                            center = (cx, cy)
                    else:
                        if len(x) > body_idx and not np.isnan(x[body_idx]) and not np.isnan(y[body_idx]):
                            cx, cy = float(x[body_idx]), float(y[body_idx])
                            center = (cx, cy)

                head = None
                if len(x) > head_idx and not np.isnan(x[head_idx]) and not np.isnan(y[head_idx]):
                    head = (float(x[head_idx]), float(y[head_idx]))

                # Shift pose from crop to full-frame coordinates to match zone polygons
                dx_crop, dy_crop = float(crop.x1), float(crop.y1)
                if center:
                    cx_full = cx + dx_crop
                    cy_full = cy + dy_crop
                    center = (cx_full, cy_full)
                    cx, cy = cx_full, cy_full
                if head:
                    head = (head[0] + dx_crop, head[1] + dy_crop)

                zone = zone_assigner.locate(cx, cy) if (zone_assigner and center) else None
                head_zone = zone_assigner.locate(head[0], head[1]) if (zone_assigner and head) else None

                old_zone = rec.get('location')
                state = rec.get('state') or rec.get('state_name')
                timestamp = rec.get('timestamp', rec.get('timestamps', rec.get('ts', rec.get('time', rec.get('t', 0)))))

                times_buf.append(timestamp)
                zones_buf.append(zone)
                old_zones_buf.append(old_zone)
                states_buf.append(state)
                centers_buf.append(center)
                heads_buf.append(head)

            excel_path = None
            outputs = []
            if config.get('use_zones', True) and zones_for_video and times_buf:
                excel_path = calculate_zone_statistics_full(
                    stem, times_buf, zones_buf, states_buf,
                    config['output_dir'], zones_for_video, centers_buf, heads_buf,
                    config.get('min_dwell_time', 0.2),
                    zone_assigner=zone_assigner,
                    file_meta=info.get('meta'),
                    angle_threshold=config.get('angle_threshold', 45.0),
                    ia_confirm_ms=config.get('ia_confirm_ms', 100.0),
                    ia_line_extend_px=config.get('ia_line_extend_px', 15.0),
                    ia_inner_offset_px=config.get('ia_inner_offset_px', 50.0),
                    ia_parallel_tol_deg=config.get('ia_parallel_tol_deg', 15.0),
                    params=config
                )
                if excel_path:
                    outputs.append(f"Excel: {os.path.basename(excel_path)}")

                if config.get('save_timeline_plot', True):
                    timeline_path = save_zone_timeline_plot(
                        stem, times_buf, old_zones_buf, zones_buf, states_buf,
                        config['output_dir'], zones_for_video
                    )
                    if timeline_path:
                        outputs.append("Timeline plot")

                if config.get('save_distance_plot', True):
                    dist_path = save_distance_plot(
                        stem, times_buf, centers_buf, states_buf,
                        config['output_dir'], config.get('speed_smooth_sec', 1.0)
                    )
                    if dist_path:
                        outputs.append("Distance plot")

            if excel_path:
                msg = f"Analysis complete ({', '.join(outputs)})"
                return True, msg, excel_path
            return False, "Analysis failed or no data", None

        # Correct-only: apply corrections to tracking data from TXT, write corrected TXT + Excel, no video needed
        if config.get('processing_mode') == 'correct_only':
            zones_full = info.get('zones_full') or info.get('zones', {})
            zones_orig = info.get('zones', {})
            zones_for_video = zones_full or zones_orig
            zone_assigner = FastZoneAssigner(zones_for_video) if zones_for_video else None

            head_idx = config.get('head_idx', 0)
            body_idx = config.get('body_idx', 1)
            pcut = config.get('pcut', 0.1)

            # Collect all frames from TXT for batch correction
            all_records = []
            all_poses = []
            all_timestamps = []

            for rec in iter_original_frames(txt_path):
                if not rec:
                    continue
                pose = rec.get('pose_array') or []
                if not pose:
                    all_records.append(rec)
                    all_poses.append(np.zeros((0, 3), dtype=np.float32))
                    all_timestamps.append(0.0)
                    continue
                pose_arr = np.array(pose, dtype=np.float32)
                if pose_arr.ndim == 1:
                    pose_arr = pose_arr.reshape(-1, 3)
                if pose_arr.shape[1] == 2:
                    conf_col = np.ones((pose_arr.shape[0], 1), dtype=np.float32)
                    pose_arr = np.concatenate([pose_arr, conf_col], axis=1)
                pose_arr = pose_arr[:, :3]

                ts = 0.0
                for key in ('timestamp', 'timestamps', 'ts', 'time', 't'):
                    if key in rec:
                        try:
                            ts = float(rec[key])
                            break
                        except Exception:
                            pass

                all_records.append(rec)
                all_poses.append(pose_arr)
                all_timestamps.append(ts)

            if not all_records:
                return False, "No data in TXT file", None

            # Run correction via CorrectionManager
            correction_mgr = CorrectionManager(config)
            correction_mgr.start()

            corrected_results = []
            for i, (rec, pose_arr, ts) in enumerate(zip(all_records, all_poses, all_timestamps)):
                meta = {'orig_data': rec, 'timestamp': ts, 'frame_idx': i}
                batch_out = correction_mgr.add_frame(pose_arr, ts, meta, None)
                if batch_out:
                    corrected_results.extend(batch_out)

            remaining = correction_mgr.flush()
            if remaining:
                corrected_results.extend(remaining)
            correction_mgr.stop()

            # Write corrected TXT and collect stats
            out_txt = os.path.join(config['output_dir'], base.replace('.txt', '_retracked.txt'))
            if not config.get('overwrite') and os.path.exists(out_txt):
                return True, "Skipped (exists)", None

            times_buf = []
            zones_buf = []
            old_zones_buf = []
            states_buf = []
            centers_buf = []
            heads_buf = []

            with open(out_txt, 'w', encoding='utf-8', buffering=65536) as out_file:
                out_file.write(f'B {json.dumps(info.get("meta"))}\n')
                out_file.write(f'M {json.dumps(zones_full)}\n')
                out_file.write(f'V {json.dumps(V)}\n')

                for res in corrected_results:
                    x = res['x']
                    y = res['y']
                    conf = res['conf']
                    meta = res['meta']
                    orig_data = meta.get('orig_data', {})

                    # Filter by confidence
                    for j in range(len(x)):
                        if len(conf) > j and conf[j] < pcut:
                            x[j] = np.nan
                            y[j] = np.nan

                    # Calculate center
                    center = None
                    cx, cy = np.nan, np.nan
                    if config.get('infer_center_from_all', True):
                        valid = ~(np.isnan(x) | np.isnan(y))
                        if np.any(valid):
                            cx = float(np.mean(x[valid]))
                            cy = float(np.mean(y[valid]))
                            center = (cx, cy)
                    else:
                        if len(x) > body_idx and not np.isnan(x[body_idx]) and not np.isnan(y[body_idx]):
                            cx, cy = float(x[body_idx]), float(y[body_idx])
                            center = (cx, cy)

                    # Head position
                    head = None
                    if len(x) > head_idx and not np.isnan(x[head_idx]) and not np.isnan(y[head_idx]):
                        head = (float(x[head_idx]), float(y[head_idx]))

                    # Shift pose from crop to full-frame coordinates to match zone polygons
                    dx_crop, dy_crop = float(crop.x1), float(crop.y1)
                    if center:
                        cx = cx + dx_crop
                        cy = cy + dy_crop
                        center = (cx, cy)
                    if head:
                        head = (head[0] + dx_crop, head[1] + dy_crop)

                    # Zone assignment
                    zone = zone_assigner.locate(cx, cy) if (zone_assigner and center) else None
                    old_zone = orig_data.get('location')
                    state = orig_data.get('state') or orig_data.get('state_name')
                    timestamp = meta.get('timestamp', 0)

                    times_buf.append(timestamp)
                    zones_buf.append(zone)
                    old_zones_buf.append(old_zone)
                    states_buf.append(state)
                    centers_buf.append(center)
                    heads_buf.append(head)

                    # Build corrected record
                    out_rec = orig_data.copy()
                    out_rec['center'] = [cx, cy, 1.0] if center else []
                    out_rec['location'] = zone
                    out_rec['frames'] = meta.get('frame_idx', 0)
                    if len(x) > 0:
                        out_rec['pose_array'] = [[float(px), float(py), float(pc)]
                                                  for px, py, pc in zip(x, y, conf)]
                    out_file.write(json.dumps(out_rec) + '\n')

            # Generate Excel
            excel_path = None
            outputs = []
            if config.get('use_zones', True) and zones_for_video and times_buf:
                if config.get('generate_excel', True):
                    excel_path = calculate_zone_statistics_full(
                        stem, times_buf, zones_buf, states_buf,
                        config['output_dir'], zones_for_video, centers_buf, heads_buf,
                        config.get('min_dwell_time', 0.2),
                        zone_assigner=zone_assigner,
                        file_meta=info.get('meta'),
                        angle_threshold=config.get('angle_threshold', 45.0),
                        ia_confirm_ms=config.get('ia_confirm_ms', 100.0),
                        ia_line_extend_px=config.get('ia_line_extend_px', 15.0),
                        ia_inner_offset_px=config.get('ia_inner_offset_px', 50.0),
                        ia_parallel_tol_deg=config.get('ia_parallel_tol_deg', 15.0),
                        params=config
                    )
                    if excel_path:
                        outputs.append(f"Excel: {os.path.basename(excel_path)}")

                if config.get('save_timeline_plot', True):
                    timeline_path = save_zone_timeline_plot(
                        stem, times_buf, old_zones_buf, zones_buf, states_buf,
                        config['output_dir'], zones_for_video
                    )
                    if timeline_path:
                        outputs.append("Timeline plot")

                if config.get('save_distance_plot', True):
                    dist_path = save_distance_plot(
                        stem, times_buf, centers_buf, states_buf,
                        config['output_dir'], config.get('speed_smooth_sec', 1.0)
                    )
                    if dist_path:
                        outputs.append("Distance plot")

            msg = f"Corrected {len(corrected_results)} frames"
            if outputs:
                msg += f" ({', '.join(outputs)})"
            return True, msg, excel_path

        # Output paths
        out_txt = os.path.join(config['output_dir'], base.replace('.txt', '_retracked.txt'))
        out_video = os.path.join(config['output_dir'], stem + '_retracked.avi')  # Annotated video
        out_video_state = os.path.join(config['output_dir'], stem + '.avi')  # State-only video (original name)

        if not config.get('overwrite') and os.path.exists(out_txt):
            return True, "Skipped (exists)", None

        # Validate video path
        if not video_path or not os.path.exists(video_path):
            return False, f"Video not found: {video_path}", None

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Cannot open video: {video_path}", None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if w == 0 or h == 0:
            return False, f"Invalid video dimensions: {w}x{h}", None

        # Setup zones (use zones_full which is already shifted)
        # NOTE: _IA zones already exist in the data file, no need to generate imaginary ones
        zones_full = info.get('zones_full') or info.get('zones', {})
        zones_orig = info.get('zones', {})
        video_is_cropped = (w == crop.w and h == crop.h)
        zones_for_video = zones_orig if video_is_cropped and zones_orig else zones_full

        zone_assigner = FastZoneAssigner(zones_for_video) if zones_for_video else None
        zone_layer = precompute_zone_layer(w, h, zones_for_video) if zones_for_video else None

        # Setup pipeline components
        frame_buffer = FrameBuffer(READER_BUFFER_SIZE, (h, w, 3))
        inference_queue = queue.Queue(maxsize=32)

        # Start threads
        txt_iter = iter_original_frames(txt_path)
        reader = FrameReaderThread(video_path, frame_buffer, txt_iter,
                                   config.get('skip_frames', 1))

        dlc_thread = None
        if config.get('processing_mode') == 'dlc_retrack':
            dlc_thread = DLCInferenceThread(
                frame_buffer, inference_queue,
                config['model_path'], crop.as_tuple(),
                config.get('resize_factor', 0.4),
                config.get('gpu_id', '0')
            )

        # Video writer for annotated output (with timeline remux)
        writer_thread = None
        if config.get('write_video'):
            writer_thread = VideoWriterThread(out_video, fps, (w, h), use_timeline_remux=True)
            writer_thread.start()

        # Video writer for state-only output (with timeline remux)
        state_writer_thread = None
        if config.get('write_video'):
            state_writer_thread = VideoWriterThread(out_video_state, fps, (w, h), use_timeline_remux=True)
            state_writer_thread.start()

        correction_mgr = CorrectionManager(config)
        correction_mgr.start()

        reader.start()
        if dlc_thread:
            dlc_thread.start()

        self.signals.log.emit(f"  {base}: Started processing ({total_frames} frames, {w}x{h})")

        # Processing loop
        out_file = open(out_txt, 'w', encoding='utf-8', buffering=65536)
        out_file.write(f'B {json.dumps(info.get("meta"))}\n')
        out_file.write(f'M {json.dumps(zones_full)}\n')
        out_file.write(f'V {json.dumps(V)}\n')

        frame_count = 0
        frames_read = 0  # Track how many frames we receive from reader
        results_buffer = []

        # Buffers for statistics/plots
        times_buf = []
        zones_buf = []
        old_zones_buf = []
        states_buf = []
        centers_buf = []
        heads_buf = []

        # Pre-compute IA outer edge lines for facing detection
        ia_zone_centroids = {}
        ia_edge_lines = {}
        ia_inner_lines = {}
        ia_line_normals = {}
        if zones_for_video:
            zroot = zones_root(zones_for_video)
            arena_center = get_arena_center(zroot)
            for zn in zroot.keys():
                if '_IA' in zn:
                    ia_zone_centroids[zn] = resolve_ia_target_centroid(zroot, zn)
                    zdata = zroot.get(zn, {})
                    ia_offset = get_zone_ia_inner_offset(zdata, config.get('ia_inner_offset_px', 50.0))
                    ia_extend = get_zone_ia_line_extend(zdata, config.get('ia_line_extend_px', 15.0))
                    outer_line, inner_line, normal = build_ia_reference_lines(
                        zroot[zn].get('values', []),
                        arena_center,
                        line_extend_px=ia_extend,
                        inner_offset_px=ia_offset,
                        parallel_tol_deg=config.get('ia_parallel_tol_deg', 15.0)
                    )
                    ia_edge_lines[zn] = outer_line
                    ia_inner_lines[zn] = inner_line
                    ia_line_normals[zn] = normal

        # Context for _process_corrected_frame
        process_ctx = {
            'zone_assigner': zone_assigner,
            'zone_layer': zone_layer,
            'out_file': out_file,
            'writer_thread': writer_thread,
            'state_writer_thread': state_writer_thread,  # For state-only video
            'config': config,
            'crop': crop,
            'times_buf': times_buf,
            'zones_buf': zones_buf,
            'old_zones_buf': old_zones_buf,
            'states_buf': states_buf,
            'centers_buf': centers_buf,
            'heads_buf': heads_buf,
            # IA zone tracking
            'ia_zone_centroids': ia_zone_centroids,
            'ia_edge_lines': ia_edge_lines,
            'ia_inner_lines': ia_inner_lines,
            'ia_line_normals': ia_line_normals,
            'ia_zone_entries': {},     # {zone_name: count} - body enters zone
            'ia_zone_duration': {},    # {zone_name: seconds} - time body in zone
            'ia_angle_entries': {},    # {zone_name: count} - enter+facing OR turn back to face
            'ia_angle_duration': {},   # {zone_name: seconds} - time facing (while IN zone)
            'prev_zone': None,         # For entry tracking
            'prev_facing': {},         # {zone_name: bool} - was facing last frame
            'prev_in_zone': {},        # {zone_name: bool} - was in zone last frame
            'turned_away': {},         # {zone_name: bool} - turned away while in zone (for turn-back counting)
            'turn_back_start_time': {},# {zone_name: timestamp_ms} - debounce for turn-back
            'facing_start_time': {},   # {zone_name: timestamp_ms} - facing confirmation
            'ia_in_zone_start': {},    # {zone_name: timestamp_ms} - IA entry debounce
            'facing_debug': {},        # Debug info for overlay
            'current_state': None,     # Reset counters on state change
            'prev_timestamp': None,    # Raw timestamp
            'prev_ts_adj': None,       # Adjusted timestamp (ms) for unwrapped time
            'ts_offset': 0.0,          # Offset for timestamp reset
            'last_pos_dt_ms': 0.0,     # Last positive dt in ms
            'trajectory_pts': [],      # For trajectory/path drawing
        }

        try:
            while self.is_running:
                # Get inference result
                if dlc_thread:
                    try:
                        result = inference_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                else:
                    # No DLC mode - read directly
                    buf_result = frame_buffer.get(timeout=1.0)
                    if buf_result is None:
                        continue
                    frame, meta = buf_result
                    # Check for end signal from reader
                    if meta.get('end') or meta.get('error'):
                        break
                    pose = extract_pose_array(meta.get('orig_data', {}), crop=crop, frame_w=w, frame_h=h)
                    result = {'frame': frame, 'meta': meta, 'pose': pose}

                if result.get('end') or result.get('error') or result.get('meta', {}).get('end'):
                    break

                frames_read += 1
                frame = result['frame']
                meta = result['meta']
                pose = result['pose']
                timestamp = meta.get('timestamp', 0)

                # Add to correction batch (include frame for synchronization!)
                corrected = correction_mgr.add_frame(pose, timestamp, meta, frame)

                if corrected:
                    results_buffer.extend(corrected)

                # Process completed corrections
                while results_buffer:
                    res = results_buffer.pop(0)
                    # Use the frame stored with the result, not the current frame!
                    res_frame = res.get('frame')
                    if res_frame is None:
                        res_frame = np.zeros((h, w, 3), dtype=np.uint8)
                    self._process_corrected_frame(res, res_frame, process_ctx)
                    frame_count += 1

                # Progress update
                if frame_count % 100 == 0:
                    pct = int(frame_count / max(1, total_frames) * 100)
                    self.signals.log.emit(f"  {base}: {frame_count} frames ({pct}%)")

        finally:
            # Flush remaining
            try:
                remaining = correction_mgr.flush()
                if remaining:
                    for res in remaining:
                        res_frame = res.get('frame')
                        if res_frame is None:
                            res_frame = np.zeros((h, w, 3), dtype=np.uint8)
                        self._process_corrected_frame(res, res_frame, process_ctx)
                        frame_count += 1
            except Exception as e:
                self.signals.log.emit(f"  {base}: Flush error: {e}")

            out_file.close()

            reader.stop()
            if dlc_thread:
                dlc_thread.stop()
            if writer_thread:
                writer_thread.stop()
            if state_writer_thread:
                state_writer_thread.stop()

            correction_mgr.stop()

            reader.join(timeout=2)
            if dlc_thread:
                dlc_thread.join(timeout=2)
            if writer_thread:
                writer_thread.join(timeout=2)
            if state_writer_thread:
                state_writer_thread.join(timeout=2)

            # Debug: show how many frames the reader produced
            self.signals.log.emit(f"  {base}: Reader produced {reader.frame_count} frames")

        # Generate statistics and plots
        outputs = []
        excel_path = None
        if config.get('use_zones', True) and zones_for_video and times_buf:
            if config.get('generate_excel', True):
                excel_path = calculate_zone_statistics_full(
                    stem, times_buf, zones_buf, states_buf,
                    config['output_dir'], zones_for_video, centers_buf, heads_buf,
                    config.get('min_dwell_time', 0.2),
                    zone_assigner=zone_assigner,
                    file_meta=info.get('meta'),
                    angle_threshold=config.get('angle_threshold', 45.0),
                    ia_confirm_ms=config.get('ia_confirm_ms', 100.0),
                    ia_line_extend_px=config.get('ia_line_extend_px', 15.0),
                    ia_inner_offset_px=config.get('ia_inner_offset_px', 50.0),
                    ia_parallel_tol_deg=config.get('ia_parallel_tol_deg', 15.0),
                    params=config
                )
                if excel_path:
                    outputs.append(f"Excel: {os.path.basename(excel_path)}")

            if config.get('save_timeline_plot', True):
                timeline_path = save_zone_timeline_plot(
                    stem, times_buf, old_zones_buf, zones_buf, states_buf,
                    config['output_dir'], zones_for_video
                )
                if timeline_path:
                    outputs.append("Timeline plot")

            if config.get('save_distance_plot', True):
                dist_path = save_distance_plot(
                    stem, times_buf, centers_buf, states_buf,
                    config['output_dir'], config.get('speed_smooth_sec', 1.0)
                )
                if dist_path:
                    outputs.append("Distance plot")

        msg = f"Processed {frame_count} frames (read: {frames_read})"
        if outputs:
            msg += f" ({', '.join(outputs)})"
        return True, msg, excel_path

    def _process_corrected_frame(self, corrected: Dict, frame: np.ndarray, ctx: Dict):
        """Process a corrected frame - zone assignment, writing, collecting data."""
        x = corrected['x']
        y = corrected['y']
        conf = corrected['conf']
        meta = corrected['meta']

        zone_assigner = ctx['zone_assigner']
        zone_layer = ctx['zone_layer']
        out_file = ctx['out_file']
        writer_thread = ctx['writer_thread']
        config = ctx['config']

        # Keypoint indices
        head_idx = config.get('head_idx', 0)
        body_idx = config.get('body_idx', 1)
        pcut = config.get('pcut', 0.1)

        # Filter by confidence
        for i in range(len(x)):
            if len(conf) > i and conf[i] < pcut:
                x[i] = np.nan
                y[i] = np.nan

        # Calculate center (body position)
        if config.get('infer_center_from_all', True):
            valid = ~(np.isnan(x) | np.isnan(y))
            if np.any(valid):
                cx = float(np.mean(x[valid]))
                cy = float(np.mean(y[valid]))
                center = (cx, cy)
            else:
                center = None
                cx, cy = np.nan, np.nan
        else:
            # Use body keypoint
            if len(x) > body_idx and not np.isnan(x[body_idx]) and not np.isnan(y[body_idx]):
                cx, cy = float(x[body_idx]), float(y[body_idx])
                center = (cx, cy)
            else:
                center = None
                cx, cy = np.nan, np.nan

        # Head position
        if len(x) > head_idx and not np.isnan(x[head_idx]) and not np.isnan(y[head_idx]):
            head = (float(x[head_idx]), float(y[head_idx]))
        else:
            head = None

        # Shift pose from crop to full-frame coordinates to match zone polygons
        # (only when zones are known to be in full-frame coords, i.e. crop_to_full flag)
        crop_box = ctx.get('crop')
        if crop_box and ctx.get('crop_to_full'):
            dx_crop, dy_crop = float(crop_box.x1), float(crop_box.y1)
            if center:
                cx = cx + dx_crop
                cy = cy + dy_crop
                center = (cx, cy)
            if head:
                head = (head[0] + dx_crop, head[1] + dy_crop)

        # Zone assignment
        zone = zone_assigner.locate(cx, cy) if (zone_assigner and center) else None
        head_zone = zone_assigner.locate(head[0], head[1]) if (zone_assigner and head) else None

        # Get original zone and state from orig_data
        orig_data = meta.get('orig_data', {})
        old_zone = orig_data.get('location')
        state = orig_data.get('state') or orig_data.get('state_name')
        timestamp = meta.get('timestamp', meta.get('timestamps', meta.get('ts', meta.get('time', meta.get('t', 0)))))

        # Collect data for statistics
        ctx['times_buf'].append(timestamp)
        ctx['zones_buf'].append(zone)
        ctx['old_zones_buf'].append(old_zone)
        ctx['states_buf'].append(state)
        ctx['centers_buf'].append(center)
        ctx['heads_buf'].append(head)

        # Build record
        rec = orig_data.copy()
        rec['center'] = [cx, cy, 1.0] if center else []
        rec['location'] = zone
        rec['frames'] = meta.get('frame_idx', 0)
        if len(x) > 0:
            rec['pose_array'] = [[float(px), float(py), float(pc)]
                                 for px, py, pc in zip(x, y, conf)]

        # Write record
        out_file.write(json.dumps(rec) + '\n')

        # Track IA zone entries, duration, and facing
        ia_zone_centroids = ctx.get('ia_zone_centroids', {})
        ia_edge_lines = ctx.get('ia_edge_lines', {})
        ia_inner_lines = ctx.get('ia_inner_lines', {})
        ia_line_normals = ctx.get('ia_line_normals', {})
        ia_zone_entries = ctx.get('ia_zone_entries', {})
        ia_zone_duration = ctx.get('ia_zone_duration', {})
        ia_angle_entries = ctx.get('ia_angle_entries', {})
        ia_angle_duration = ctx.get('ia_angle_duration', {})
        prev_zone = ctx.get('prev_zone')
        prev_facing = ctx.get('prev_facing', {})
        prev_in_zone = ctx.get('prev_in_zone', {})
        current_state = ctx.get('current_state')
        prev_timestamp = ctx.get('prev_timestamp')
        prev_ts_adj = ctx.get('prev_ts_adj')
        ts_offset = ctx.get('ts_offset', 0.0)
        last_pos_dt_ms = ctx.get('last_pos_dt_ms', 0.0)
        turn_back_start_time = ctx.get('turn_back_start_time', {})
        angle_threshold = config.get('angle_threshold', 45.0)
        ia_confirm_ms = max(config.get('ia_confirm_ms', 100.0), 0.0)
        ia_in_zone_start = ctx.get('ia_in_zone_start', {})

        # Calculate dt (time since last frame) in seconds using unwrapped timestamps
        try:
            raw_ts = float(timestamp)
        except Exception:
            raw_ts = 0.0

        dt = 0.0
        if prev_timestamp is not None:
            try:
                raw_dt = raw_ts - float(prev_timestamp)
            except Exception:
                raw_dt = 0.0
            if raw_dt > 0:
                last_pos_dt_ms = raw_dt
        if prev_ts_adj is None:
            adj_ts = raw_ts
        else:
            if raw_ts + ts_offset < prev_ts_adj:
                ts_offset += (prev_ts_adj - (raw_ts + ts_offset)) + (last_pos_dt_ms if last_pos_dt_ms > 0 else 0.0)
            adj_ts = raw_ts + ts_offset
            dt = (adj_ts - prev_ts_adj) / 1000.0
            if dt < 0:
                dt = 0.0

        ctx['prev_timestamp'] = raw_ts
        ctx['prev_ts_adj'] = adj_ts
        ctx['ts_offset'] = ts_offset
        ctx['last_pos_dt_ms'] = last_pos_dt_ms

        # Get turned_away state
        turned_away = ctx.get('turned_away', {})
        facing_start_time = ctx.get('facing_start_time', {})
        facing_debug = {}

        # Reset counters on state change
        if state != current_state:
            ia_zone_entries = {}
            ia_zone_duration = {}
            ia_angle_entries = {}
            ia_angle_duration = {}
            prev_facing = {}
            prev_in_zone = {}
            turned_away = {}
            turn_back_start_time = {}
            facing_start_time = {}
            facing_debug = {}
            ia_in_zone_start = {}
            ctx['ia_zone_entries'] = ia_zone_entries
            ctx['ia_zone_duration'] = ia_zone_duration
            ctx['ia_angle_entries'] = ia_angle_entries
            ctx['ia_angle_duration'] = ia_angle_duration
            ctx['prev_facing'] = prev_facing
            ctx['prev_in_zone'] = prev_in_zone
            ctx['turned_away'] = turned_away
            ctx['turn_back_start_time'] = turn_back_start_time
            ctx['facing_start_time'] = facing_start_time
            ctx['facing_debug'] = facing_debug
            ctx['ia_in_zone_start'] = ia_in_zone_start
            ctx['current_state'] = state

        # Track zone entries and duration for ALL _IA zones
        is_facing_any_ia = False
        for ia_zone, zc in ia_zone_centroids.items():
            # Use HEAD-in-zone for IA entry/dwell; head/body define facing vector.
            is_in_zone_raw = (head_zone == ia_zone)
            inner_line = ia_inner_lines.get(ia_zone)
            line_normal = ia_line_normals.get(ia_zone)
            crossed = True
            if is_in_zone_raw and inner_line and line_normal and head:
                crossed = point_is_past_line((head[0], head[1]), inner_line[0], line_normal, toward_center=False)
            is_in_zone = is_in_zone_raw and crossed

            start_time = ia_in_zone_start.get(ia_zone)
            if is_in_zone:
                if start_time is None:
                    start_time = adj_ts
                    ia_in_zone_start[ia_zone] = start_time
                in_zone_confirmed = (adj_ts - start_time) >= ia_confirm_ms
            else:
                ia_in_zone_start[ia_zone] = None
                in_zone_confirmed = False

            was_in_zone = prev_in_zone.get(ia_zone, False)

            # Zone ENTRY: transition from NOT in zone to IN zone
            if in_zone_confirmed and not was_in_zone:
                ia_zone_entries[ia_zone] = ia_zone_entries.get(ia_zone, 0) + 1
                # Reset turned_away when entering zone
                turned_away[ia_zone] = False

            # Zone EXIT: reset turned_away state
            if not in_zone_confirmed and was_in_zone:
                turned_away[ia_zone] = False
                turn_back_start_time[ia_zone] = None
                facing_start_time[ia_zone] = None

            # Zone DURATION: accumulate time while in zone
            if in_zone_confirmed and dt > 0:
                ia_zone_duration[ia_zone] = ia_zone_duration.get(ia_zone, 0.0) + dt

            # Update prev_in_zone for next frame
            prev_in_zone[ia_zone] = in_zone_confirmed

            # Check if facing this _IA zone
            edge_line = ia_edge_lines.get(ia_zone)
            is_facing = False
            ang_edge = None
            edge_pt = None
            if center and head and edge_line and in_zone_confirmed:
                ang_edge, edge_pt = compute_facing_angle_to_edge(head, center, edge_line[0], edge_line[1])
                if ang_edge is not None and ang_edge <= angle_threshold:
                    is_facing = True

            if in_zone_confirmed:
                facing_debug = {
                    'zone': ia_zone,
                    'angle_edge': ang_edge,
                    'edge_line': edge_line,
                    'inner_line': inner_line,
                    'edge_point': edge_pt,
                    'centroid': zc,
                    'is_facing': is_facing
                }

            facing_confirmed = False
            if is_facing:
                if facing_start_time.get(ia_zone) is None:
                    facing_start_time[ia_zone] = adj_ts
                elif adj_ts - facing_start_time[ia_zone] >= ia_confirm_ms:
                    facing_confirmed = True
            else:
                facing_start_time[ia_zone] = None

            was_facing = prev_facing.get(ia_zone, False)

            # Angle ENTRY counting: confirmed facing transition while IN zone
            if in_zone_confirmed and facing_confirmed and not was_facing:
                ia_angle_entries[ia_zone] = ia_angle_entries.get(ia_zone, 0) + 1

            # Angle DURATION: accumulate time while IN zone AND facing
            if in_zone_confirmed and is_facing and dt > 0:
                ia_angle_duration[ia_zone] = ia_angle_duration.get(ia_zone, 0.0) + dt

            # Update prev_facing for next frame
            prev_facing[ia_zone] = facing_confirmed

            # Track if facing any IA zone (for interaction indicator)
            if in_zone_confirmed and is_facing:
                is_facing_any_ia = True

        ctx['prev_zone'] = zone
        ctx['prev_facing'] = prev_facing
        ctx['prev_in_zone'] = prev_in_zone
        ctx['turned_away'] = turned_away
        ctx['turn_back_start_time'] = turn_back_start_time
        ctx['facing_start_time'] = facing_start_time
        ctx['facing_debug'] = facing_debug
        ctx['ia_in_zone_start'] = ia_in_zone_start
        ctx['ia_zone_entries'] = ia_zone_entries
        ctx['ia_zone_duration'] = ia_zone_duration
        ctx['ia_angle_entries'] = ia_angle_entries
        ctx['ia_angle_duration'] = ia_angle_duration

        # Video overlay
        if writer_thread and frame.size > 0:
            overlay = frame.copy()
            h, w = frame.shape[:2]

            # Apply zone layer
            if zone_layer is not None:
                alpha = config.get('zone_alpha', 0.22)
                cv2.addWeighted(zone_layer, alpha, overlay, 1-alpha, 0, overlay)

            # Keypoint indices
            head_idx = config.get('head_idx', 0)
            tail_idx = 2  # Typically tail is index 2

            # Draw keypoints with special shapes for head/tail
            palette = [(150, 50, 255), (1, 190, 200), (255, 128, 0), (238, 130, 238),
                       (0, 165, 255), (180, 130, 70), (50, 205, 154), (180, 105, 255)]
            marker_r = config.get('marker_radius_px', 4)

            for i in range(len(x)):
                if not np.isnan(x[i]) and not np.isnan(y[i]):
                    ix, iy = int(x[i]), int(y[i])
                    if i == head_idx:
                        cv2.circle(overlay, (ix, iy), marker_r + 2, (0, 0, 255), -1)
                        cv2.circle(overlay, (ix, iy), marker_r + 2, (0, 0, 139), 1)
                    elif i == tail_idx:
                        cv2.circle(overlay, (ix, iy), marker_r + 2, (0, 255, 0), -1)
                        cv2.circle(overlay, (ix, iy), marker_r + 2, (0, 100, 0), 1)
                    else:
                        color = palette[i % len(palette)]
                        cv2.circle(overlay, (ix, iy), marker_r, color, -1)

            # Draw trajectory (path history)
            trajectory_pts = ctx.get('trajectory_pts', [])
            if center:
                trajectory_pts.append((int(cx), int(cy)))
                # Keep last 50 points
                if len(trajectory_pts) > 50:
                    trajectory_pts = trajectory_pts[-50:]
                ctx['trajectory_pts'] = trajectory_pts

            if len(trajectory_pts) > 1:
                # Draw trajectory as fading line
                for i in range(1, len(trajectory_pts)):
                    alpha = int(255 * (i / len(trajectory_pts)))
                    color = (alpha, alpha // 2, 0)  # Fading orange
                    cv2.line(overlay, trajectory_pts[i-1], trajectory_pts[i], color, 1, cv2.LINE_AA)

            # Draw body center as yellow circle
            if center:
                bcx, bcy = int(cx), int(cy)
                cv2.circle(overlay, (bcx, bcy), marker_r + 2, (0, 200, 255), -1)
                cv2.circle(overlay, (bcx, bcy), marker_r + 2, (0, 100, 200), 1)

                # Draw INTERACTION CIRCLE when facing any IA zone
                if is_facing_any_ia:
                    cv2.circle(overlay, (bcx, bcy), 40, (0, 255, 0), 3)  # Green circle
                    cv2.putText(overlay, "INT", (bcx - 15, bcy - 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw heading with view cone and directional head triangle
            if center and head:
                hx, hy = int(head[0]), int(head[1])
                dx = head[0] - cx
                dy = head[1] - cy
                norm = math.sqrt(dx*dx + dy*dy)
                if norm > 1e-6:
                    ux, uy = dx / norm, dy / norm
                    body_dir_rad = math.atan2(dy, dx)

                    # Draw VIEW CONE (45 degree half-angle = 90 degree total viewing cone)
                    cone_length = 50
                    cone_half_angle = math.radians(45)

                    cone_left_x = hx + int(cone_length * math.cos(body_dir_rad - cone_half_angle))
                    cone_left_y = hy + int(cone_length * math.sin(body_dir_rad - cone_half_angle))
                    cone_right_x = hx + int(cone_length * math.cos(body_dir_rad + cone_half_angle))
                    cone_right_y = hy + int(cone_length * math.sin(body_dir_rad + cone_half_angle))

                    # Draw filled cone (yellow-orange)
                    cone_pts = np.array([[hx, hy], [cone_left_x, cone_left_y], [cone_right_x, cone_right_y]], dtype=np.int32)
                    cone_overlay = overlay.copy()
                    cv2.fillPoly(cone_overlay, [cone_pts], (0, 200, 255))  # Yellow-orange fill
                    cv2.addWeighted(cone_overlay, 0.3, overlay, 0.7, 0, overlay)
                    cv2.polylines(overlay, [cone_pts], True, (0, 150, 200), 1)  # Edge

                    # Draw heading arrow
                    arrow_len = 28
                    end_x = int(hx + ux * arrow_len)
                    end_y = int(hy + uy * arrow_len)
                    cv2.arrowedLine(overlay, (hx, hy), (end_x, end_y),
                                   (150, 50, 255), 1, cv2.LINE_AA, tipLength=0.25)

            # Facing debug overlay (shows what the code uses)
            facing_debug = ctx.get('facing_debug', {})
            debug_y = y_offset + 18 if 'y_offset' in locals() else 40
            if facing_debug and head:
                hx, hy = int(head[0]), int(head[1])
                edge_line = facing_debug.get('edge_line')
                inner_line = facing_debug.get('inner_line')
                edge_pt = facing_debug.get('edge_point')
                ang_e = facing_debug.get('angle_edge')
                face_flag = facing_debug.get('is_facing')
                if edge_line and all(np.isfinite([edge_line[0][0], edge_line[0][1], edge_line[1][0], edge_line[1][1]])):
                    p1 = (int(edge_line[0][0]), int(edge_line[0][1]))
                    p2 = (int(edge_line[1][0]), int(edge_line[1][1]))
                    cv2.line(overlay, p1, p2, (0, 255, 255), 3)
                if inner_line and all(np.isfinite([inner_line[0][0], inner_line[0][1], inner_line[1][0], inner_line[1][1]])):
                    p1i = (int(inner_line[0][0]), int(inner_line[0][1]))
                    p2i = (int(inner_line[1][0]), int(inner_line[1][1]))
                    cv2.line(overlay, p1i, p2i, (0, 200, 255), 2)
                if edge_pt and all(np.isfinite([edge_pt[0], edge_pt[1]])):
                    cv2.line(overlay, (hx, hy), (int(edge_pt[0]), int(edge_pt[1])), (0, 255, 255), 1)
                debug_text = f"FaceE:{ang_e:.1f} Face:{int(bool(face_flag))} Conf:{int(bool(ctx.get('prev_facing', {}).get(facing_debug.get('zone'), False)))}" if ang_e is not None else "Face:N/A"
                cv2.putText(overlay, debug_text, (10, debug_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, debug_text, (10, debug_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw state text (top right)
            if state:
                (sw, sh), _ = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(overlay, state, (w - sw - 10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw IA zone stats (top left) - Zone entries and Angle entries with duration
            y_offset = 25
            for zone_name in ia_zone_centroids.keys():
                if '_IA' not in zone_name:
                    continue

                # Zone Entry: count(duration)
                z_count = ia_zone_entries.get(zone_name, 0)
                z_dur = ia_zone_duration.get(zone_name, 0.0)
                if z_count > 0 or z_dur > 0:
                    zone_text = f"{zone_name}: {z_count}({z_dur:.1f}s)"
                    cv2.putText(overlay, zone_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(overlay, zone_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)  # Cyan
                    y_offset += 18

                # Angle Entry: count(duration) - only when IN zone AND facing
                a_count = ia_angle_entries.get(zone_name, 0)
                a_dur = ia_angle_duration.get(zone_name, 0.0)
                if a_count > 0 or a_dur > 0:
                    angle_text = f"{zone_name} Angle: {a_count}({a_dur:.1f}s)"
                    cv2.putText(overlay, angle_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(overlay, angle_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)  # Green
                    y_offset += 18

            # Draw zone text (bottom right)
            zone_text = zone if zone else 'None'
            (zw, zh), _ = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            zx = max(10, w - zw - 10)
            zy = max(zh + 10, h - 10)
            cv2.putText(overlay, zone_text, (zx, zy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, zone_text, (zx, zy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # Write annotated video (with timestamp for timeline remux)
            writer_thread.write(overlay, timestamp)

        # Write state-only video (clean frame with only state name)
        state_writer_thread = ctx.get('state_writer_thread')
        if state_writer_thread and frame.size > 0:
            h, w = frame.shape[:2]
            state_frame = frame.copy()

            # Only draw state name - nothing else
            if state:
                # Large state text in center of frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                (tw, th), _ = cv2.getTextSize(state, font, font_scale, thickness)
                tx = (w - tw) // 2
                ty = 50  # Near top

                # Draw with shadow for visibility
                cv2.putText(state_frame, state, (tx + 2, ty + 2), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(state_frame, state, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Write with timestamp for timeline remux
            state_writer_thread.write(state_frame, timestamp)


# =============================================================================
# UI COMPONENTS
# =============================================================================

class ConfigPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Mode
        mode_grp = QGroupBox("Processing Mode")
        mode_lay = QVBoxLayout(mode_grp)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["DLC Retrack", "Correct Only", "Analysis Only"])
        mode_lay.addWidget(self.mode_combo)
        layout.addWidget(mode_grp)

        # Folders
        folders_grp = QGroupBox("Folders")
        folders_lay = QGridLayout(folders_grp)

        self.txt_folder = QLineEdit("./video_data")
        self.videos_folder = QLineEdit("./videos")
        self.output_folder = QLineEdit("./output")
        self.model_path = QLineEdit()

        for i, (label, widget) in enumerate([
            ("TXT Data:", self.txt_folder),
            ("Videos:", self.videos_folder),
            ("Output:", self.output_folder),
            ("DLC Model:", self.model_path)
        ]):
            folders_lay.addWidget(QLabel(label), i, 0)
            folders_lay.addWidget(widget, i, 1)
            btn = QPushButton("...")
            btn.setMaximumWidth(30)
            btn.clicked.connect(lambda _, w=widget: self._browse(w))
            folders_lay.addWidget(btn, i, 2)

        layout.addWidget(folders_grp)

        # GPU/Performance
        perf_grp = QGroupBox("Performance")
        perf_lay = QGridLayout(perf_grp)

        self.gpu_id = QLineEdit("0")
        self.resize_factor = QDoubleSpinBox()
        self.resize_factor.setRange(0.1, 1.0)
        self.resize_factor.setValue(1.0)
        self.skip_frames = QSpinBox()
        self.skip_frames.setRange(1, 10)
        self.skip_frames.setValue(1)
        self.num_workers = QSpinBox()
        self.num_workers.setRange(1, 16)
        self.num_workers.setValue(NUM_CORRECTION_WORKERS)

        perf_lay.addWidget(QLabel("GPU ID:"), 0, 0)
        perf_lay.addWidget(self.gpu_id, 0, 1)
        perf_lay.addWidget(QLabel("Resize:"), 1, 0)
        perf_lay.addWidget(self.resize_factor, 1, 1)
        perf_lay.addWidget(QLabel("Skip Frames:"), 2, 0)
        perf_lay.addWidget(self.skip_frames, 2, 1)
        perf_lay.addWidget(QLabel("Workers:"), 3, 0)
        perf_lay.addWidget(self.num_workers, 3, 1)

        layout.addWidget(perf_grp)

        # Tracking
        track_grp = QGroupBox("Tracking")
        track_lay = QGridLayout(track_grp)

        self.pcut = QDoubleSpinBox()
        self.pcut.setRange(0, 1)
        self.pcut.setValue(0.1)
        self.infer_center = QCheckBox("Infer center from all KPs")
        self.infer_center.setChecked(True)

        self.head_idx = QSpinBox()
        self.head_idx.setRange(0, 20)
        self.head_idx.setValue(0)
        self.head_idx.setToolTip("Keypoint index for head (used for angle calculation)")

        self.body_idx = QSpinBox()
        self.body_idx.setRange(0, 20)
        self.body_idx.setValue(1)
        self.body_idx.setToolTip("Keypoint index for body center (used for swap resolver)")

        track_lay.addWidget(QLabel("Likelihood Cutoff:"), 0, 0)
        track_lay.addWidget(self.pcut, 0, 1)
        track_lay.addWidget(self.infer_center, 1, 0, 1, 2)
        track_lay.addWidget(QLabel("Head KP Index:"), 2, 0)
        track_lay.addWidget(self.head_idx, 2, 1)
        track_lay.addWidget(QLabel("Body KP Index:"), 3, 0)
        track_lay.addWidget(self.body_idx, 3, 1)

        layout.addWidget(track_grp)

        # Correction
        corr_grp = QGroupBox("Correction (Batch Processed)")
        corr_lay = QGridLayout(corr_grp)

        self.swap_resolver = QCheckBox("Swap Resolver")
        self.swap_resolver.setChecked(True)
        self.rolling_median = QCheckBox("Rolling Median")
        self.rolling_median.setChecked(True)
        self.speed_correction = QCheckBox("Speed-based Correction")
        self.speed_correction.setChecked(True)

        self.max_speed = QSpinBox()
        self.max_speed.setRange(500, 10000)
        self.max_speed.setValue(3000)

        self.swap_threshold = QDoubleSpinBox()
        self.swap_threshold.setRange(0.1, 0.9)
        self.swap_threshold.setValue(0.3)
        self.swap_threshold.setSingleStep(0.05)
        self.swap_threshold.setToolTip("Cost improvement threshold to trigger swap (0.3 = 30%)")

        corr_lay.addWidget(self.swap_resolver, 0, 0, 1, 2)
        corr_lay.addWidget(QLabel("Swap Threshold:"), 1, 0)
        corr_lay.addWidget(self.swap_threshold, 1, 1)
        corr_lay.addWidget(self.rolling_median, 2, 0, 1, 2)
        corr_lay.addWidget(self.speed_correction, 3, 0, 1, 2)
        corr_lay.addWidget(QLabel("Max Speed (px/s):"), 4, 0)
        corr_lay.addWidget(self.max_speed, 4, 1)

        layout.addWidget(corr_grp)

        # Output
        out_grp = QGroupBox("Output")
        out_lay = QVBoxLayout(out_grp)

        self.write_video = QCheckBox("Write Video")
        self.write_video.setChecked(True)
        self.generate_excel = QCheckBox("Generate Excel")
        self.generate_excel.setChecked(True)
        self.save_timeline = QCheckBox("Save Timeline Plot")
        self.save_timeline.setChecked(True)
        self.save_distance = QCheckBox("Save Distance Plot")
        self.save_distance.setChecked(True)
        self.overwrite = QCheckBox("Overwrite Existing")
        self.overwrite.setChecked(True)

        self.zone_alpha = QDoubleSpinBox()
        self.zone_alpha.setRange(0, 0.6)
        self.zone_alpha.setValue(0.22)
        self.marker_radius = QSpinBox()
        self.marker_radius.setRange(1, 20)
        self.marker_radius.setValue(4)

        out_lay.addWidget(self.write_video)
        out_lay.addWidget(self.generate_excel)
        out_lay.addWidget(self.save_timeline)
        out_lay.addWidget(self.save_distance)
        out_lay.addWidget(self.overwrite)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Zone Alpha:"))
        h2.addWidget(self.zone_alpha)
        h2.addWidget(QLabel("Marker:"))
        h2.addWidget(self.marker_radius)
        out_lay.addLayout(h2)

        layout.addWidget(out_grp)

        # Zone Analysis Options
        zone_grp = QGroupBox("Zone Analysis Options")
        zone_lay = QGridLayout(zone_grp)

        self.use_zones = QCheckBox("Enable Zone Analysis")
        self.use_zones.setChecked(True)
        zone_lay.addWidget(self.use_zones, 0, 0, 1, 2)

        zone_lay.addWidget(QLabel("Min Dwell Time (s):"), 1, 0)
        self.min_dwell = QDoubleSpinBox()
        self.min_dwell.setRange(0, 2.0)
        self.min_dwell.setValue(0.2)
        self.min_dwell.setSingleStep(0.05)
        self.min_dwell.setToolTip("Minimum time in zone for valid entry")
        zone_lay.addWidget(self.min_dwell, 1, 1)

        zone_lay.addWidget(QLabel("IA Zone Depth (px):"), 2, 0)
        self.ia_zone_depth = QSpinBox()
        self.ia_zone_depth.setRange(0, 200)
        self.ia_zone_depth.setValue(70)
        self.ia_zone_depth.setToolTip("Depth of imaginary IA zones - full-width strips at frame edges (0=disable)")
        zone_lay.addWidget(self.ia_zone_depth, 2, 1)

        zone_lay.addWidget(QLabel("IA Line Extend (px):"), 3, 0)
        self.ia_line_extend = QSpinBox()
        self.ia_line_extend.setRange(0, 200)
        self.ia_line_extend.setValue(15)
        self.ia_line_extend.setToolTip("Extend IA outer line for facing calc")
        zone_lay.addWidget(self.ia_line_extend, 3, 1)

        zone_lay.addWidget(QLabel("IA Inner Offset (px):"), 4, 0)
        self.ia_inner_offset = QSpinBox()
        self.ia_inner_offset.setRange(0, 200)
        self.ia_inner_offset.setValue(50)
        self.ia_inner_offset.setToolTip("Offset of inner IA line toward center")
        zone_lay.addWidget(self.ia_inner_offset, 4, 1)

        zone_lay.addWidget(QLabel("IA Confirm (ms):"), 5, 0)
        self.ia_confirm_ms = QSpinBox()
        self.ia_confirm_ms.setRange(0, 1000)
        self.ia_confirm_ms.setValue(100)
        self.ia_confirm_ms.setToolTip("Min time head must stay past inner line to confirm IA")
        zone_lay.addWidget(self.ia_confirm_ms, 5, 1)

        zone_lay.addWidget(QLabel("Angle Threshold (deg):"), 6, 0)
        self.angle_threshold = QDoubleSpinBox()
        self.angle_threshold.setRange(10, 120)
        self.angle_threshold.setValue(60)
        self.angle_threshold.setToolTip("Angle threshold for 'facing zone' detection")
        zone_lay.addWidget(self.angle_threshold, 6, 1)

        zone_lay.addWidget(QLabel("Speed Smooth (s):"), 7, 0)
        self.speed_smooth = QDoubleSpinBox()
        self.speed_smooth.setRange(0, 5)
        self.speed_smooth.setValue(1.0)
        zone_lay.addWidget(self.speed_smooth, 7, 1)

        layout.addWidget(zone_grp)

        # One-Euro Filter Options
        euro_grp = QGroupBox("One-Euro Filter")
        euro_lay = QGridLayout(euro_grp)

        self.one_euro = QCheckBox("Enable One-Euro Filter")
        self.one_euro.setChecked(True)
        self.one_euro.setToolTip("Smooth positions with minimal lag")
        euro_lay.addWidget(self.one_euro, 0, 0, 1, 2)

        euro_lay.addWidget(QLabel("Min Cutoff:"), 1, 0)
        self.euro_min_cutoff = QDoubleSpinBox()
        self.euro_min_cutoff.setRange(0.1, 10)
        self.euro_min_cutoff.setValue(1.0)
        euro_lay.addWidget(self.euro_min_cutoff, 1, 1)

        euro_lay.addWidget(QLabel("Beta:"), 2, 0)
        self.euro_beta = QDoubleSpinBox()
        self.euro_beta.setRange(0, 1)
        self.euro_beta.setValue(0.007)
        self.euro_beta.setDecimals(4)
        euro_lay.addWidget(self.euro_beta, 2, 1)

        layout.addWidget(euro_grp)

        layout.addStretch()

    def _browse(self, widget):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            widget.setText(path)

    def get_config(self) -> Dict:
        modes = ['dlc_retrack', 'correct_only', 'analysis_only']
        cfg = {
            'processing_mode': modes[self.mode_combo.currentIndex()],
            'video_data_dir': self.txt_folder.text(),
            'videos_dir': self.videos_folder.text(),
            'output_dir': self.output_folder.text(),
            'model_path': self.model_path.text(),
            'gpu_id': self.gpu_id.text(),
            'resize_factor': self.resize_factor.value(),
            'skip_frames': self.skip_frames.value(),
            'num_workers': self.num_workers.value(),
            'pcut': self.pcut.value(),
            'infer_center_from_all': self.infer_center.isChecked(),
            'head_idx': self.head_idx.value(),
            'body_idx': self.body_idx.value(),
            'enable_swap_resolver': self.swap_resolver.isChecked(),
            'swap_cost_threshold': self.swap_threshold.value(),
            'use_rolling_median': self.rolling_median.isChecked(),
            'max_speed_px_sec': self.max_speed.value() if self.speed_correction.isChecked() else 0,
            'write_video': self.write_video.isChecked(),
            'generate_excel': self.generate_excel.isChecked(),
            'save_timeline_plot': self.save_timeline.isChecked(),
            'save_distance_plot': self.save_distance.isChecked(),
            'overwrite': self.overwrite.isChecked(),
            'zone_alpha': self.zone_alpha.value(),
            'marker_radius_px': self.marker_radius.value(),
            # Zone Analysis Options
            'use_zones': self.use_zones.isChecked(),
            'min_dwell_time': self.min_dwell.value(),
            'ia_zone_depth': self.ia_zone_depth.value(),
            'ia_line_extend_px': self.ia_line_extend.value(),
            'ia_inner_offset_px': self.ia_inner_offset.value(),
            'ia_confirm_ms': self.ia_confirm_ms.value(),
            'ia_parallel_tol_deg': 15.0,
            'angle_threshold': self.angle_threshold.value(),
            'speed_smooth_sec': self.speed_smooth.value(),
            # One-Euro Filter Options
            'use_one_euro': self.one_euro.isChecked(),
            'euro_min_cutoff': self.euro_min_cutoff.value(),
            'euro_beta': self.euro_beta.value(),
        }
        auto = guess_head_body_indices(discover_bodyparts(cfg.get('model_path', '')))
        if auto:
            cfg['head_idx'], cfg['body_idx'] = auto
        return cfg


class FileListWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.file_info = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.select_all_btn = QPushButton("Select All")
        self.deselect_btn = QPushButton("Deselect All")
        toolbar.addWidget(self.refresh_btn)
        toolbar.addWidget(self.select_all_btn)
        toolbar.addWidget(self.deselect_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self.status = QLabel("0 files")
        layout.addWidget(self.status)

        self.select_all_btn.clicked.connect(self._select_all)
        self.deselect_btn.clicked.connect(self._deselect_all)

    def load_files(self, txt_dir: str, videos_dir: str, processing_mode: str = 'dlc_retrack'):
        self.list_widget.clear()
        self.file_info.clear()

        # In correct_only or analysis_only mode, video is not required
        video_optional = processing_mode in ('correct_only', 'analysis_only')

        # Normalize paths
        txt_dir = os.path.normpath(txt_dir) if txt_dir else ""
        videos_dir = os.path.normpath(videos_dir) if videos_dir else ""

        # Check if directories exist
        if not txt_dir or not os.path.isdir(txt_dir):
            self.status.setText(f"ERROR: TXT folder not found: {txt_dir}")
            return

        # Find all txt files
        txt_files = sorted(glob.glob(os.path.join(txt_dir, '*.txt')))

        if not txt_files:
            self.status.setText(f"No .txt files found in: {txt_dir}")
            return

        loaded = 0
        errors = 0
        no_video = 0

        for txt_path in txt_files:
            base = os.path.basename(txt_path)
            try:
                stem, _ = parse_video_name_from_txt(txt_path)
                meta, zones, V = parse_headers(txt_path)
                video_path = find_video_path(videos_dir, stem)

                # Create crop box from V
                V = V if V else [0, 100, 0, 100]
                crop = crop_from_V(V)

                # Shift zones to full frame coordinates if needed
                zones_full = zones
                if zones and zones_are_cropped(zones, crop):
                    zones_full = shift_zones_to_full(zones, crop)

                self.file_info[base] = {
                    'txt_path': txt_path,
                    'stem': stem,
                    'meta': meta,
                    'zones': zones,           # Original zones
                    'zones_full': zones_full, # Shifted to full frame
                    'V': V,
                    'crop': crop,
                    'video_path': video_path
                }

                # Create item with status indicator
                if video_path:
                    item = QListWidgetItem(f"✓ {base}")
                    loaded += 1
                else:
                    item = QListWidgetItem(f"⚠ {base} (no video)")
                    no_video += 1
                    loaded += 1

                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                # Auto-check files without video when video is not required (correct_only/analysis_only)
                item.setCheckState(Qt.CheckState.Checked if (video_path or video_optional) else Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, base)  # Store original name
                self.list_widget.addItem(item)

            except Exception as e:
                self.file_info[base] = {'error': str(e), 'txt_path': txt_path}
                item = QListWidgetItem(f"✗ {base} ({str(e)[:30]})")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, base)
                self.list_widget.addItem(item)
                errors += 1

        status_parts = [f"{loaded} files"]
        if no_video > 0:
            status_parts.append(f"{no_video} without video")
        if errors > 0:
            status_parts.append(f"{errors} errors")
        self.status.setText(" | ".join(status_parts))

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Checked)

    def _deselect_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Unchecked)

    def get_selected(self) -> Dict:
        selected = {}
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                # Get original base name from UserRole
                base = item.data(Qt.ItemDataRole.UserRole)
                if not base:
                    # Fallback: extract from text
                    text = item.text()
                    if text.startswith(('✓ ', '⚠ ', '✗ ')):
                        base = text[2:].split(' (')[0]
                    else:
                        base = text

                if base in self.file_info and 'error' not in self.file_info[base]:
                    selected[base] = self.file_info[base]
        return selected


class ZoneEditorWidget(QWidget):
    """Zone editor for viewing and modifying zones"""
    zones_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.current_file = None
        self.zones_data = {}
        self.video_frame = None
        self.scale_factor = 1.0
        self.crop = None  # CropBox for showing crop region
        self.file_list = []  # List of (base, info) tuples
        self.current_file_idx = -1  # Current file index
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # Left: Zone list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)

        # File selector with Next/Prev buttons
        file_grp = QGroupBox("Select File")
        file_lay = QVBoxLayout(file_grp)

        # Navigation buttons row
        nav_lay = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self._prev_file)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._next_file)
        nav_lay.addWidget(self.prev_btn)
        nav_lay.addWidget(self.next_btn)
        file_lay.addLayout(nav_lay)

        # Current file label
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #2a2a2a;")
        file_lay.addWidget(self.file_label)

        # File counter label
        self.file_counter = QLabel("0 / 0")
        self.file_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_lay.addWidget(self.file_counter)

        left_layout.addWidget(file_grp)

        # Zone list
        zones_grp = QGroupBox("Zones")
        zones_lay = QVBoxLayout(zones_grp)
        self.zone_list = QListWidget()
        self.zone_list.currentItemChanged.connect(self._on_zone_selected)
        zones_lay.addWidget(self.zone_list)

        # Zone buttons
        btn_lay = QHBoxLayout()
        self.add_zone_btn = QPushButton("Add")
        self.add_zone_btn.clicked.connect(self._add_zone)
        self.delete_zone_btn = QPushButton("Delete")
        self.delete_zone_btn.clicked.connect(self._delete_zone)
        self.rename_zone_btn = QPushButton("Rename")
        self.rename_zone_btn.clicked.connect(self._rename_zone)
        btn_lay.addWidget(self.add_zone_btn)
        btn_lay.addWidget(self.delete_zone_btn)
        btn_lay.addWidget(self.rename_zone_btn)
        zones_lay.addLayout(btn_lay)
        left_layout.addWidget(zones_grp)

        # IA controls
        ia_grp = QGroupBox("IA Params (per zone)")
        ia_lay = QGridLayout(ia_grp)
        ia_lay.addWidget(QLabel("Default IA depth (px):"), 0, 0)
        self.ia_default_offset = QSpinBox()
        self.ia_default_offset.setRange(0, 200)
        self.ia_default_offset.setValue(50)
        self.ia_default_offset.setToolTip("Used when an IA zone has no saved depth")
        self.ia_default_offset.valueChanged.connect(self._on_default_ia_offset_changed)
        ia_lay.addWidget(self.ia_default_offset, 0, 1)

        ia_lay.addWidget(QLabel("Default IA extend (px):"), 1, 0)
        self.ia_default_extend = QSpinBox()
        self.ia_default_extend.setRange(0, 200)
        self.ia_default_extend.setValue(15)
        self.ia_default_extend.setToolTip("Used when an IA zone has no saved extend")
        self.ia_default_extend.valueChanged.connect(self._on_default_ia_extend_changed)
        ia_lay.addWidget(self.ia_default_extend, 1, 1)

        ia_lay.addWidget(QLabel("Selected IA depth (px):"), 2, 0)
        self.ia_zone_offset = QSpinBox()
        self.ia_zone_offset.setRange(0, 200)
        self.ia_zone_offset.setValue(50)
        self.ia_zone_offset.setEnabled(False)
        ia_lay.addWidget(self.ia_zone_offset, 2, 1)

        ia_lay.addWidget(QLabel("Selected IA extend (px):"), 3, 0)
        self.ia_zone_extend = QSpinBox()
        self.ia_zone_extend.setRange(0, 200)
        self.ia_zone_extend.setValue(15)
        self.ia_zone_extend.setEnabled(False)
        ia_lay.addWidget(self.ia_zone_extend, 3, 1)

        self.apply_ia_zone_btn = QPushButton("Apply to selected IA")
        self.apply_ia_zone_btn.setEnabled(False)
        self.apply_ia_zone_btn.clicked.connect(self._apply_ia_params_to_selected)
        ia_lay.addWidget(self.apply_ia_zone_btn, 4, 0, 1, 2)

        left_layout.addWidget(ia_grp)

        # Zone Transform Controls
        transform_grp = QGroupBox("Zone Transform")
        transform_lay = QVBoxLayout(transform_grp)

        # Target selection (All zones or selected zone)
        target_lay = QHBoxLayout()
        self.transform_all = QCheckBox("All Zones")
        self.transform_all.setChecked(True)
        target_lay.addWidget(self.transform_all)
        transform_lay.addLayout(target_lay)

        # Move step
        step_lay = QHBoxLayout()
        step_lay.addWidget(QLabel("Move Step:"))
        self.move_step = QSpinBox()
        self.move_step.setRange(1, 50)
        self.move_step.setValue(5)
        step_lay.addWidget(self.move_step)
        step_lay.addWidget(QLabel("px"))
        transform_lay.addLayout(step_lay)

        # Movement buttons
        move_lay = QGridLayout()
        self.move_up_btn = QPushButton("↑ Up")
        self.move_up_btn.clicked.connect(lambda: self._move_zones(0, -1))
        self.move_down_btn = QPushButton("↓ Down")
        self.move_down_btn.clicked.connect(lambda: self._move_zones(0, 1))
        self.move_left_btn = QPushButton("← Left")
        self.move_left_btn.clicked.connect(lambda: self._move_zones(-1, 0))
        self.move_right_btn = QPushButton("→ Right")
        self.move_right_btn.clicked.connect(lambda: self._move_zones(1, 0))
        move_lay.addWidget(self.move_up_btn, 0, 1)
        move_lay.addWidget(self.move_left_btn, 1, 0)
        move_lay.addWidget(self.move_right_btn, 1, 2)
        move_lay.addWidget(self.move_down_btn, 2, 1)
        transform_lay.addLayout(move_lay)

        # Rotate step
        rot_lay = QHBoxLayout()
        rot_lay.addWidget(QLabel("Rotate Step:"))
        self.rotate_step = QSpinBox()
        self.rotate_step.setRange(1, 45)
        self.rotate_step.setValue(5)
        rot_lay.addWidget(self.rotate_step)
        rot_lay.addWidget(QLabel("°"))
        transform_lay.addLayout(rot_lay)

        # Rotation buttons
        rot_btn_lay = QHBoxLayout()
        self.rotate_ccw_btn = QPushButton("⟲ CCW")
        self.rotate_ccw_btn.clicked.connect(lambda: self._rotate_zones(1))
        self.rotate_cw_btn = QPushButton("⟳ CW")
        self.rotate_cw_btn.clicked.connect(lambda: self._rotate_zones(-1))
        self.reset_zones_btn = QPushButton("Reset")
        self.reset_zones_btn.clicked.connect(self._reset_zones)
        rot_btn_lay.addWidget(self.rotate_ccw_btn)
        rot_btn_lay.addWidget(self.rotate_cw_btn)
        rot_btn_lay.addWidget(self.reset_zones_btn)
        transform_lay.addLayout(rot_btn_lay)

        left_layout.addWidget(transform_grp)

        # Zone points editor
        points_grp = QGroupBox("Zone Points")
        points_lay = QVBoxLayout(points_grp)
        self.points_text = QTextEdit()
        self.points_text.setMaximumHeight(150)
        self.points_text.setPlaceholderText("[[x1,y1], [x2,y2], ...]")
        points_lay.addWidget(self.points_text)

        update_btn = QPushButton("Update Points")
        update_btn.clicked.connect(self._update_points)
        points_lay.addWidget(update_btn)
        left_layout.addWidget(points_grp)

        # Save button
        self.save_btn = QPushButton("Save Zones to File")
        self.save_btn.clicked.connect(self._save_zones)
        left_layout.addWidget(self.save_btn)

        left_layout.addStretch()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setMaximumWidth(320)
        layout.addWidget(left_scroll)

        # Right: Preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        preview_grp = QGroupBox("Zone Preview")
        preview_lay = QVBoxLayout(preview_grp)
        self.preview_label = QLabel("Load a file to preview zones")
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")

        scroll = QScrollArea()
        scroll.setWidget(self.preview_label)
        scroll.setWidgetResizable(True)
        preview_lay.addWidget(scroll)
        right_layout.addWidget(preview_grp)

        layout.addWidget(right_panel, stretch=1)

    def load_files(self, file_info: Dict):
        """Load files into file list for navigation"""
        self.file_list = []
        self.current_file_idx = -1

        for base, info in file_info.items():
            if 'error' not in info:
                self.file_list.append((base, info))

        # Update UI
        if self.file_list:
            self.current_file_idx = 0
            self._load_current_file()
        else:
            self.file_label.setText("No valid files")
            self.file_counter.setText("0 / 0")

        self._update_nav_buttons()

    def _prev_file(self):
        """Go to previous file"""
        if self.current_file_idx > 0:
            self.current_file_idx -= 1
            self._load_current_file()
        self._update_nav_buttons()

    def _next_file(self):
        """Go to next file"""
        if self.current_file_idx < len(self.file_list) - 1:
            self.current_file_idx += 1
            self._load_current_file()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        """Enable/disable navigation buttons based on current position"""
        self.prev_btn.setEnabled(self.current_file_idx > 0)
        self.next_btn.setEnabled(self.current_file_idx < len(self.file_list) - 1)
        self.file_counter.setText(f"{self.current_file_idx + 1} / {len(self.file_list)}" if self.file_list else "0 / 0")

    def _load_current_file(self):
        """Load the current file based on current_file_idx"""
        if self.current_file_idx < 0 or self.current_file_idx >= len(self.file_list):
            return

        base, info = self.file_list[self.current_file_idx]
        self.file_label.setText(base)
        self._on_file_selected(info)

    def _on_file_selected(self, info):
        """Handle file selection - internal method"""
        if not info:
            return

        self.current_file = info
        # Use zones_full (already shifted to full frame coords)
        self.zones_data = copy.deepcopy(info.get('zones_full') or info.get('zones', {}))
        self.crop = info.get('crop')
        self._ensure_ia_params(save_if_missing=True)

        # Load first frame of video for preview
        video_path = info.get('video_path')
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.video_frame = frame
                self._update_preview()

        self._refresh_zone_list()

    def _refresh_zone_list(self):
        """Refresh the zone list"""
        self.zone_list.clear()
        zroot = zones_root(self.zones_data) if self.zones_data else {}

        for name in zroot.keys():
            self.zone_list.addItem(name)

    def _on_zone_selected(self, current, previous):
        """Handle zone selection"""
        if not current:
            return

        zone_name = current.text()
        zroot = zones_root(self.zones_data)
        if zone_name in zroot:
            pts = zroot[zone_name].get('values', [])
            self.points_text.setText(json.dumps(pts, indent=2))

        is_ia = '_IA' in zone_name
        self.ia_zone_offset.setEnabled(is_ia)
        self.ia_zone_extend.setEnabled(is_ia)
        self.apply_ia_zone_btn.setEnabled(is_ia)
        if is_ia and zone_name in zroot:
            self.ia_zone_offset.setValue(int(get_zone_ia_inner_offset(
                zroot[zone_name], self.ia_default_offset.value()
            )))
            self.ia_zone_extend.setValue(int(get_zone_ia_line_extend(
                zroot[zone_name], self.ia_default_extend.value()
            )))

        self._update_preview(highlight_zone=zone_name)

    def _add_zone(self):
        """Add a new zone"""
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Add Zone", "Zone name:")
        if ok and name:
            zroot = zones_root(self.zones_data)
            if 'zones' not in self.zones_data:
                self.zones_data['zones'] = {}
            self.zones_data['zones'][name] = {
                'values': [[100, 100], [200, 100], [200, 200], [100, 200]],
                'color': [255, 0, 0]
            }
            self._ensure_ia_params(save_if_missing=False)
            self._refresh_zone_list()
            self._update_preview()

    def _delete_zone(self):
        """Delete selected zone"""
        current = self.zone_list.currentItem()
        if not current:
            return

        zone_name = current.text()
        zroot = zones_root(self.zones_data)
        if zone_name in zroot:
            del zroot[zone_name]
            self._refresh_zone_list()
            self._update_preview()

    def _rename_zone(self):
        """Rename selected zone"""
        from PyQt6.QtWidgets import QInputDialog
        current = self.zone_list.currentItem()
        if not current:
            return

        old_name = current.text()
        new_name, ok = QInputDialog.getText(self, "Rename Zone", "New name:", text=old_name)
        if ok and new_name and new_name != old_name:
            zroot = zones_root(self.zones_data)
            if old_name in zroot:
                zroot[new_name] = zroot.pop(old_name)
                self._ensure_ia_params(save_if_missing=False)
                self._refresh_zone_list()
                self._update_preview()

    def _update_points(self):
        """Update zone points from text"""
        current = self.zone_list.currentItem()
        if not current:
            return

        zone_name = current.text()
        try:
            pts = json.loads(self.points_text.toPlainText())
            zroot = zones_root(self.zones_data)
            if zone_name in zroot:
                zroot[zone_name]['values'] = pts
                self._update_preview(highlight_zone=zone_name)
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Error", f"Invalid JSON: {e}")

    def _on_default_ia_offset_changed(self):
        """Apply default IA depth to zones missing a saved value."""
        self._ensure_ia_params(save_if_missing=True)

    def _on_default_ia_extend_changed(self):
        """Apply default IA extend to zones missing a saved value."""
        self._ensure_ia_params(save_if_missing=True)

    def _apply_ia_params_to_selected(self):
        """Save IA params for the selected IA zone."""
        current = self.zone_list.currentItem()
        if not current:
            return
        zone_name = current.text()
        if '_IA' not in zone_name:
            return
        zroot = zones_root(self.zones_data)
        if zone_name in zroot:
            zroot[zone_name]['ia_inner_offset_px'] = int(self.ia_zone_offset.value())
            zroot[zone_name]['ia_line_extend_px'] = int(self.ia_zone_extend.value())
            self._update_preview(highlight_zone=zone_name)

    def _ensure_ia_params(self, save_if_missing: bool = False):
        """Ensure each IA zone has saved inner offset and line extend values."""
        if not self.zones_data:
            return
        zroot = zones_root(self.zones_data)
        if not zroot:
            return
        default_val = int(self.ia_default_offset.value())
        default_extend = int(self.ia_default_extend.value())
        changed = False
        for name, z in zroot.items():
            if '_IA' in name:
                if 'ia_inner_offset_px' not in z:
                    z['ia_inner_offset_px'] = default_val
                    changed = True
                if 'ia_line_extend_px' not in z:
                    z['ia_line_extend_px'] = default_extend
                    changed = True
        if changed:
            self._update_preview()
            if save_if_missing:
                self._save_zones(silent=True)

    def _move_zones(self, dx_dir: int, dy_dir: int):
        """Move zones in the specified direction"""
        if not self.zones_data:
            return

        step = self.move_step.value()
        dx = dx_dir * step
        dy = dy_dir * step

        zroot = zones_root(self.zones_data)
        apply_all = self.transform_all.isChecked()

        # Get target zones
        if apply_all:
            target_zones = list(zroot.keys())
        else:
            current = self.zone_list.currentItem()
            if current:
                target_zones = [current.text()]
            else:
                return

        # Apply translation
        for zone_name in target_zones:
            if zone_name in zroot:
                pts = zroot[zone_name].get('values', [])
                new_pts = [[p[0] + dx, p[1] + dy] for p in pts]
                zroot[zone_name]['values'] = new_pts

        self._update_preview()

    def _rotate_zones(self, direction: int):
        """Rotate zones around their collective center"""
        if not self.zones_data:
            return

        angle_deg = self.rotate_step.value() * direction
        angle_rad = math.radians(angle_deg)

        zroot = zones_root(self.zones_data)
        apply_all = self.transform_all.isChecked()

        # Get target zones
        if apply_all:
            target_zones = list(zroot.keys())
        else:
            current = self.zone_list.currentItem()
            if current:
                target_zones = [current.text()]
            else:
                return

        # Find center of all target zones
        all_pts = []
        for zone_name in target_zones:
            if zone_name in zroot:
                pts = zroot[zone_name].get('values', [])
                all_pts.extend(pts)

        if not all_pts:
            return

        # Calculate centroid
        cx = sum(p[0] for p in all_pts) / len(all_pts)
        cy = sum(p[1] for p in all_pts) / len(all_pts)

        # Apply rotation around centroid
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        for zone_name in target_zones:
            if zone_name in zroot:
                pts = zroot[zone_name].get('values', [])
                new_pts = []
                for p in pts:
                    x, y = p[0] - cx, p[1] - cy  # Translate to origin
                    rx = x * cos_a - y * sin_a   # Rotate
                    ry = x * sin_a + y * cos_a
                    new_pts.append([rx + cx, ry + cy])  # Translate back
                zroot[zone_name]['values'] = new_pts

        self._update_preview()

    def _reset_zones(self):
        """Reset zones to original from file"""
        if not self.current_file:
            return

        # Reload zones from file
        self.zones_data = copy.deepcopy(self.current_file.get('zones_full') or self.current_file.get('zones', {}))
        self._refresh_zone_list()
        self._update_preview()

    def _update_preview(self, highlight_zone: str = None):
        """Update the preview image"""
        if self.video_frame is None:
            return

        frame = self.video_frame.copy()
        h, w = frame.shape[:2]

        # Draw crop box first (white rectangle)
        if self.crop:
            cv2.rectangle(frame, (self.crop.x1, self.crop.y1),
                         (self.crop.x2, self.crop.y2), (255, 255, 255), 2)
            # Label
            cv2.putText(frame, "CROP", (self.crop.x1 + 5, self.crop.y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw all zones
        zroot = zones_root(self.zones_data) if self.zones_data else {}
        palette = [(150, 50, 255), (1, 190, 200), (255, 128, 0), (238, 130, 238),
                   (0, 165, 255), (180, 130, 70), (50, 205, 154), (180, 105, 255)]

        for i, (name, z) in enumerate(zroot.items()):
            pts = z.get('values', [])
            if len(pts) >= 3:
                try:
                    poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    poly = np.ascontiguousarray(poly)

                    color = palette[i % len(palette)]
                    thickness = 3 if name == highlight_zone else 1

                    # Fill with transparency
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [poly], color)
                    alpha = 0.4 if name == highlight_zone else 0.2
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # Draw outline
                    cv2.polylines(frame, [poly], True, color, thickness)

                    # Draw zone name
                    centroid = np.mean(pts, axis=0).astype(int)
                    cv2.putText(frame, name, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (255, 255, 255), 2)
                except Exception as e:
                    print(f"Error drawing zone {name}: {e}")

        # Draw IA outer/inner reference lines for visual validation
        if zroot:
            arena_center = get_arena_center(zroot)
            for name, z in zroot.items():
                if '_IA' not in name:
                    continue
                pts = z.get('values', [])
                ia_offset = get_zone_ia_inner_offset(z, self.ia_default_offset.value())
                ia_extend = get_zone_ia_line_extend(z, self.ia_default_extend.value())
                outer_line, inner_line, _ = build_ia_reference_lines(
                    pts, arena_center, line_extend_px=ia_extend, inner_offset_px=ia_offset
                )
                if outer_line:
                    p1 = (int(outer_line[0][0]), int(outer_line[0][1]))
                    p2 = (int(outer_line[1][0]), int(outer_line[1][1]))
                    cv2.line(frame, p1, p2, (0, 255, 255), 2)
                if inner_line:
                    p1i = (int(inner_line[0][0]), int(inner_line[0][1]))
                    p2i = (int(inner_line[1][0]), int(inner_line[1][1]))
                    cv2.line(frame, p1i, p2i, (0, 200, 255), 2)

        # Scale for display
        max_w, max_h = 800, 600
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Convert to QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        qimg = QImage(rgb.data, w2, h2, ch * w2, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pixmap)

    def _save_zones(self, silent: bool = False):
        """Save zones back to the txt file"""
        if not self.current_file:
            QMessageBox.warning(self, "Error", "No file selected")
            return

        txt_path = self.current_file.get('txt_path')
        if not txt_path:
            return

        try:
            # Read original file
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Update M line with new zones
            new_lines = []
            for line in lines:
                if line.strip().startswith('M '):
                    new_lines.append(f'M {json.dumps(self.zones_data)}\n')
                else:
                    new_lines.append(line)

            # Write back
            with open(txt_path, 'w', encoding='utf-8') as f:
                if isinstance(self.zones_data, dict):
                    self.zones_data['coords'] = 'full'
                f.writelines(new_lines)

            # Update the file_info
            self.current_file['zones'] = copy.deepcopy(self.zones_data)
            self.current_file['zones_full'] = copy.deepcopy(self.zones_data)
            self.zones_changed.emit(self.zones_data)

            if not silent:
                QMessageBox.information(self, "Success", f"Zones saved to {os.path.basename(txt_path)}")

        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Retracking Pipeline - High Performance")
        self.setMinimumSize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: Config
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(350)
        self.config = ConfigPanel()
        scroll.setWidget(self.config)
        layout.addWidget(scroll)

        # Right: Tabs
        tabs = QTabWidget()

        # Files tab
        files_widget = QWidget()
        files_layout = QVBoxLayout(files_widget)
        self.file_list = FileListWidget()
        self.file_list.refresh_btn.clicked.connect(self._load_files)
        files_layout.addWidget(self.file_list)

        load_btn = QPushButton("Load Files")
        load_btn.clicked.connect(self._load_files)
        files_layout.addWidget(load_btn)
        tabs.addTab(files_widget, "Files")

        # Zone Editor tab
        self.zone_editor = ZoneEditorWidget()
        self.zone_editor.zones_changed.connect(self._on_zones_changed)
        tabs.addTab(self.zone_editor, "Zone Editor")

        # Processing tab
        proc_widget = QWidget()
        proc_layout = QVBoxLayout(proc_widget)

        self.progress = QProgressBar()
        proc_layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        proc_layout.addWidget(self.log)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("START")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        proc_layout.addLayout(btn_layout)

        tabs.addTab(proc_widget, "Processing")
        layout.addWidget(tabs)

    def _load_files(self):
        config = self.config.get_config()
        os.makedirs(config['output_dir'], exist_ok=True)

        txt_dir = config['video_data_dir']
        vid_dir = config['videos_dir']

        self._log(f"Loading from TXT: {txt_dir}")
        self._log(f"Looking for videos in: {vid_dir}")

        self.file_list.load_files(txt_dir, vid_dir, config.get('processing_mode', 'dlc_retrack'))

        # Also populate Zone Editor
        self.zone_editor.load_files(self.file_list.file_info)

        # Show result
        status = self.file_list.status.text()
        self._log(f"Result: {status}")

    def _log(self, msg: str):
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _on_zones_changed(self, zones_data: Dict):
        """Sync updated zones into file list info for processing."""
        current = self.zone_editor.current_file
        if not current:
            return
        txt_path = current.get('txt_path')
        if not txt_path:
            return
        for base, info in self.file_list.file_info.items():
            if info.get('txt_path') == txt_path:
                info['zones'] = copy.deepcopy(zones_data)
                info['zones_full'] = copy.deepcopy(zones_data)
                break

    def _start(self):
        config = self.config.get_config()
        selected = self.file_list.get_selected()

        if not selected:
            QMessageBox.warning(self, "Warning", "No files selected")
            return

        self.pipeline = HighPerformancePipeline(config, selected)
        self.pipeline.signals.progress.connect(lambda v, m: (self.progress.setValue(v), self._log(m)))
        self.pipeline.signals.file_started.connect(lambda f: self._log(f"Processing: {f}"))
        self.pipeline.signals.file_completed.connect(lambda f, s, m: self._log(f"  {f}: {'OK' if s else 'FAIL'} - {m}"))
        self.pipeline.signals.log.connect(self._log)
        self.pipeline.signals.finished.connect(self._on_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._log("Starting pipeline...")
        self.pipeline.start()

    def _stop(self):
        if self.pipeline:
            self.pipeline.stop()
            self._log("Stopping...")

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._log("Pipeline finished")


def main():
    mp.set_start_method('spawn', force=True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
