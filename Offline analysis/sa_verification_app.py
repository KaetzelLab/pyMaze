#!/usr/bin/env python3
"""
Post-hoc Spontaneous Alternation Verification App

Select a TXT folder + Video folder. Auto-discovers all txt files, matches
videos, scores alternation during object_habituation and object_novelty
using an IN_ARM/IN_CENTRE state machine.

Interactive video verification with sequential frame reading (no seek lag).
Export all mice to a single cumulative Excel — one row per mouse.
"""

import os
import re
import sys
import json
import glob
import time
import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from collections import defaultdict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QSplitter, QGroupBox, QComboBox, QSlider, QHeaderView, QMessageBox,
    QTextEdit, QAbstractItemView, QListWidget, QListWidgetItem, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QBrush

# ═════════════════════════════════════════════════════════════════════════════
# ZONE DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

IA_ZONES = {"Object1_IA", "Object2_IA"}

# lst_center_arm — any of these zones means mouse has exited the arm.
# Uses actual zone names from tracking data.
CENTRE_ZONES = {
    "Social_Buffer", "Social_Open_Corner", "Familiar_Social_Corner",
    "Open_Buffer", "Open_Novel_Corner",
    "Novel_Buffer", "Novel_Object1_Corner",
    "Object1_Buffer", "Object1_Object2_Corner",
    "Object2_Buffer", "Object2_Familiar_Corner",
    "Familiar_Buffer",
    "CenterZone",
}

TARGET_STATES = {"State: object_habituation", "State: object_novelty"}

ZONE_COLORS = {
    "Object1_IA": (0, 200, 255),
    "Object2_IA": (255, 100, 0),
    "CenterZone": (200, 200, 200),
}
CENTRE_COLOR = (180, 220, 180)
DEFAULT_ZONE_COLOR = (100, 100, 100)


# ═════════════════════════════════════════════════════════════════════════════
# FILE DISCOVERY  (same logic as v1_2.py)
# ═════════════════════════════════════════════════════════════════════════════

def parse_video_name_from_txt(txt_name: str) -> Tuple[str, str]:
    base = os.path.basename(txt_name)
    name_no_ext = os.path.splitext(base)[0]

    if '_video_data_-' in base:
        parts = base.split('_video_data_-')
        subject = parts[0]
        ts = parts[-1].replace('.txt', '')
        return f'{subject}-{ts}', ts

    if '_video_data-' in base:
        parts = base.split('_video_data-')
        subject = parts[0]
        ts = parts[-1].replace('.txt', '')
        return f'{subject}-{ts}', ts

    if '-' in name_no_ext:
        return name_no_ext, name_no_ext.split('-')[-1]

    return name_no_ext, name_no_ext


def find_video_path(vdir: str, stem: str) -> Optional[str]:
    if not vdir or not os.path.isdir(vdir):
        return None

    for ext in ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'):
        p = os.path.join(vdir, stem + ext)
        if os.path.exists(p):
            return p

    for ext in ('.mp4', '.avi', '.mov', '.mkv'):
        for vf in glob.glob(os.path.join(vdir, f'*{ext}')):
            vf_base = os.path.splitext(os.path.basename(vf))[0]
            if stem in vf_base or vf_base in stem:
                return vf

    if '-' in stem:
        ts_part = stem.split('-')[-1]
        for ext in ('.mp4', '.avi', '.mov', '.mkv'):
            for vf in glob.glob(os.path.join(vdir, f'*{ts_part}*{ext}')):
                return vf

    return None


# ═════════════════════════════════════════════════════════════════════════════
# DATA PARSING
# ═════════════════════════════════════════════════════════════════════════════

def _head_zone(pose, zone_polys):
    """Return zone name where head keypoint (pose[0]) is located, or None."""
    if not pose or len(pose) < 1 or len(pose[0]) < 2:
        return None
    hx, hy = pose[0][0], pose[0][1]
    if not np.isfinite(hx) or not np.isfinite(hy):
        return None
    pt = np.array([hx, hy], dtype=np.float32)
    for zname, poly in zone_polys.items():
        if cv2.pointPolygonTest(poly, (float(hx), float(hy)), False) >= 0:
            return zname
    return None


def parse_txt_file(filepath):
    meta = {}
    zones_raw = {}
    crop_box = None
    frames = []

    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("B "):
                meta = json.loads(s[2:])
            elif s.startswith("M "):
                zones_raw = json.loads(s[2:])
            elif s.startswith("V "):
                try:
                    crop_box = json.loads(s[2:])
                except Exception:
                    crop_box = eval(s[2:])
            elif s.startswith("{"):
                try:
                    rec = json.loads(s)
                    rec["_global_idx"] = len(frames)
                    rec["_video_frame"] = rec.get("frames", len(frames))
                    frames.append(rec)
                except json.JSONDecodeError:
                    continue

    zroot = zones_raw.get("zones", zones_raw)
    zone_polys = {}
    for name, zdata in zroot.items():
        if "values" in zdata:
            zone_polys[name] = np.array(zdata["values"], dtype=np.float32)

    # Compute head zone for each frame using point-in-polygon
    for rec in frames:
        pose = rec.get("pose_array", [])
        rec["_head_zone"] = _head_zone(pose, zone_polys)

    return meta, zone_polys, crop_box, frames


def extract_metadata(meta):
    result = {'mouseID': '', 'group': '', 'subgroup': '',
              'expt_date': '', 'time': '', 'start_time': ''}
    if not meta:
        return result

    def pick(keys):
        for k in keys:
            v = meta.get(k)
            if v:
                return str(v)
        return ''

    result['mouseID']   = pick(['Subject ID', 'SubjectID', 'Mouse ID', 'MouseID', 'ID'])
    result['group']     = pick(['Group'])
    result['subgroup']  = pick(['Sub Group', 'SubGroup'])

    start_time = pick(['Start time', 'Start_time', 'StartTime', 'DateTime', 'Date'])
    result['start_time'] = start_time
    if start_time:
        dm = re.search(r'(\d{4})[/-](\d{2})[/-](\d{2})', start_time)
        tm = re.search(r'(\d{2})[:\-](\d{2})[:\-](\d{2})', start_time)
        if dm:
            result['expt_date'] = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
        if tm:
            result['time'] = f"{tm.group(1)}:{tm.group(2)}:{tm.group(3)}"
    return result


# ═════════════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═════════════════════════════════════════════════════════════════════════════

def run_state_machine(frames, use_head=True):
    state = "IN_CENTRE"
    last_arm = None
    visits = []
    events = []
    phase_ranges = {}
    current_phase = None
    phase_start = None

    for frame in frames:
        state_name = frame.get("state_name") or ""
        if state_name not in TARGET_STATES:
            if current_phase is not None:
                phase_ranges[current_phase] = (phase_start, frame["_global_idx"] - 1)
                current_phase = None
            continue

        phase = state_name.replace("State: ", "")
        gidx = frame["_global_idx"]
        ts = frame.get("timestamps", 0)
        if use_head:
            loc = frame.get("_head_zone") or frame.get("location", "")
        else:
            loc = frame.get("location", "")

        if phase != current_phase:
            if current_phase is not None:
                phase_ranges[current_phase] = (phase_start, gidx - 1)
            current_phase = phase
            phase_start = gidx
            # Reset state machine for each phase — phases are independent
            state = "IN_CENTRE"
            last_arm = None

        if loc in IA_ZONES:
            arm = "Object1" if loc == "Object1_IA" else "Object2"
            if state == "IN_CENTRE":
                score = None
                if last_arm is not None:
                    score = "C" if arm != last_arm else "W"
                visits.append({
                    "arm": arm, "score": score,
                    "global_frame_idx": gidx, "timestamp_ms": ts,
                    "phase": phase, "visit_num": len(visits) + 1,
                })
                events.append({
                    "type": "VISIT", "global_frame_idx": gidx,
                    "arm": arm, "score": score,
                    "visit_num": len(visits), "state_before": "IN_CENTRE",
                })
                last_arm = arm
                state = "IN_ARM"
            elif arm != last_arm:
                # Mouse moved directly from one arm to the other
                # (through Entry zones without touching a centre zone)
                # This is a valid alternation — count it as C
                score = "C"
                visits.append({
                    "arm": arm, "score": score,
                    "global_frame_idx": gidx, "timestamp_ms": ts,
                    "phase": phase, "visit_num": len(visits) + 1,
                })
                events.append({
                    "type": "VISIT", "global_frame_idx": gidx,
                    "arm": arm, "score": score,
                    "visit_num": len(visits), "state_before": "IN_ARM_DIRECT",
                })
                last_arm = arm
                # stays IN_ARM
            else:
                events.append({
                    "type": "IGNORED_IA", "global_frame_idx": gidx,
                    "arm": arm, "reason": "still IN_ARM, same arm",
                })

        elif loc in CENTRE_ZONES:
            if state == "IN_ARM":
                events.append({
                    "type": "CENTRE_RESET", "global_frame_idx": gidx,
                    "zone": loc,
                })
                state = "IN_CENTRE"

    if current_phase is not None and frames:
        phase_ranges[current_phase] = (phase_start, frames[-1]["_global_idx"])

    return visits, phase_ranges, events


# ═════════════════════════════════════════════════════════════════════════════
# EXCEL EXPORT — matches output.xlsx format
# Two rows per mouse (one per State), visit scores in columns 1,2,3...
# ═════════════════════════════════════════════════════════════════════════════

def build_sa_rows(meta, visits):
    """
    Build rows matching output.xlsx format exactly:
      MouseID | Group | ExptDate | StartTime | subgroup | State |
      Correct | Incorrect | percent correct | 1 | 2 | 3 | ...

    Returns list of dicts (two per mouse — one per phase).
    """
    md = extract_metadata(meta)

    phase_visits = defaultdict(list)
    for v in visits:
        phase_visits[v["phase"]].append(v)

    rows = []
    for phase in ["object_habituation", "object_novelty"]:
        pv = phase_visits.get(phase, [])
        scored = [v for v in pv if v["score"] is not None]
        c_count = sum(1 for v in scored if v["score"] == "C")
        w_count = sum(1 for v in scored if v["score"] == "W")
        total_scored = c_count + w_count
        pct = round(c_count / total_scored * 100, 1) if total_scored > 0 else ""

        # Init arm: first visit arm for this phase
        if pv:
            init_arm = "O1" if pv[0]["arm"] == "Object1" else "O2"
        else:
            init_arm = ""

        row = {
            'MouseID':         md['mouseID'],
            'Group':           md['group'],
            'ExptDate':        md['expt_date'],
            'StartTime':       md['time'],
            'subgroup':        md['subgroup'],
            'State':           f"State: {phase}",
            'Correct':         c_count,
            'Incorrect':       w_count,
            'percent correct': pct,
            'Init arm':        init_arm,
        }

        # Visit pattern in columns named 1, 2, 3, ...
        for i, v in enumerate(pv, 1):
            if v["score"] is None:
                arm_short = "O1" if v["arm"] == "Object1" else "O2"
                row[i] = f"[{arm_short}]"
            else:
                row[i] = v["score"]

        rows.append(row)
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# SEQUENTIAL VIDEO READER  (no seek — frame-accurate)
# ═════════════════════════════════════════════════════════════════════════════

class VideoFrameCache:
    """
    Reads video sequentially, caches frames so random access is accurate.
    cv2 seek on compressed video is unreliable (keyframe snapping = 2-5 frame lag).
    Instead, read forward sequentially and cache frames we'll need.
    """

    def __init__(self, video_path, crop_box=None):
        self.cap = cv2.VideoCapture(video_path)
        self.crop_box = crop_box
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15
        self._cache = {}           # frame_idx -> numpy frame
        self._next_read_idx = 0    # next frame cap.read() will return
        self._max_cache = 500      # keep last N frames in memory

    def get_frame(self, idx):
        """Get frame at idx. Returns None if out of range."""
        if idx < 0 or idx >= self.total_frames:
            return None

        # Already cached
        if idx in self._cache:
            return self._apply_crop(self._cache[idx])

        # Need to seek backwards? Reset and re-read
        if idx < self._next_read_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 5))
            self._next_read_idx = max(0, idx - 5)
            # Read past any seek inaccuracy
            while self._next_read_idx < idx:
                ret, frame = self.cap.read()
                if not ret:
                    return None
                self._next_read_idx += 1

        # Read forward to target
        while self._next_read_idx <= idx:
            ret, frame = self.cap.read()
            if not ret:
                return None
            self._cache[self._next_read_idx] = frame
            self._next_read_idx += 1

        # Evict old frames
        if len(self._cache) > self._max_cache:
            keys = sorted(self._cache.keys())
            for k in keys[:len(keys) - self._max_cache]:
                del self._cache[k]

        return self._apply_crop(self._cache.get(idx))

    def _apply_crop(self, frame):
        if frame is None:
            return None
        if self.crop_box and len(self.crop_box) == 4:
            y1, y2, x1, x2 = self.crop_box
            if 0 <= y1 < y2 <= frame.shape[0] and 0 <= x1 < x2 <= frame.shape[1]:
                return frame[y1:y2, x1:x2].copy()
        return frame.copy()

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ═════════════════════════════════════════════════════════════════════════════
# GUI
# ═════════════════════════════════════════════════════════════════════════════

class SAVerificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SA Verification — Object Habituation / Novelty")
        self.setMinimumSize(1400, 850)

        self.txt_dir = ''
        self.vid_dir = ''
        self.out_dir = ''
        self.file_infos = []
        self.current_file_idx = -1

        # Current file data
        self.meta = {}
        self.video_cache = None
        self.frames_data = []
        self.zone_polys = {}
        self.crop_box = None
        self.visits = []
        self.events = []
        self.phase_ranges = {}
        self.current_global_idx = 0
        self._event_by_frame = {}
        self._sm_state_cache = []
        self._filtered_visits = []

        self._all_rows = []  # cumulative export

        self.playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._play_tick)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── Top bar: folders ──────────────────────────────────────────────
        folder_box = QGroupBox("Folders")
        fl = QHBoxLayout(folder_box)

        self.btn_txt_dir = QPushButton("TXT Folder...")
        self.btn_txt_dir.clicked.connect(self._select_txt_dir)
        self.lbl_txt_dir = QLabel("—")
        self.lbl_txt_dir.setStyleSheet("color: grey;")

        self.btn_vid_dir = QPushButton("Video Folder...")
        self.btn_vid_dir.clicked.connect(self._select_vid_dir)
        self.lbl_vid_dir = QLabel("—")
        self.lbl_vid_dir.setStyleSheet("color: grey;")

        self.btn_out_dir = QPushButton("Save Folder...")
        self.btn_out_dir.clicked.connect(self._select_out_dir)
        self.lbl_out_dir = QLabel("—")
        self.lbl_out_dir.setStyleSheet("color: grey;")

        fl.addWidget(self.btn_txt_dir)
        fl.addWidget(self.lbl_txt_dir, 1)
        fl.addWidget(self.btn_vid_dir)
        fl.addWidget(self.lbl_vid_dir, 1)
        fl.addWidget(self.btn_out_dir)
        fl.addWidget(self.lbl_out_dir, 1)
        main_layout.addWidget(folder_box)

        # ── Action buttons ────────────────────────────────────────────────
        action_box = QHBoxLayout()

        self.btn_load = QPushButton("  Load Files  ")
        self.btn_load.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_load.clicked.connect(self._load_files)
        self.btn_load.setEnabled(False)

        self.btn_run_all = QPushButton("  Run All  ")
        self.btn_run_all.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_run_all.clicked.connect(self._run_all)
        self.btn_run_all.setEnabled(False)

        self.btn_export = QPushButton("  Export Excel  ")
        self.btn_export.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_export.clicked.connect(self._export_excel)
        self.btn_export.setEnabled(False)

        self.combo_zone_mode = QComboBox()
        self.combo_zone_mode.addItems(["Head", "Center"])
        self.combo_zone_mode.setToolTip("Zone detection: Head keypoint (pose[0]) or body Center (location field)")
        self.combo_zone_mode.setFixedWidth(90)

        action_box.addWidget(self.btn_load)
        action_box.addWidget(self.btn_run_all)
        action_box.addWidget(self.btn_export)
        action_box.addWidget(QLabel("Zone:"))
        action_box.addWidget(self.combo_zone_mode)
        action_box.addStretch()
        main_layout.addLayout(action_box)

        # ── Main: file list | video | visit panel ─────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: File list
        file_widget = QWidget()
        file_layout = QVBoxLayout(file_widget)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.addWidget(QLabel("Files:"))
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._file_selected)
        file_layout.addWidget(self.file_list)
        self.lbl_file_status = QLabel("")
        self.lbl_file_status.setFont(QFont("Consolas", 8))
        file_layout.addWidget(self.lbl_file_status)
        splitter.addWidget(file_widget)

        # Centre: Video
        video_widget = QWidget()
        vl = QVBoxLayout(video_widget)
        vl.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("Select folders, Load Files, then Run All")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(540, 400)
        self.video_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        vl.addWidget(self.video_label, 1)

        ctrl_layout = QHBoxLayout()
        self.btn_prev_visit = QPushButton("|< Prev Visit")
        self.btn_prev_visit.clicked.connect(self._prev_visit)
        self.btn_step_back = QPushButton("< Step")
        self.btn_step_back.clicked.connect(lambda: self._step(-1))
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_step_fwd = QPushButton("Step >")
        self.btn_step_fwd.clicked.connect(lambda: self._step(1))
        self.btn_next_visit = QPushButton("Next Visit >|")
        self.btn_next_visit.clicked.connect(self._next_visit)

        for btn in [self.btn_prev_visit, self.btn_step_back, self.btn_play,
                     self.btn_step_fwd, self.btn_next_visit]:
            btn.setEnabled(False)
            ctrl_layout.addWidget(btn)
        vl.addLayout(ctrl_layout)

        # Slider + frame offset spinner
        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._slider_changed)
        slider_row.addWidget(self.slider, 1)

        slider_row.addWidget(QLabel("Offset:"))
        self.spin_offset = QSpinBox()
        self.spin_offset.setRange(-20, 20)
        self.spin_offset.setValue(0)
        self.spin_offset.setToolTip("Video frame offset (adjust if video doesn't match tracking)")
        self.spin_offset.valueChanged.connect(self._offset_changed)
        slider_row.addWidget(self.spin_offset)
        vl.addLayout(slider_row)

        self.lbl_info = QLabel("")
        self.lbl_info.setFont(QFont("Consolas", 9))
        vl.addWidget(self.lbl_info)
        splitter.addWidget(video_widget)

        # Right: Visit table + summary
        right_widget = QWidget()
        rl = QVBoxLayout(right_widget)

        ph_layout = QHBoxLayout()
        ph_layout.addWidget(QLabel("Phase:"))
        self.combo_phase = QComboBox()
        self.combo_phase.addItems(["All", "object_habituation", "object_novelty"])
        self.combo_phase.currentTextChanged.connect(self._filter_visits)
        ph_layout.addWidget(self.combo_phase, 1)
        rl.addLayout(ph_layout)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setFont(QFont("Consolas", 10))
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet("padding: 6px; background: #f0f0f0; border-radius: 4px;")
        rl.addWidget(self.lbl_summary)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["#", "Phase", "Arm", "Score", "Time (s)"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._table_clicked)
        rl.addWidget(self.table, 1)

        self.lbl_sequence = QLabel("")
        self.lbl_sequence.setFont(QFont("Consolas", 11))
        self.lbl_sequence.setWordWrap(True)
        self.lbl_sequence.setStyleSheet("padding: 8px; background: #fff; border: 1px solid #ccc;")
        self.lbl_sequence.setTextFormat(Qt.TextFormat.RichText)
        rl.addWidget(self.lbl_sequence)

        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setFont(QFont("Consolas", 8))
        self.event_log.setMaximumHeight(140)
        rl.addWidget(self.event_log)

        splitter.addWidget(right_widget)
        splitter.setSizes([200, 650, 450])
        main_layout.addWidget(splitter, 1)

    # ── Folder Selection ──────────────────────────────────────────────────

    def _select_txt_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select TXT Folder")
        if d:
            self.txt_dir = d
            self.lbl_txt_dir.setText(d)
            self.lbl_txt_dir.setStyleSheet("color: black;")
            if not self.vid_dir:
                self.vid_dir = d
                self.lbl_vid_dir.setText(d + "  (auto)")
                self.lbl_vid_dir.setStyleSheet("color: #555;")
            self.btn_load.setEnabled(True)

    def _select_vid_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if d:
            self.vid_dir = d
            self.lbl_vid_dir.setText(d)
            self.lbl_vid_dir.setStyleSheet("color: black;")

    def _select_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if d:
            self.out_dir = d
            self.lbl_out_dir.setText(d)
            self.lbl_out_dir.setStyleSheet("color: black;")

    def _get_out_dir(self):
        """Return output dir, falling back to txt_dir."""
        return self.out_dir or self.txt_dir

    # ── File Discovery ────────────────────────────────────────────────────

    def _load_files(self):
        vid_dir = self.vid_dir or self.txt_dir
        if not self.txt_dir or not os.path.isdir(self.txt_dir):
            QMessageBox.warning(self, "Error", "TXT folder not found")
            return

        txt_files = sorted(glob.glob(os.path.join(self.txt_dir, '*.txt')))
        if not txt_files:
            QMessageBox.warning(self, "Error", f"No .txt files in:\n{self.txt_dir}")
            return

        self.file_infos = []
        self.file_list.clear()
        self._all_rows = []

        found = no_vid = 0
        for txt_path in txt_files:
            base = os.path.basename(txt_path)
            stem, _ = parse_video_name_from_txt(txt_path)
            video_path = find_video_path(vid_dir, stem)

            self.file_infos.append({
                'txt_path': txt_path, 'video_path': video_path,
                'stem': stem, 'base': base, 'analyzed': False,
            })

            tag = "" if video_path else "  (no video)"
            self.file_list.addItem(f"  {base}{tag}")
            if video_path:
                found += 1
            else:
                no_vid += 1

        self.lbl_file_status.setText(f"{len(txt_files)} txt, {found} video, {no_vid} no video")
        self.btn_run_all.setEnabled(len(self.file_infos) > 0)
        self._log(f"Loaded {len(txt_files)} files")

    # ── Run All ───────────────────────────────────────────────────────────

    def _run_all(self):
        if not self.out_dir:
            QMessageBox.warning(self, "No Save Folder",
                                "Please select a Save Folder before running.")
            return

        self.btn_run_all.setEnabled(False)
        self._all_rows = []
        use_head = self.combo_zone_mode.currentText() == "Head"
        self._log(f"Zone detection mode: {'Head keypoint' if use_head else 'Body center'}")

        for idx, info in enumerate(self.file_infos):
            self._log(f"Analyzing {info['base']}...")

            meta, zone_polys, crop_box, frames = parse_txt_file(info['txt_path'])
            visits, phase_ranges, events = run_state_machine(frames, use_head=use_head)

            info.update({
                'meta': meta, 'zone_polys': zone_polys, 'crop_box': crop_box,
                'frames_data': frames, 'visits': visits, 'events': events,
                'phase_ranges': phase_ranges, 'analyzed': True,
            })

            # Rows for cumulative + individual excel
            mouse_rows = build_sa_rows(meta, visits)
            self._all_rows.extend(mouse_rows)

            # Save individual per-mouse Excel to Save Folder
            md = extract_metadata(meta)
            self._save_individual_excel(md, mouse_rows)

            # Save annotated video to Save Folder
            self._save_annotated_video_for(info)

            # Log per-phase summary
            for phase in ["object_habituation", "object_novelty"]:
                pv = [v for v in visits if v["phase"] == phase]
                scored = [v for v in pv if v["score"] is not None]
                c = sum(1 for v in scored if v["score"] == "C")
                w = sum(1 for v in scored if v["score"] == "W")
                t = c + w
                pct = f"{c/t*100:.0f}%" if t > 0 else "—"
                self._log(f"  {phase}: {len(pv)} visits, C={c} W={w} Alt={pct}")

            # Update list item
            item = self.file_list.item(idx)
            total_v = len(visits)
            total_c = sum(1 for v in visits if v["score"] == "C")
            total_w = sum(1 for v in visits if v["score"] == "W")
            vid_tag = "" if info['video_path'] else "  [no vid]"
            item.setText(f"  {md['mouseID']} | V={total_v} C={total_c} W={total_w} | {info['base']}{vid_tag}")

            QApplication.processEvents()

        # Save cumulative Excel automatically to Save Folder
        if self._all_rows:
            self._save_cumulative_excel()

        self.btn_run_all.setEnabled(True)
        self.btn_export.setEnabled(len(self._all_rows) > 0)
        self._log(f"Done. {len(self.file_infos)} mice analyzed.")

        if self.file_infos:
            self.file_list.setCurrentRow(0)

    # ── File Selection → Load for Verification ────────────────────────────

    def _file_selected(self, row):
        if row < 0 or row >= len(self.file_infos):
            return
        info = self.file_infos[row]
        if not info.get('analyzed'):
            self._clear_verification()
            self.lbl_summary.setText("Not yet analyzed. Click Run All first.")
            return

        self.current_file_idx = row
        self.meta = info['meta']
        self.zone_polys = info['zone_polys']
        self.crop_box = info['crop_box']
        self.frames_data = info['frames_data']
        self.visits = info['visits']
        self.events = info['events']
        self.phase_ranges = info['phase_ranges']

        self._event_by_frame = {}
        for ev in self.events:
            gidx = ev["global_frame_idx"]
            if gidx not in self._event_by_frame:
                self._event_by_frame[gidx] = []
            self._event_by_frame[gidx].append(ev)

        self._sm_state_cache = self._precompute_sm_states()

        # Open video via sequential reader
        if self.video_cache is not None:
            self.video_cache.release()
            self.video_cache = None

        has_video = False
        if info['video_path'] and os.path.isfile(info['video_path']):
            self.video_cache = VideoFrameCache(info['video_path'], self.crop_box)
            has_video = self.video_cache.total_frames > 0

        for btn in [self.btn_prev_visit, self.btn_step_back, self.btn_play,
                     self.btn_step_fwd, self.btn_next_visit]:
            btn.setEnabled(has_video)
        self.slider.setEnabled(has_video)

        if has_video:
            self.slider.setMinimum(0)
            self.slider.setMaximum(len(self.frames_data) - 1)

        self._filter_visits()

        if has_video and self.visits:
            self._jump_to_frame(self.visits[0]["global_frame_idx"])
        elif has_video:
            self._jump_to_frame(0)
        else:
            self.video_label.setText("No video for this file")

    def _clear_verification(self):
        self.table.setRowCount(0)
        self.lbl_sequence.setText("")
        self.lbl_summary.setText("")
        self.lbl_info.setText("")
        self.video_label.clear()
        self.video_label.setText("—")

    # ── Export ────────────────────────────────────────────────────────────

    def _make_ordered_df(self, rows):
        """Build DataFrame with correct column order matching output.xlsx."""
        df = pd.DataFrame(rows)
        meta_cols = ['MouseID', 'Group', 'ExptDate', 'StartTime', 'subgroup',
                     'State', 'Correct', 'Incorrect', 'percent correct', 'Init arm']
        visit_cols = sorted([c for c in df.columns if isinstance(c, int)])
        ordered = [c for c in meta_cols if c in df.columns] + visit_cols
        return df[ordered]

    def _save_individual_excel(self, md, mouse_rows):
        """Save individual per-mouse Excel to Save Folder."""
        out = self._get_out_dir()
        mouse_id = md['mouseID'] or 'unknown'
        safe_id = re.sub(r'[^\w\-]', '_', mouse_id)
        path = os.path.join(out, f"{safe_id}_SA.xlsx")
        df = self._make_ordered_df(mouse_rows)
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer, index=False, sheet_name="Zone Statistics")
        self._log(f"  Individual Excel: {path}")

    def _save_cumulative_excel(self):
        """Save cumulative Excel for all mice to Save Folder."""
        out = self._get_out_dir()
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(out, f"SA_results_{ts}.xlsx")
        df = self._make_ordered_df(self._all_rows)
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer, index=False, sheet_name="Zone Statistics")
        self._log(f"Cumulative Excel saved: {path}")

    def _export_excel(self):
        if not self._all_rows:
            QMessageBox.warning(self, "No Data", "Run All first.")
            return
        self._save_cumulative_excel()
        QMessageBox.information(self, "Exported",
                                f"Saved {len(self._all_rows)} rows to Save Folder.")

    def _save_annotated_video_for(self, info):
        """Write annotated video for a given mouse info dict (object phases only).
        Called automatically during Run All. Uses its own VideoFrameCache."""
        if not info.get('video_path') or not os.path.isfile(info['video_path']):
            self._log(f"  Skipping video — no video file for {info['base']}")
            return

        md = extract_metadata(info['meta'])
        frames_data = info['frames_data']

        # Determine frame range: only object_habituation + object_novelty
        phase_frames = [f["_global_idx"] for f in frames_data
                        if (f.get("state_name") or "") in TARGET_STATES]
        if not phase_frames:
            self._log(f"  Skipping video — no object phase frames")
            return

        start_idx = phase_frames[0]
        end_idx = phase_frames[-1]
        total = end_idx - start_idx + 1

        mouse_id = md['mouseID'] or 'unknown'
        safe_id = re.sub(r'[^\w\-]', '_', mouse_id)
        path = os.path.join(self.out_dir, f"{safe_id}_{info['stem']}_SA_annotated.avi")

        # Temporarily set instance state so _annotate_frame works
        old_zone_polys = getattr(self, 'zone_polys', {})
        old_frames_data = getattr(self, 'frames_data', [])
        old_visits = getattr(self, 'visits', [])
        old_event_by_frame = getattr(self, '_event_by_frame', {})
        old_sm_cache = getattr(self, '_sm_state_cache', None)
        old_crop_box = getattr(self, 'crop_box', None)

        self.zone_polys = info['zone_polys']
        self.frames_data = info['frames_data']
        self.visits = info['visits']
        self.events = info['events']

        self._event_by_frame = {}
        for ev in self.events:
            gidx = ev["global_frame_idx"]
            if gidx not in self._event_by_frame:
                self._event_by_frame[gidx] = []
            self._event_by_frame[gidx].append(ev)

        self._sm_state_cache = self._precompute_sm_states()

        self.crop_box = info['crop_box']
        vcache = VideoFrameCache(info['video_path'], info['crop_box'])
        offset = self.spin_offset.value()

        self._log(f"  Saving video: {total} frames...")
        QApplication.processEvents()

        # Use absolute video frame number for first sample
        first_vf = frames_data[start_idx]["_video_frame"] + offset
        sample = vcache.get_frame(first_vf)
        if sample is None:
            self._log(f"  Error: cannot read video frame")
            vcache.release()
            self.zone_polys = old_zone_polys
            self.frames_data = old_frames_data
            self.visits = old_visits
            self._event_by_frame = old_event_by_frame
            self._sm_state_cache = old_sm_cache
            self.crop_box = old_crop_box
            return

        h, w = sample.shape[:2]
        fps = vcache.fps
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

        for gidx in range(start_idx, end_idx + 1):
            vf = frames_data[gidx]["_video_frame"] + offset if gidx < len(frames_data) else gidx
            raw = vcache.get_frame(vf)
            if raw is None:
                continue
            frame_data = frames_data[gidx] if gidx < len(frames_data) else {}
            annotated = self._annotate_frame(raw, frame_data, gidx)
            writer.write(annotated)

            done = gidx - start_idx + 1
            if done % 500 == 0:
                self._log(f"    {done}/{total} frames...")
                QApplication.processEvents()

        writer.release()
        vcache.release()
        self._log(f"  Video saved: {path}")

        # Restore previous state
        self.zone_polys = old_zone_polys
        self.frames_data = old_frames_data
        self.visits = old_visits
        self._event_by_frame = old_event_by_frame
        self._sm_state_cache = old_sm_cache
        self.crop_box = old_crop_box

    def _log(self, msg):
        self.event_log.append(msg)

    # ── Visit Table ───────────────────────────────────────────────────────

    def _filter_visits(self):
        phase_filter = self.combo_phase.currentText()
        if phase_filter == "All":
            filtered = self.visits
        else:
            filtered = [v for v in self.visits if v["phase"] == phase_filter]
        self._filtered_visits = filtered
        self._populate_table(filtered)
        self._update_summary(filtered)
        self._update_sequence(filtered)

    def _populate_table(self, visits):
        self.table.setRowCount(len(visits))
        for i, v in enumerate(visits):
            self.table.setItem(i, 0, QTableWidgetItem(str(v["visit_num"])))
            self.table.setItem(i, 1, QTableWidgetItem(v["phase"]))
            self.table.setItem(i, 2, QTableWidgetItem(v["arm"]))

            score_item = QTableWidgetItem(v["score"] if v["score"] else "-")
            if v["score"] == "C":
                score_item.setBackground(QBrush(QColor(144, 238, 144)))
            elif v["score"] == "W":
                score_item.setBackground(QBrush(QColor(255, 160, 160)))
            self.table.setItem(i, 3, score_item)
            self.table.setItem(i, 4, QTableWidgetItem(f"{v['timestamp_ms']/1000:.1f}"))

    def _update_summary(self, visits):
        scored = [v for v in visits if v["score"] is not None]
        c_count = sum(1 for v in scored if v["score"] == "C")
        w_count = sum(1 for v in scored if v["score"] == "W")
        total = c_count + w_count
        pct = (c_count / total * 100) if total > 0 else float("nan")

        md = extract_metadata(self.meta)
        lines = [
            f"Mouse: {md['mouseID']}  |  {md['subgroup']}",
            f"Phase: {self.combo_phase.currentText()}",
            f"Total visits: {len(visits)}",
            f"Scored: {total}  (C={c_count}, W={w_count})",
            f"Alternation: {pct:.1f}%  (chance=50%)",
        ]
        self.lbl_summary.setText("\n".join(lines))

    def _update_sequence(self, visits):
        parts = []
        for v in visits:
            if v["score"] is None:
                arm_short = "O1" if v["arm"] == "Object1" else "O2"
                parts.append(f'<span style="color:#888;">[{arm_short}]</span>')
            elif v["score"] == "C":
                parts.append('<span style="color:green; font-weight:bold;">C</span>')
            else:
                parts.append('<span style="color:red; font-weight:bold;">W</span>')
        self.lbl_sequence.setText("Sequence: " + ", ".join(parts))

    def _table_clicked(self, row, col):
        if 0 <= row < len(self._filtered_visits):
            self._jump_to_frame(self._filtered_visits[row]["global_frame_idx"])

    # ── Video Display ─────────────────────────────────────────────────────

    def _jump_to_frame(self, global_idx):
        if not self.frames_data or self.video_cache is None:
            return
        global_idx = max(0, min(global_idx, len(self.frames_data) - 1))
        self.current_global_idx = global_idx

        # Use absolute video frame number from data, not line index
        video_idx = self.frames_data[global_idx]["_video_frame"] + self.spin_offset.value()
        raw_frame = self.video_cache.get_frame(video_idx)
        if raw_frame is None:
            return

        frame_data = self.frames_data[global_idx]
        annotated = self._annotate_frame(raw_frame, frame_data, global_idx)
        self._show_frame(annotated)

        self.slider.blockSignals(True)
        self.slider.setValue(global_idx)
        self.slider.blockSignals(False)
        self._update_info(frame_data, global_idx)
        self._highlight_current_visit(global_idx)

    def _offset_changed(self, val):
        """Re-display current frame with new offset."""
        self._jump_to_frame(self.current_global_idx)

    def _annotate_frame(self, frame, frame_data, global_idx):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        loc = frame_data.get("location", "")
        state_name = frame_data.get("state_name", "")

        # Zone outlines
        for zname, pts in self.zone_polys.items():
            ipts = pts.astype(np.int32)
            if zname in IA_ZONES:
                cv2.polylines(overlay, [ipts], True, ZONE_COLORS.get(zname, (255, 255, 0)), 2)
            elif zname in CENTRE_ZONES:
                cv2.polylines(overlay, [ipts], True, CENTRE_COLOR, 1)
            else:
                cv2.polylines(overlay, [ipts], True, DEFAULT_ZONE_COLOR, 1)

        # Highlight current zone
        if loc and loc in self.zone_polys:
            ipts = self.zone_polys[loc].astype(np.int32)
            if loc in IA_ZONES:
                color = ZONE_COLORS.get(loc, (255, 255, 0))
                cv2.fillPoly(overlay, [ipts], color)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                overlay = frame.copy()
                cv2.polylines(overlay, [ipts], True, color, 3)
            elif loc in CENTRE_ZONES:
                cv2.fillPoly(overlay, [ipts], CENTRE_COLOR)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                overlay = frame.copy()
                cv2.polylines(overlay, [ipts], True, CENTRE_COLOR, 2)

        # Pose skeleton — head (index 0) drawn larger
        pose = frame_data.get("pose_array", [])
        if pose and len(pose) >= 2:
            kp_colors = [(0, 255, 255), (0, 255, 0), (255, 0, 255)]
            valid_kps = []
            for ki, kp in enumerate(pose):
                if len(kp) >= 2 and np.isfinite(kp[0]) and np.isfinite(kp[1]):
                    cx, cy = int(kp[0]), int(kp[1])
                    c = kp_colors[ki] if ki < len(kp_colors) else (255, 255, 255)
                    r = 7 if ki == 0 else 4  # head keypoint larger
                    cv2.circle(overlay, (cx, cy), r, c, -1)
                    cv2.circle(overlay, (cx, cy), r + 1, (0, 0, 0), 1)
                    valid_kps.append((cx, cy))
                else:
                    valid_kps.append(None)
            # Label head
            if valid_kps and valid_kps[0]:
                cv2.putText(overlay, "H", (valid_kps[0][0] + 8, valid_kps[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            for i in range(min(len(valid_kps) - 1, 2)):
                if valid_kps[i] and valid_kps[i + 1]:
                    cv2.line(overlay, valid_kps[i], valid_kps[i + 1], (255, 255, 255), 1)

        # HUD
        sm_state, last_arm, _ = self._get_sm_state_at(global_idx)
        head_z = frame_data.get("_head_zone", "")
        y_t = 20
        phase_str = state_name.replace("State: ", "") if state_name else "—"
        self._put_text(overlay, f"Phase: {phase_str}", (10, y_t), (255, 255, 255)); y_t += 22
        self._put_text(overlay, f"Body: {loc}  Head: {head_z}", (10, y_t), (255, 255, 200)); y_t += 22
        sm_col = (100, 255, 100) if sm_state == "IN_CENTRE" else (100, 180, 255)
        self._put_text(overlay, f"State: {sm_state}", (10, y_t), sm_col); y_t += 22
        if last_arm:
            self._put_text(overlay, f"Last arm: {last_arm}", (10, y_t), (200, 200, 200))

        # Events
        for ev in self._event_by_frame.get(global_idx, []):
            if ev["type"] == "VISIT":
                s, arm = ev["score"], ev["arm"]
                if s == "C":
                    txt, col = f"VISIT #{ev['visit_num']}: {arm} -> CORRECT (alternated)", (0, 220, 0)
                elif s == "W":
                    txt, col = f"VISIT #{ev['visit_num']}: {arm} -> WRONG (same arm)", (0, 0, 255)
                else:
                    txt, col = f"VISIT #{ev['visit_num']}: {arm} (first visit)", (255, 200, 0)
                self._put_text_big(overlay, txt, (10, h - 50), col)
            elif ev["type"] == "IGNORED_IA":
                self._put_text(overlay, f"IA IGNORED ({ev['reason']})", (10, h - 25), (128, 128, 255))
            elif ev["type"] == "CENTRE_RESET":
                self._put_text(overlay, f"CENTRE RESET via {ev['zone']}", (10, h - 25), (100, 255, 100))

        # Look-ahead/behind
        if not self._event_by_frame.get(global_idx):
            for v in self.visits:
                diff = v["global_frame_idx"] - global_idx
                if 1 <= diff <= 60:
                    self._put_text(overlay, f"-> Visit #{v['visit_num']} in {diff} frames",
                                   (10, h - 25), (200, 200, 100))
                    break
                elif -30 <= diff < 0:
                    tag = v["score"] if v["score"] else "first"
                    self._put_text(overlay, f"Visit #{v['visit_num']} was {abs(diff)} frames ago [{tag}]",
                                   (10, h - 25), (200, 180, 100))
                    break

        # IA labels
        for zname in IA_ZONES:
            if zname in self.zone_polys:
                pts = self.zone_polys[zname]
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                cv2.putText(overlay, "Obj1" if "Object1" in zname else "Obj2",
                            (cx - 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return overlay

    # ── State Machine Cache ───────────────────────────────────────────────

    def _precompute_sm_states(self):
        """Build per-frame (sm_state, last_arm) cache from already-computed events."""
        n = len(self.frames_data)
        # Build sorted list of state transitions from events
        transitions = []  # (global_frame_idx, new_state, new_last_arm)
        sm_state = "IN_CENTRE"
        last_arm = None
        for ev in sorted(self.events, key=lambda e: e["global_frame_idx"]):
            if ev["type"] == "VISIT":
                sm_state = "IN_ARM"
                last_arm = ev["arm"]
                transitions.append((ev["global_frame_idx"], sm_state, last_arm))
            elif ev["type"] == "CENTRE_RESET":
                sm_state = "IN_CENTRE"
                transitions.append((ev["global_frame_idx"], sm_state, last_arm))

        # Fill cache — carry state forward between transitions
        cache = [("IN_CENTRE", None)] * n
        ti = 0
        cur_state, cur_arm = "IN_CENTRE", None
        for i in range(n):
            while ti < len(transitions) and transitions[ti][0] <= i:
                _, cur_state, cur_arm = transitions[ti]
                ti += 1
            cache[i] = (cur_state, cur_arm)
        return cache

    def _get_sm_state_at(self, idx):
        if self._sm_state_cache and 0 <= idx < len(self._sm_state_cache):
            return self._sm_state_cache[idx][0], self._sm_state_cache[idx][1], None
        return "IN_CENTRE", None, None

    # ── Drawing ───────────────────────────────────────────────────────────

    def _put_text(self, img, text, pos, color, scale=0.5, thickness=1):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def _put_text_big(self, img, text, pos, color):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _show_frame(self, frame):
        h, w = frame.shape[:2]
        lw, lh = self.video_label.width(), self.video_label.height()
        if lw > 10 and lh > 10:
            s = min(lw / w, lh / h, 1.0)
            if s < 1.0:
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def _update_info(self, frame_data, global_idx):
        loc = frame_data.get("location", "?")
        ts = frame_data.get("timestamps", 0)
        sn = (frame_data.get("state_name") or "").replace("State: ", "")
        spd = frame_data.get("speed", 0)
        vf = frame_data.get("_video_frame", global_idx)
        off = self.spin_offset.value()
        self.lbl_info.setText(
            f"Idx: {global_idx}/{len(self.frames_data)-1}  |  "
            f"VideoFrame: {vf}+{off}  |  "
            f"Zone: {loc}  |  State: {sn}  |  "
            f"Time: {ts/1000:.2f}s  |  Speed: {spd:.1f}"
        )

    def _highlight_current_visit(self, global_idx):
        best_row, best_dist = -1, float("inf")
        for i, v in enumerate(self._filtered_visits):
            d = abs(v["global_frame_idx"] - global_idx)
            if d < best_dist:
                best_dist, best_row = d, i
        if best_row >= 0 and best_dist < 100:
            self.table.selectRow(best_row)

    # ── Navigation ────────────────────────────────────────────────────────

    def _step(self, delta):
        self._jump_to_frame(max(0, min(self.current_global_idx + delta, len(self.frames_data) - 1)))

    def _next_visit(self):
        for v in self._filtered_visits:
            if v["global_frame_idx"] > self.current_global_idx:
                self._jump_to_frame(v["global_frame_idx"])
                return
        if self._filtered_visits:
            self._jump_to_frame(self._filtered_visits[0]["global_frame_idx"])

    def _prev_visit(self):
        for v in reversed(self._filtered_visits):
            if v["global_frame_idx"] < self.current_global_idx:
                self._jump_to_frame(v["global_frame_idx"])
                return
        if self._filtered_visits:
            self._jump_to_frame(self._filtered_visits[-1]["global_frame_idx"])

    def _toggle_play(self):
        if self.playing:
            self.playing = False
            self.play_timer.stop()
            self.btn_play.setText("Play")
        else:
            self.playing = True
            self.btn_play.setText("Pause")
            self.play_timer.start(max(16, int(1000 / (self.video_cache.fps if self.video_cache else 15))))

    def _play_tick(self):
        if self.current_global_idx >= len(self.frames_data) - 1:
            self._toggle_play()
            return
        self._step(1)

    def _slider_changed(self, value):
        self._jump_to_frame(value)

    def closeEvent(self, event):
        if self.video_cache is not None:
            self.video_cache.release()
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SAVerificationApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
