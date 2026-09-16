import math

import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from PyQt5.QtWidgets import QInputDialog, QListWidgetItem
from gui.create_zone import Ui_Form
from PyQt5 import QtWidgets
from matplotlib.widgets import Cursor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from collections import defaultdict
from shapely.geometry.polygon import Polygon
import json
import numpy as np


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=10, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()


def print_item():
    print('double click')


class OpenSecondaryGUI:
    def __init__(self):
        self._zone_array = None
        self._zone_gui = None
        self.window = None
        self.PlotZones = None
        self.canvas = None

    def create_zone(self, snap_image=None):
        global zone_dict
        global _length_scale
        zone_dict = defaultdict(dict)
        self.window = QtWidgets.QWidget()
        self._zone_gui = Ui_Form()
        self._zone_gui.setupUi(self.window)
        self.window.show()
        self.canvas = MplCanvas(self, width=10, height=10, dpi=100)
        toolbar = NavigationToolbar(self.canvas, self)
        self._zone_gui.horizontalLayout.addWidget(toolbar, 0)
        self._zone_gui.horizontalLayout_4.addWidget(self.canvas, 0)
        self.canvas.axes.cla()
        self.PlotZones = PlotZones()
        self.canvas.axes.imshow(snap_image, origin='lower')
        self.canvas.axes.invert_yaxis()
        plt.tight_layout()
        self.canvas.draw()

        self._zone_gui.pushButton_CreateCircle.clicked.connect(
            lambda: self.PlotZones.create_zone_circle(axis_name=self.canvas.axes,
                                                      zone_name=self._zone_gui.listWidget_ZoneList))
        self._zone_gui.pushButton_CreateRectangle.clicked.connect(
            lambda: self.PlotZones.create_zone_rectangle(axis_name=self.canvas.axes,
                                                         zone_name=self._zone_gui.listWidget_ZoneList))
        self._zone_gui.pushButton_CreatePolygon.clicked.connect(
            lambda: self.PlotZones.create_zone_polygon(axis_name=self.canvas.axes,
                                                       zone_name=self._zone_gui.listWidget_ZoneList))
        self._zone_gui.pushButton_SaveZoneConfig.clicked.connect(
            lambda: save_config(scale_value=self._zone_gui.lineEdit_ScaleLength))
        self._zone_gui.pushButton_AddScale.clicked.connect(
            lambda: self.PlotZones.measuse_length(axis_name=self.canvas.axes,
                                                  zone_name=self._zone_gui.listWidget_ZoneList))

        ##Arena
        self._zone_gui.pushButton_ArenaRectangle.clicked.connect(
            lambda: self.PlotZones.create_arena_rectangle(axis_name=self.canvas.axes,
                                                          zone_name=self._zone_gui.listWidget_ArenaList))

        self._zone_gui.pushButton_ArenaPolygon.clicked.connect(
            lambda: self.PlotZones.create_arena_polygon(axis_name=self.canvas.axes,
                                                        zone_name=self._zone_gui.listWidget_ArenaList))

        # self._zone_gui.listWidget_ZoneList.itemDoubleClicked.connect(lambda: print_item)

    # def self._zone_gui(self, event):
    #     print('closed')
    # close = QtWidgets.QMessageBox.question(self, "QUIT",
    #                                        "Are you sure want to stop process?",
    #                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    # if close == QtWidgets.QMessageBox.Yes:
    #     event.accept()
    # else:
    #     event.ignore()


def save_config(scale_value=None):
    import json
    with open('zone_config.json', 'w') as fp:
        zone_dict['scale']['length'] = scale_value.text()
        json.dump(zone_dict, fp)


class PlotZones:
    def __init__(self):
        self.es = None
        self.cursor = None
        self._keyID = None
        self._count_zones = 0

    def create_zone_circle(self, axis_name=None, zone_name=None):
        text, ok = QInputDialog.getText(None, 'type zone name', 'Type zone name and press Enter:')
        if ok and len(text) != 0:
            self._keyID = text
            self._count_zones += 1
            print(text, self._count_zones)
            item1 = QListWidgetItem(self._keyID)
            zone_name.addItem(item1)

            props = dict(color='red', linestyle='-', linewidth=0.1, alpha=0.3)
            self.cursor = Cursor(axis_name, useblit=False, color='white', linewidth=1)
            self.es = mwidgets.EllipseSelector(axis_name, onselect=self.onselect_zone_circle, interactive=True)

    def onselect_zone_circle(self, eclick, erelease):
        dx, dy = abs(erelease.xdata - eclick.xdata), abs(erelease.ydata - eclick.ydata)
        if dx == dy:
            _type = 'circle'
        else:
            _type = 'ellipse'
        _points = [eclick.xdata + dx / 2, eclick.ydata + dy / 2]
        zone_dict['zones'][self._keyID] = {}
        zone_dict['zones'][self._keyID]['name'] = self._keyID
        zone_dict['zones'][self._keyID]['type'] = _type
        zone_dict['zones'][self._keyID]['shape_dim'] = dx / 2, dy / 2
        zone_dict['zones'][self._keyID]['values'] = [eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata]
        zone_dict['zones'][self._keyID]['points'] = _points

    def create_zone_rectangle(self, axis_name=None, zone_name=None):
        text, ok = QInputDialog.getText(None, 'type zone name', 'Type zone name and press Enter:')
        if ok and len(text) != 0:
            self._keyID = text
            self._count_zones += 1
            print(text, self._count_zones)
            item1 = QListWidgetItem(self._keyID)
            zone_name.addItem(item1)
            props = dict(color='orange', linestyle='-', linewidth=0.1, alpha=0.2)
            self.cursor = Cursor(axis_name, useblit=False, color='white', linewidth=1)
            self.es = mwidgets.RectangleSelector(axis_name, onselect=self.onselect_zone_rectangle, interactive=True)

    def onselect_zone_rectangle(self, eclick, erelease):
        dx, dy = abs(erelease.xdata - eclick.xdata), abs(erelease.ydata - eclick.ydata)
        _type = 'rectangle'
        _points = [eclick.xdata + dx / 2, eclick.ydata + dy / 2]
        zone_dict['zones'][self._keyID] = {}
        zone_dict['zones'][self._keyID]['name'] = self._keyID
        zone_dict['zones'][self._keyID]['type'] = _type
        zone_dict['zones'][self._keyID]['shape_dim'] = dx / 2, dy / 2
        zone_dict['zones'][self._keyID]['values'] = [eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata]
        zone_dict['zones'][self._keyID]['points'] = _points

    def create_zone_polygon(self, axis_name=None, zone_name=None):
        text, ok = QInputDialog.getText(None, 'type zone name', 'Type zone name and press Enter:')
        if ok and len(text) != 0:
            self._keyID = text
            self._count_zones += 1
            print(text, self._count_zones)
            item1 = QListWidgetItem(self._keyID)
            zone_name.addItem(item1)
            props = dict(color='r', linestyle='-', linewidth=2, alpha=0.7)
            self.cursor = Cursor(axis_name, useblit=False, color='white', linewidth=1)
            self.es = mwidgets.PolygonSelector(axis_name, onselect=self.onselect_zone_polygon)

    def onselect_zone_polygon(self, erelease):
        _type = 'polygon'
        zone_dict['zones'][self._keyID] = {}
        zone_dict['zones'][self._keyID]['name'] = self._keyID
        zone_dict['zones'][self._keyID]['type'] = _type
        zone_dict['zones'][self._keyID]['shape_dim'] = None
        zone_dict['zones'][self._keyID]['values'] = erelease
        zone_dict['zones'][self._keyID]['points'] = None

    def measuse_length(self, axis_name=None, zone_name=None):
        props = dict(color='r', linestyle='-', linewidth=2, alpha=0.7)
        self.cursor = Cursor(axis_name, useblit=False, color='white', linewidth=1)
        self.es = mwidgets.SpanSelector(axis_name, onselect=self.onselect_line, direction='horizontal')

    def onselect_line(self, vmin, vmax):
        zone_dict['scale'] = {}
        zone_dict['scale']['length'] = 0
        zone_dict['scale']['values'] = vmin, vmax

    ##Arena
    def create_arena_rectangle(self, axis_name=None, zone_name=None):
        text, ok = QInputDialog.getText(None, 'type zone name', 'Type zone name and press Enter:')
        if ok and len(text) != 0:
            self._keyID = text
            self._count_zones += 1
            print(text, self._count_zones)
            item1 = QListWidgetItem(self._keyID)
            zone_name.addItem(item1)
            props = dict(color='orange', linestyle='-', linewidth=0.1, alpha=0.2)
            self.cursor = Cursor(axis_name, useblit=False, color='white', linewidth=1)
            self.es = mwidgets.RectangleSelector(axis_name, onselect=self.onselect_arena_rectangle, interactive=True)

    def onselect_arena_rectangle(self, eclick, erelease):
        zone_dict['arena'][self._keyID] = {}
        zone_dict['arena'][self._keyID]['name'] = self._keyID
        zone_dict['arena'][self._keyID]['crop_values'] = [round(eclick.ydata), round(erelease.ydata),
                                                          round(eclick.xdata), round(erelease.xdata)]

    def create_arena_polygon(self, axis_name=None, zone_name=None):
        text, ok = QInputDialog.getText(None, 'type zone name', 'Type zone name and press Enter:')
        if ok and len(text) != 0:
            self._keyID = text
            self._count_zones += 1
            print(text, self._count_zones)
            item1 = QListWidgetItem(self._keyID)
            zone_name.addItem(item1)
            props = dict(color='r', linestyle='-', linewidth=2, alpha=0.7)
            self.cursor = Cursor(axis_name, useblit=False, color='white', linewidth=1)
            self.es = mwidgets.PolygonSelector(axis_name, onselect=self.onselect_arena_polygon)

    def onselect_arena_polygon(self, erelease):
        _type = 'polygon'
        zone_dict['arena'][self._keyID] = {}
        zone_dict['arena'][self._keyID]['name'] = self._keyID
        zone_dict['arena'][self._keyID]['type'] = _type
        zone_dict['arena'][self._keyID]['shape_dim'] = None
        zone_dict['arena'][self._keyID]['values'] = erelease
        zone_dict['arena'][self._keyID]['points'] = None


ZOOM_PCT_PER_STEP = 1.0   # 1% size change per spinbox step
ROTATE_DEG_PER_STEP = 1.8  # QDial 0..99 (default 50) → ±90° around center


class EditZoneConfig:
    """Loads a zone-config JSON, exposes live editing (translate/scale/rotate),
    and writes the edited config back. All transforms are applied to the
    ORIGINAL coordinates loaded from disk and composed at each refresh, so
    sliders/dials act as absolute offsets and don't drift.
    """

    def __init__(self):
        self.pixel_per_cm = None
        self.json_config_file = None
        self.zone_names_key = None
        # Working geometry (mirrored onto the StartMaze instance).
        self.polygons_list = []
        self.zone_list = []
        self.zone_list_key = []
        # Original (un-edited) coords keyed by zone name.
        self._orig_zone_coords = {}
        # Absolute edit state (matches widget values directly).
        self.edit_dx = 0.0
        self.edit_dy = 0.0
        self.edit_zoom_steps = 0
        self.edit_rotate_steps = 0
        # Optional crop loaded from arena.crop_values: [ch1, ch2, cw1, cw2].
        self.arena_crop = None

    # ---------------- Load ----------------

    def read_config_file(self, path):
        with open(path, 'r') as zone_object:
            self.json_config_file = json.load(zone_object)
        self.zone_names_key = list(self.json_config_file['zones'].keys())
        # Snapshot originals so transforms compose without drift.
        self._orig_zone_coords = {
            name: [list(pt) for pt in self.json_config_file['zones'][name]['values']]
            for name in self.zone_names_key
        }
        # Reset edit state on every fresh load.
        self.edit_dx = 0.0
        self.edit_dy = 0.0
        self.edit_zoom_steps = 0
        self.edit_rotate_steps = 0
        # Scale (pixel/cm) — guard against missing/zero length.
        try:
            length = float(self.json_config_file['scale']['length'])
            v_min, v_max = self.json_config_file['scale']['values']
            self.pixel_per_cm = (v_max - v_min) / length if length else None
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            self.pixel_per_cm = None
        # Optional crop window from arena rectangle, if user drew one.
        # NOTE: callers pass self=StartMaze (legacy pattern), so internal
        # helpers must be reached via the class, not via `self`.
        self.arena_crop = EditZoneConfig._extract_arena_crop(self.json_config_file)

    @staticmethod
    def _extract_arena_crop(cfg):
        """Return [ch1, ch2, cw1, cw2] from the first arena entry that has
        crop_values, or None. Order matches the original convention used by
        FrameDataLogger (rows then cols, both as [low, high])."""
        arena = cfg.get('arena')
        if not isinstance(arena, dict):
            return None
        for entry in arena.values():
            cv = entry.get('crop_values') if isinstance(entry, dict) else None
            if cv and len(cv) == 4:
                ch1, ch2, cw1, cw2 = cv
                # Tolerate user drawing in either direction.
                ch1, ch2 = sorted((int(ch1), int(ch2)))
                cw1, cw2 = sorted((int(cw1), int(cw2)))
                return [ch1, ch2, cw1, cw2]
        return None

    # ---------------- Compose & rebuild ----------------

    def coords_shapely_object(self):
        """Initial build after load. Equivalent to applying zero edits."""
        EditZoneConfig._refresh_geometry(self)

    def _refresh_geometry(self):
        """Rebuild polygons_list / zone_list / zone_list_key and write the
        composed coords back into json_config_file so save_edited_config()
        captures the on-screen state. Internal helpers are reached via the
        class, not via `self`, because `self` here is the StartMaze instance
        (legacy calling pattern)."""
        self.zone_list_key = list(self.zone_names_key) if self.zone_names_key else []
        self.zone_list = []
        self.polygons_list = []

        cx, cy = EditZoneConfig._composed_centroid(self)
        cos_t = math.cos(math.radians(self.edit_rotate_steps * ROTATE_DEG_PER_STEP))
        sin_t = math.sin(math.radians(self.edit_rotate_steps * ROTATE_DEG_PER_STEP))
        scale = max(0.01, 1.0 + (self.edit_zoom_steps * ZOOM_PCT_PER_STEP / 100.0))

        for name in self.zone_list_key:
            pts = self._orig_zone_coords.get(name, [])
            transformed = []
            for x, y in pts:
                # translate so centroid is at origin, then scale, rotate,
                # translate back, and finally apply the user x/y offset
                px = (x - cx) * scale
                py = (y - cy) * scale
                rx = px * cos_t - py * sin_t
                ry = px * sin_t + py * cos_t
                transformed.append((rx + cx + self.edit_dx, ry + cy + self.edit_dy))
            self.json_config_file['zones'][name]['values'] = [list(p) for p in transformed]
            polygon = Polygon(np.squeeze(transformed))
            self.polygons_list.append(polygon)
            self.zone_list.append([np.array(polygon.exterior.coords, np.int32)])

    def _composed_centroid(self):
        """Centroid of all original points — the pivot for scale/rotate."""
        xs, ys = [], []
        for pts in self._orig_zone_coords.values():
            for x, y in pts:
                xs.append(x)
                ys.append(y)
        if not xs:
            return 0.0, 0.0
        return sum(xs) / len(xs), sum(ys) / len(ys)

    # ---------------- Edit handlers (called from start_maze.py) ----------------

    def move_maze_left_right(self, value):
        self.edit_dx = float(value)
        EditZoneConfig._refresh_geometry(self)

    def move_maze_top_bottom(self, value):
        self.edit_dy = float(value)
        EditZoneConfig._refresh_geometry(self)

    def scale_zones(self, value):
        """Spinbox value is interpreted as percent change (1 step ≈ 1%)."""
        self.edit_zoom_steps = int(value)
        EditZoneConfig._refresh_geometry(self)

    def rotate_zones(self, value):
        """QDial 0..99 with default 50: midpoint → 0°, edges → ±~88°."""
        self.edit_rotate_steps = int(value) - 50
        EditZoneConfig._refresh_geometry(self)

    def reset_edits(self):
        self.edit_dx = 0.0
        self.edit_dy = 0.0
        self.edit_zoom_steps = 0
        self.edit_rotate_steps = 0
        EditZoneConfig._refresh_geometry(self)

    # ---------------- Save ----------------

    def save_edited_config(self, path='zone_config_user_edited.json'):
        """Persist composed coords. The arena and scale blocks are kept as-is."""
        if self.json_config_file is None:
            return
        with open(path, 'w') as fp:
            json.dump(self.json_config_file, fp)
