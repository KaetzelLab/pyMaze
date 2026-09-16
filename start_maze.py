import os
import sys
import time
import cv2
import math
import numpy as np
from datetime import timedelta
from shapely.geometry import Point
from gui.frame_to_file_logger import FrameDataLogger
from gui.create_zone_functions import OpenSecondaryGUI
from PyQt5.QtCore import pyqtSlot, QTimer, QThread
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from dlclive import DLCLive
from gui.create_zone_functions import PlotZones, EditZoneConfig
from gui.pyMaze_main import Ui_MainWindow
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtGui, QtCore
from datetime import datetime
from serial import SerialException
from serial.tools import list_ports

# Add parent directory to path to allow imports.
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not top_dir in sys.path: sys.path.insert(0, top_dir)

# Add parent directory to path to allow imports.
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not top_dir in sys.path: sys.path.insert(0, top_dir)

from com.pycboard import Pycboard, PyboardError, _djb2_file
from com.data_logger import Data_logger

from config.paths import data_dir, tasks_dir
from config.gui_settings import update_interval
from gui.dialogs import Settings_dialog, Board_config_dialog, Variables_dialog

model_path = 'models/SpontMaze_withOutImplant_training'
path_video = '/home/dennis/Documents/PyMazes/video/138_TM_12-2022-08-18-170556.avi'
dlc_live = DLCLive(model_path, display=False, resize=1)


# #----------------------------------------------------------------
# # Create widgets.
def gui_excepthook(error_type, error_msg, traceback):
    sys.__excepthook__(error_type, error_msg, traceback)


sys.excepthook = gui_excepthook


# ----------------------------------------------------------------
# Main GUI Functions
# -----------------------------------------------------------------
class StartMaze(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(QtWidgets.QMainWindow, self).__init__(parent)
        super().__init__(parent)
        self.experiment_paused = None
        self.setupUi(self)
        self.setWindowTitle('PyMaze')
        self.setWindowIcon(QIcon('gui/icons/python.svg'))
        self.state_name = None
        self.output = None
        self.qImg = None
        self.array = []
        self.location_head = None
        self.location_center = None
        self.variables_dialog = None
        self.zone_list_key = []
        self.polygons_list = []
        self.zone_list = []
        self.config_file_pathname = None
        self.data_file = None
        # tracking and video  related variable
        self.VideoSteamer.setText("Enter Camera ID and press connect")
        self.EditZoneConfig = EditZoneConfig
        self.int_width = None
        self.int_height = None
        self.x_cord = None
        self.mcu_running = False
        self.track_subject = False
        self.new_frame_time = None
        self.init_time = None
        self.pixel_per_cm = 0
        self.frame_for_mcu = 0
        self.config_loaded = False
        self.marker_size = self.spinBox.value()
        self._width = None
        self._height = None
        self.PlotZones = PlotZones()
        self.cropped_image = None
        self.image = None
        self.aspect_height = None
        self.aspect_width = None
        self.cap = None
        self.init_width = 1280
        self.init_height = 720
        self.scaling_height = None
        self.scaling_width = None
        self.json_config_file = None
        self.cursor = None
        self.init_time = 0
        self.rect = None
        self.prev_frame_time = 0
        self.velocity = 0
        self.frame_time = 0
        self.prev_coord = [0, 0]
        self.curr_coord = [0, 0]
        self.pixel_factor = 0.5
        self.time_stamps = 0
        self.record_video = False
        self.write_video_info = False
        self.aspect_ratio_width = 0
        self.aspect_ratio_Height = 0
        self.image_scaling_value = 0
        self.n_frame = 0
        self.tracking_frame = 0
        self.get_zone_name = None
        self.zone_1 = np.array([], np.int32)
        self.cam_capture_timer = QTimer()
        self.cam_capture_timer.timeout.connect(self.video_streamer)
        self.pushButton_create_zone.clicked.connect(
            lambda: OpenSecondaryGUI.create_zone(self, snap_image=self.cropped_image))
        self.pushButton_create_zone.clicked.connect(lambda: self.cam_capture_timer.stop())
        self.pushButton_conenct_to_cam.clicked.connect(lambda: self.connect_camera())  # connect to camera
        self.pushButton_disconenct_to_cam.clicked.connect(lambda: self.disconnect_camera())  # disconnect to camera
        self.pushButton_init_tracking.clicked.connect(lambda: self.init_tracking())  # initiate tracking
        self.pushButton_pause_experiment.clicked.connect(lambda: self.continue_state())
        self.pushButton_restart.clicked.connect(lambda: self.continue_state())
        self.pushButton_stop_tracking.clicked.connect(lambda: self.stop_tracking())  # stop tracking
        # load_config_file
        self.pushButton_load_config.clicked.connect(lambda: self.load_config_file())  # stop tracking
        self.ZoneMoveLeft.valueChanged.connect(
            lambda: self.EditZoneConfig.move_maze_left_right(self, value=self.ZoneMoveLeft.value()))
        self.ZoneMoveTop.valueChanged.connect(
            lambda: self.EditZoneConfig.move_maze_top_bottom(self, value=self.ZoneMoveTop.value()))
        self.ZoneZoomIn.valueChanged.connect(
            lambda: self.EditZoneConfig.scale_zones(self, value=self.ZoneZoomIn.value()))
        self.RotateZone.valueChanged.connect(
            lambda: self.EditZoneConfig.rotate_zones(self, value=self.RotateZone.value()))
        self.pushButton_save_zone_config.clicked.connect(lambda: self.EditZoneConfig.save_edited_config(self))
        self.spinBox.valueChanged.connect(lambda: self.set_marker_size())
        # MCU (Microcontroller) related variables
        self.experimenter_name = None
        self.thres_movement = 1
        self.mcu_connected = False
        self.thresh_detct_ci = 0.8
        # Crop window for the camera frame: image[ch1:ch2, cw1:cw2].
        # Defaults match the historical hard-coded crop the DLC models were
        # trained against and that existing zone JSONs are drawn in. A loaded
        # zone-config can override these via arena.crop_values; if it has no
        # arena block, the defaults stay so DLC and the zone polygons keep
        # sharing the same coordinate system. None on either axis means
        # "don't crop on that axis" (Python slicing semantics).
        self.ch1, self.ch2 = 200, 500
        self.lineEdit_experiment.setText('aCG')
        self.cw1, self.cw2 = 100, 500
        self.start_time = None
        self.state_start_time = 0
        self.board = None
        self.board = None  # Pycboard class instance.
        self.task = None  # Pycboard class instance.
        self.task_hash = None  # Used to check if file has changed.
        self.sm_info = None  # Information about current state machine.
        self.data_dir = None  # data directory
        self.subject_id = None
        self.subject_group = None
        self.subject_subgroup = None
        self.exp_name = None  # experimenter name
        self.project = None
        self.FrameDataLogger = FrameDataLogger
        self.data_logger = Data_logger(print_func=self.print_to_log)
        self.mcu_connected = False  # Whether gui is connected to pyboard.
        self.uploaded = False  # Whether selected task is on board.
        self.subject_changed = False
        self.available_tasks = None
        self.available_ports = None
        self.refresh_interval = 1000  # Interval to refresh tasks and ports when not running (ms).
        # Buttons for connect teh gui to pyboards
        self.pushButton_connect_mcu.clicked.connect(
            lambda: self.disconnect_mcu() if self.mcu_connected else self.connect_mcu())
        # Config button function
        self.pushButton_config_mcu.clicked.connect(lambda: self.Board_config_dialog.exec_())
        self.pushButton_next_stage.clicked.connect(lambda: self.forward_inter_state())
        self.pushButton_forward_to_nextstage.clicked.connect(lambda: self.forward_next_state())
        # Upload button function
        self.pushButton_upload_task.clicked.connect(lambda: self.setup_task())
        # variables button function
        self.pushButton_var_mcu.clicked.connect(lambda x: self.variables_dialog.exec_())
        # Start button function
        self.pushButton_start_mcu.clicked.connect(lambda: self.start_task())
        # Stop button function
        self.pushButton_stop_mcu.clicked.connect(lambda: self.stop_task())
        # Select data directory
        self.pushButton_data_dir.clicked.connect(self.select_data_dir)
        # LineEdit
        self.lineEdit_status.setReadOnly(True)
        self.lineEdit_data_dir.setText(data_dir)
        self.lineEdit_data_dir.textChanged.connect(self.test_data_path)
        self.lineEdit_subid.textChanged.connect(self.test_data_path)
        self.lineEdit_subject_group.textChanged.connect(self.test_data_path)
        self.lineEdit_subject_subgroup.textChanged.connect(self.test_data_path)
        self.comboBox_srport.setEditable(True)
        self.Board_config_dialog = Board_config_dialog(parent=self)  # Create dialogs.
        self.process_timer = QtCore.QTimer()  # Timer to regularly call process_data() during run.
        self.process_timer.timeout.connect(self.process_data)
        self.refresh_timer = QtCore.QTimer()  # Timer to regularly call refresh() when not running.
        self.refresh_timer.timeout.connect(self.refresh)
        # Initial mcu setup.
        self.disconnect_mcu()  # Set initial state as disconnected.
        self.refresh()
        self.refresh_timer.start(self.refresh_interval)
        self._apply_scaling_policies()
        self.disable_widgets()
        self.pushButton_stop_tracking.setEnabled(False)
        self.pushButton_init_tracking.setEnabled(False)
        self.disable_zone_edit_tool()
        self.pushButton_save_zone_config.setEnabled(False)
        self.lineEdit_config_filename.setEnabled(False)
        self.pushButton_load_config.setEnabled(False)
        self.pushButton_create_zone.setEnabled(False)
        self.pushButton_conenct_to_cam.setEnabled(True)
        self.pushButton_disconenct_to_cam.setEnabled(False)
        self.pushButton_load_config.setEnabled(True)
        self.RecordAnnotatedVideo.setEnabled(False)
        self.pushButton_next_stage.setEnabled(False)

    def _apply_scaling_policies(self):
        """Override the most restrictive size policies from the auto-generated
        UI so the window can actually be resized. The .ui file pins the main
        window to a 1004x631 minimum, the video label to Fixed/Fixed at
        480x320, and surrounds the video with vertical expanding spacers that
        steal all extra space. We don't touch widget content — only resize
        behaviour and layout stretch factors.
        """
        # Window: let it shrink and grow freely.
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(QtCore.QSize(800, 500))

        # Video label: become the elastic widget of the left column.
        self.VideoSteamer.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.VideoSteamer.setMinimumSize(QtCore.QSize(320, 240))
        self.VideoSteamer.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.VideoSteamer.setAlignment(QtCore.Qt.AlignCenter)

        # Give the main content row all extra vertical space (header stays put).
        try:
            self.gridLayout.setRowStretch(0, 0)
            self.gridLayout.setRowStretch(1, 1)
        except AttributeError:
            pass
        # Split the central row 60/40 between video panel and control panel.
        try:
            self.horizontalLayout_11.setStretch(0, 3)
            self.horizontalLayout_11.setStretch(2, 2)
        except AttributeError:
            pass
        # Inside the video column, give the video itself the stretch instead
        # of the empty spacer below it.
        try:
            self.verticalLayout_6.setStretch(0, 1)  # VideoSteamer
            self.verticalLayout_6.setStretch(1, 0)  # spacer
        except AttributeError:
            pass

    def set_marker_size(self):
        self.marker_size = self.spinBox.value()

    def _show_frame(self, qimg):
        """Display a QImage on VideoSteamer, scaled to the label's current
        size with aspect ratio preserved. Without this the pixmap is shown at
        native resolution and the label can't shrink below it."""
        pix = QtGui.QPixmap.fromImage(qimg)
        target = self.VideoSteamer.size()
        if target.width() > 1 and target.height() > 1:
            pix = pix.scaled(
                target,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        self.VideoSteamer.setPixmap(pix)

    def disable_zone_edit_tool(self):
        self.spinBox_line_size.setEnabled(False)
        self.label_4.setEnabled(False)
        self.ZoneMoveLeft.setEnabled(False)
        self.ZoneMoveTop.setEnabled(False)
        self.ZoneZoomIn.setEnabled(False)
        self.RotateZone.setEnabled(False)
        self.pushButton_save_zone_config.setEnabled(False)
        self.lineEdit_config_filename.setEnabled(False)
        self.pushButton_create_zone.setEnabled(False)
        self.pushButton_load_config.setEnabled(False)
        self.ShrinkValue.setEnabled(False)
        self.CameraAspect_width.setEnabled(False)
        self.CameraAspect_height.setEnabled(False)

    def enable_zone_edit_tool(self):
        self.ZoneMoveLeft.setEnabled(True)
        self.ZoneMoveTop.setEnabled(True)
        self.ZoneZoomIn.setEnabled(True)
        self.RotateZone.setEnabled(True)
        self.pushButton_save_zone_config.setEnabled(True)
        self.lineEdit_config_filename.setEnabled(True)
        self.pushButton_create_zone.setEnabled(True)
        self.pushButton_load_config.setEnabled(True)
        self.ShrinkValue.setEnabled(True)
        self.CameraAspect_width.setEnabled(True)
        self.CameraAspect_height.setEnabled(True)

    def calculate_velocity(self):
        x1, y1 = self.prev_coord
        x2, y2 = self.curr_coord
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if x1 != 0 and y1 != 0:
            if dx == 0:
                distance = dy
            elif dy == 0:
                distance = dx
            else:
                distance = math.sqrt(dx ** 2 + dy ** 2)
            self.velocity = 0
            if distance > self.thres_movement:
                self.velocity = (distance * (self.pixel_per_cm / 100)) / self.frame_time
        else:
            self.velocity = 0

    def load_config_file(self):
        self.config_file_pathname, _ = QFileDialog.getOpenFileName(self, "Open Files", ".json", "Config Files (*.json)")
        self.lineEdit_config_filename.setText(self.config_file_pathname)
        self.pushButton_save_zone_config.setEnabled(True)
        if self.config_file_pathname:
            self.EditZoneConfig.read_config_file(self, path=self.config_file_pathname)
            self.EditZoneConfig.coords_shapely_object(self)
            # Apply the arena crop window ONLY if the config defines one.
            # If there's no arena block we keep whatever ch1..cw2 already are
            # (the constructor defaults, or whatever a previous load set), so
            # DLC keeps seeing the same crop the zones were drawn in.
            if self.arena_crop:
                self.ch1, self.ch2, self.cw1, self.cw2 = self.arena_crop
            self.lineEdit_notification.setText('')
            # Block signals while resetting so we don't trigger 4 rebuilds.
            for widget, val in (
                (self.ZoneMoveLeft, 0),
                (self.ZoneMoveTop, 0),
                (self.ZoneZoomIn, 0),
                (self.RotateZone, 50),
            ):
                widget.blockSignals(True)
                widget.setValue(val)
                widget.blockSignals(False)
            self.EditZoneConfig.reset_edits(self)
            self.spinBox_line_size.setEnabled(True)
            self.label_4.setEnabled(True)
            self.enable_zone_edit_tool()
            self.config_loaded = True
        else:
            self.lineEdit_notification.setText('Select valid config json file!!!!')

    def init_tracking(self):
        """Begin tracking. Opens the per-frame TXT log when a subject-ID is
        set (independent of the AVI checkbox); arms the AVI writer so it can
        be created at the first frame with the actual cropped frame size.
        Safe to call repeatedly — no-op if already tracking. Camera must be
        connected; otherwise we display a notification and bail out."""
        if self.track_subject:
            return
        if self.cap is None or not self.cap.isOpened():
            self.lineEdit_notification.setText('Connect a camera before tracking.')
            return

        has_subid = bool(self.lineEdit_subid.text().strip())

        # TXT log: opens whenever a subject-ID is present. Independent of AVI.
        if has_subid:
            self.exp_name = self.lineEdit_experiment.text()
            self.experimenter_name = self.lineEdit_experimenter.text()
            self.subject_id = self.lineEdit_subid.text()
            self.subject_group = self.lineEdit_subject_group.text()
            self.subject_subgroup = self.lineEdit_subject_subgroup.text()
            data_dir = self.lineEdit_data_dir.text()
            croping_details = [self.ch1, self.ch2, self.cw1, self.cw2]
            self.FrameDataLogger.open_data_file(
                self, data_dir, self.exp_name, self.experimenter_name,
                self.subject_id, self.subject_group, self.subject_subgroup,
                self.json_config_file, croping_details,
            )
            self.write_video_info = True

        # AVI writer: deferred until first frame so we know real dimensions.
        # The legacy code hard-coded 640x480 which silently dropped every
        # frame whose source resolution differed (e.g. 1280x720 cameras).
        self._avi_pending = self.RecordNormalVideo.isChecked() and has_subid
        self.record_video = False  # flipped to True once VideoWriter is open

        self.tracking_frame = 0
        self.time_stamps = 0
        self.track_subject = True
        self.disable_zone_edit_tool()
        self.pushButton_init_tracking.setEnabled(False)
        self.pushButton_stop_tracking.setEnabled(True)
        self.pushButton_load_config.setEnabled(False)
        self.pushButton_create_zone.setEnabled(False)
        self.pushButton_disconenct_to_cam.setEnabled(False)

    def stop_tracking(self):
        """End tracking and close any files that init_tracking opened.
        Idempotent — safe to call when not tracking."""
        if not self.track_subject:
            return
        self.track_subject = False
        self.write_video_info = False
        self.record_video = False
        self._avi_pending = False

        self.enable_zone_edit_tool()
        self.pushButton_stop_tracking.setEnabled(False)
        self.pushButton_init_tracking.setEnabled(True)
        self.pushButton_load_config.setEnabled(True)
        self.pushButton_create_zone.setEnabled(True)
        self.pushButton_disconenct_to_cam.setEnabled(True)

        try:
            self.FrameDataLogger.close_files(self)
        except Exception:
            pass
        if self.output is not None:
            try:
                self.output.release()
            except Exception:
                pass
            self.output = None

    # General methods
    def print_to_log(self, print_string, end='\n'):
        self.LiveUpdateLine.moveCursor(QtGui.QTextCursor.End)
        self.LiveUpdateLine.insertPlainText(print_string + end)
        self.LiveUpdateLine.moveCursor(QtGui.QTextCursor.End)
        self.LiveUpdateLine.repaint()
        pass

    def clear_task_combo_box(self):
        self.comboBox_task.clear()

    def clear_srport_combo_box(self):
        self.comboBox_srport.clear()

    def add_to_srport_combo_box(self, ports):
        self.comboBox_srport.addItems(sorted(ports))

    def enable_srport_combo_box(self, state):
        self.comboBox_srport.setEnabled(state)

    def enable_task_combo_box(self, state):
        self.comboBox_task.setEnabled(state)

    def combo_box_add_items(self, tasks):
        self.comboBox_task.addItems(sorted(tasks))

    # Disable or Enable pussButtons when GUI opened
    def disable_widgets(self):
        # all subID line are enabled
        self.lineEdit_subid.setEnabled(True)
        self.pushButton_start_mcu.setEnabled(False)
        # all stop buttons are disabled
        self.pushButton_stop_mcu.setEnabled(False)
        # all upload buttons are disabled
        self.pushButton_upload_task.setEnabled(False)
        self.comboBox_task.setEnabled(False)  # task drop box are disabled
        self.pushButton_var_mcu.setEnabled(False)  # variables buttons are disabled
        self.pushButton_config_mcu.setEnabled(False)  # config buttons are disabled
        self.lineEdit_status.setEnabled(False)  # status line are disabled
        self.comboBox_srport.setEnabled(False)  # serial port are disabled
        self.pushButton_connect_mcu.setEnabled(False)  # connect buttons are disabled
        self.pushButton_data_dir.setEnabled(True)
        self.comboBox_srport.setEnabled(True)
        self.pushButton_connect_mcu.setEnabled(True)
        self.pushButton_pause_experiment.setEnabled(False)
        self.pushButton_forward_to_nextstage.setEnabled(False)
        self.pushButton_restart.setEnabled(False)

    # Disable or Enable pussButtons when GUI opened
    def enable_widgets(self, com_list):
        if len(com_list) >= 1:
            self.comboBox_srport.setEnabled(True)
            self.pushButton_connect_mcu.setEnabled(True)

    def test_data_path(self):
        # Checks whether data dir and subject ID are valid.
        self.data_dir = self.lineEdit_data_dir.text()
        subject_id = self.lineEdit_subid.text()
        if os.path.isdir(self.data_dir) and subject_id:
            self.pushButton_start_mcu.setText('RECORD')
            return True
        else:
            self.pushButton_start_mcu.setText('START')

    def scan_ports(self):
        # Scan serial ports for connected boards and update ports list if changed.
        ports = set([c[0] for c in list_ports.comports()
                     if ('Pyboard' in c[1]) or ('USB Serial Device' in c[1])])
        port_list = list(ports)
        if not ports == self.available_ports:
            self.clear_srport_combo_box()
            self.enable_widgets(port_list)
            self.add_to_srport_combo_box(ports)
            self.available_ports = ports

    def scan_tasks(self):
        # Scan task folder for available tasks and update tasks list if changed.
        tasks = set([t.split('.')[0] for t in os.listdir(tasks_dir)
                     if t[-3:] == '.py'])
        if not tasks == self.available_tasks:
            self.clear_task_combo_box()
            self.combo_box_add_items(sorted(tasks))
            self.available_tasks = tasks
        if self.task:
            try:
                task = self.comboBox_task.currentText()
                task_path = os.path.join(tasks_dir, task + '.py')
                if not self.task_hash == _djb2_file(task_path):  # Task file modified.
                    self.task_changed()
            except FileNotFoundError:
                pass

    def task_changed(self):
        self.uploaded = False
        self.pushButton_upload_task.setText('Upload')
        self.pushButton_start_mcu.setEnabled(False)

    def connect_mcu(self):
        try:
            self.lineEdit_status.setText('Connecting...')
            self.pushButton_stop_mcu.setEnabled(False)
            self.lineEdit_subid.setEnabled(False)
            self.pushButton_var_mcu.setEnabled(False)
            self.comboBox_srport.setEnabled(False)
            self.pushButton_connect_mcu.setEnabled(False)
            self.repaint()
            self.board = Pycboard(self.comboBox_srport.currentText(), print_func=self.print_to_log,
                                  data_logger=self.data_logger)
            if not self.board.status['framework']:
                self.board.load_framework()
            self.mcu_connected = True
            self.pushButton_config_mcu.setEnabled(True)
            self.pushButton_connect_mcu.setEnabled(True)
            self.comboBox_task.setEnabled(True)
            self.pushButton_upload_task.setEnabled(True)
            self.lineEdit_subid.setEnabled(True)
            self.pushButton_connect_mcu.setText('Disconnect')
            self.lineEdit_status.setText('Connected')
        except SerialException:
            self.lineEdit_status.setText('Connection failed')
            self.pushButton_connect_mcu.setEnabled(True)

    def disconnect_mcu(self):
        # Disconnect from pyboard.
        if self.board: self.board.close()
        self.board = None
        self.mcu_connected = False
        self.pushButton_config_mcu.setEnabled(False)
        self.pushButton_var_mcu.setEnabled(False)
        self.comboBox_task.setEnabled(False)
        self.pushButton_start_mcu.setEnabled(False)
        self.pushButton_stop_mcu.setEnabled(False)
        self.pushButton_upload_task.setEnabled(False)
        self.lineEdit_subid.setEnabled(False)
        self.enable_srport_combo_box(True)
        self.pushButton_connect_mcu.setText('Connect')
        self.lineEdit_status.setText('Not connected')
        self.lineEdit_status.setEnabled(False)
        self.frame_for_mcu = 0

    def disconnect_camera(self):
        self.cam_capture_timer.stop()
        self.pushButton_stop_tracking.setEnabled(False)
        self.pushButton_init_tracking.setEnabled(False)
        self.pushButton_conenct_to_cam.setEnabled(True)
        self.pushButton_disconenct_to_cam.setEnabled(False)
        self.disable_zone_edit_tool()
        self.pushButton_load_config.setEnabled(True)
        try:
            self.output.release()
        except:
            pass

    def connect_camera(self):
        self.pushButton_init_tracking.setEnabled(True)
        self.initial_camera_setup()
        self.pushButton_conenct_to_cam.setEnabled(False)
        self.pushButton_disconenct_to_cam.setEnabled(True)
        self.lineEdit_config_filename.setEnabled(True)
        self.pushButton_load_config.setEnabled(True)
        self.pushButton_create_zone.setEnabled(True)
        if not str(self.lineEdit_cameraID.text()):
            self.lineEdit_cameraID.setText('0')
            cam_ID = 0
        elif self.lineEdit_cameraID.text().startswith('v'):
            cam_ID = path_video
        else:
            cam_ID = int(self.lineEdit_cameraID.text())
        try:
            if not self.cam_capture_timer.isActive():
                self.init_time = time.time()
                self.cap = cv2.VideoCapture(cam_ID)
                #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                #.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cam_capture_timer.start(65)
            else:
                self.cam_capture_timer.stop()
                self.cap.release()
                self.output.release()
                cv2.destroyAllWindows()
        except error as e:
            print(e)

    def status_update(self, msg):
        self.lineEdit_status.setText(msg)

    def setup_task(self):
        try:
            task = self.comboBox_task.currentText()
            if self.uploaded:
                self.lineEdit_status.setText('Resetting task..')
            else:
                self.lineEdit_status.setText('Uploading..')
                self.task_hash = _djb2_file(os.path.join(tasks_dir, task + '.py'))
            self.pushButton_start_mcu.setEnabled(False)
            self.pushButton_var_mcu.setEnabled(False)
            self.repaint()
            self.sm_info = self.board.setup_state_machine(task, uploaded=self.uploaded)
            self.variables_dialog = Variables_dialog(self)
            self.pushButton_var_mcu.setEnabled(True)
            # self.task_plot.set_state_machine(self.sm_info)
            self.pushButton_start_mcu.setEnabled(True)
            self.lineEdit_subid.setEnabled(True)
            self.pushButton_stop_mcu.setEnabled(False)
            self.lineEdit_status.setText('Uploaded : ' + task)
            self.task = task
            self.uploaded = True
            self.pushButton_upload_task.setText('Reset')
        except PyboardError:
            self.lineEdit_status.setText('Error setting up state machine.')

    def select_data_dir(self):
        self.lineEdit_data_dir.setText(QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder'))

    def start_task(self):
        # experiment_name, experimenter_name, subject_ID, subject_group,subject_subgroup
        self.frame_for_mcu = 0
        self.exp_name = str(self.lineEdit_experiment.text())
        self.experimenter_name = str(self.lineEdit_experimenter.text())
        self.subject_group = str(self.lineEdit_subject_group.text())
        self.subject_subgroup = str(self.lineEdit_subject_subgroup.text())
        if self.test_data_path():
            self.subject_id = str(self.lineEdit_subid.text())
            self.data_logger.open_data_file(self.data_dir, self.exp_name, self.experimenter_name, self.subject_id,
                                            self.subject_group,
                                            self.subject_subgroup)
        self.board.start_framework()
        self.start_time = time.time()
        self.comboBox_task.setEnabled(False)
        self.pushButton_upload_task.setEnabled(False)
        self.pushButton_start_mcu.setEnabled(False)
        self.pushButton_config_mcu.setEnabled(False)
        self.pushButton_stop_mcu.setEnabled(True)
        self.pushButton_connect_mcu.setEnabled(False)
        self.lineEdit_subid.setEnabled(True)
        self.mcu_running = True
        self.tracking_frame = 0
        self.time_stamps = 0
        print('\nRun started at: {}\n'.format(datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
        self.process_timer.start(update_interval)
        self.refresh_timer.stop()
        self.lineEdit_status.setText('Running: ' + self.task)
        self.pushButton_pause_experiment.setEnabled(True)
        self.pushButton_next_stage.setEnabled(True)
        self.pushButton_forward_to_nextstage.setEnabled(True)
        self.pushButton_restart.setEnabled(True)
        # Auto-start tracking so coords reach the MCU and the per-frame log
        # is written without the user having to hit Init-tracking separately.
        self.init_tracking()

    def stop_task(self, error=False, stopped_by_task=False):
        # Auto-stop tracking so files close cleanly and we stop sending coords
        # to a board whose framework just halted.
        if self.track_subject:
            self.stop_tracking()
        self.process_timer.stop()
        self.refresh_timer.start(self.refresh_interval)
        if not (error or stopped_by_task):
            self.board.stop_framework()
            QtCore.QTimer.singleShot(100, self.process_data)  # Catch output after framework stops.
        self.data_logger.close_files()
        self.pushButton_start_mcu.setEnabled(True)
        self.pushButton_connect_mcu.setEnabled(True)
        self.comboBox_task.setEnabled(True)
        self.pushButton_upload_task.setEnabled(True)
        self.pushButton_stop_mcu.setEnabled(False)
        self.mcu_running = False
        self.frame_for_mcu = 0
        self.pushButton_pause_experiment.setEnabled(False)
        self.lineEdit_status.setText('Uploaded : ' + self.task)
        self.pushButton_pause_experiment.setEnabled(False)
        self.pushButton_next_stage.setEnabled(False)
        self.pushButton_forward_to_nextstage.setEnabled(False)
        self.pushButton_restart.setEnabled(False)

    # Timer updates
    def process_data(self):
        # Called regularly during run to process data from board.
        try:
            new_data = self.board.process_data()
            # update timer here
            run_time = time.time() - self.start_time
            run_time = str(timedelta(seconds=run_time))[:7]
            self.lcdNumber_timer.display(run_time)
            self.lineEdit_warning_notification(new_data)
            if not self.board.framework_running:
                self.stop_task(stopped_by_task=True)
                self.mcu_running = False
                self.pushButton_pause_experiment.setEnabled(False)
                self.pushButton_next_stage.setEnabled(False)
                self.pushButton_forward_to_nextstage.setEnabled(False)
                self.pushButton_restart.setEnabled(False)
        except PyboardError as e:
            self.print_to_log('\nError during framework run.')
            self.stop_task(error=True)

    def continue_state(self):
        pass
        #if self.mcu_running and not self.experiment_paused:
            #self.board.set_coordinates('pause_state', 1)
            #self.pushButton_pause_experiment.setText('Expt paused')
            #self.pushButton_restart.setText('Restart')
            #self.pushButton_pause_experiment.setEnabled(False)
            #self.pushButton_restart.setEnabled(True)
            #self.experiment_paused = True
        #elif self.mcu_running and self.experiment_paused:
            #self.board.set_coordinates('pause_state', 2)
            #self.pushButton_pause_experiment.setText('Pause')
            #self.pushButton_restart.setText('Restarted')
            #self.pushButton_pause_experiment.setEnabled(True)
            #self.pushButton_restart.setEnabled(False)
            #self.experiment_paused = False

    def forward_inter_state(self):
        if self.mcu_running:
            self.board.set_coordinates('pass_inter_state', 1)
        self.pushButton_next_stage.setEnabled(False)

    def forward_next_state(self):
        reply = QMessageBox.question(self, 'Transfer to next stage',
                                     'Are you sure you want to pass next stage?                           '
                                     '                                                                     '
                                     ' IF YOU PRESS YES, THEN WAIT FOR FEW SECONDS BEFORE PRESS NEXT STAGE AGAIN',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if self.mcu_running and reply == QMessageBox.Yes:
            self.board.set_coordinates('forward_next_state', 1)
        else:
            self.board.set_coordinates('forward_next_state', 0)
            pass

    def lineEdit_warning_notification(self, new_data):
        if new_data:
            for value in new_data:
                for string in value:
                    if str(string).startswith('inter'):
                        self.pushButton_next_stage.setEnabled(True)
                    elif str(string).startswith('Warning'):
                        self.lineEdit_notification.setText(str(string))
                    elif str(string).startswith('State'):
                        self.state_start_time = time.time()
                        self.state_name = str(string)
                    elif str(string).startswith('Exit'):
                        self.state_start_time = 0
                        self.state_name = str('')
                    else:
                        self.lineEdit_notification.setText(" ")
                        self.pushButton_next_stage.setEnabled(False)

    def refresh(self):
        # Called regularly when not running to update tasks and ports.
        self.scan_tasks()
        self.scan_ports()

    def scale_video_image(self):
        height, width, _ = self.image.shape
        if self.ShrinkValue.value() > 0:
            self.scaling_width = int((height * self.ShrinkValue.value()) / 100)
            self.scaling_height = int((width * self.ShrinkValue.value()) / 100)

    def initial_camera_setup(self):
        self.aspect_width = self.CameraAspect_width.value()
        self.aspect_height = self.CameraAspect_height.value()

    def video_streamer(self):
        ret, self.image = self.cap.read()
        if ret:
            self.cropped_image = self.image[self.ch1:self.ch2, self.cw1:self.cw2, :]
            # self.cropped_image = self.image[0:400, 150:640, :]
            self.new_frame_time = time.time()
            self.frame_time = (self.new_frame_time - self.prev_frame_time)
            fps = 1 / self.frame_time
            self.prev_frame_time = self.new_frame_time
            self.init_height, self.init_width, _ = self.cropped_image.shape
            # shrink image if necessary
            self.scaling_height = int((self.init_height * self.ShrinkValue.value()) / 100)
            self.scaling_width = int((self.init_width * self.ShrinkValue.value()) / 100)
            self.cropped_image = cv2.resize(self.cropped_image, (self.scaling_width, self.scaling_height),
                                            interpolation=cv2.INTER_AREA)
            # conver RBG to BGR
            self.cropped_image = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2RGB)
            for i in self.zone_list:
                self.cropped_image = cv2.polylines(self.cropped_image, i, isClosed=True, color=(150, 50, 255),
                                                   thickness=self.spinBox_line_size.value(), lineType=cv2.LINE_AA)
            height, width, channel = self.cropped_image.shape
            step = channel * width
            self.n_frame += 1
            if self.n_frame == 1:
                print("initiating dlc inference might take while do not close window")
                dlc_live.init_inference(self.cropped_image)
            # Min size is set once in _apply_scaling_policies(); resizing it
            # every frame to the image dimensions would prevent the window
            # from being shrunk smaller than the camera output.
            cv2.putText(self.cropped_image, str("FPS %.1f" % fps), (2, 25), cv2.FONT_HERSHEY_PLAIN, 0.5,
                        (170, 255, 127), cv2.LINE_4)
            if self.track_subject:
                self.time_stamps += self.frame_time * 1000
                array = dlc_live.get_pose(self.cropped_image).tolist()
                head, center, tailbase = array[0:3]
                if self.config_loaded:
                    self.tracking_frame += 1
                    self.location_head = None
                    self.location_center = None
                    if head[2] > self.thresh_detct_ci:
                        cv2.circle(self.cropped_image, (int(head[0]), int(head[1])), self.marker_size, (150, 50, 255), -1)
                    if center[2] > self.thresh_detct_ci:
                        cv2.circle(self.cropped_image, (int(center[0]), int(center[1])), self.marker_size,
                                   (1, 190, 200), -1)
                        self.curr_coord = int(center[0]), int(center[1])
                        self.calculate_velocity()
                    if tailbase[2] > self.thresh_detct_ci:
                        cv2.circle(self.cropped_image, (int(tailbase[0]), int(tailbase[1])), self.marker_size,
                                   (255, 128, 0), -1)
                    # Find the location of the mouse
                    if head[2] > self.thresh_detct_ci:
                        for index, zone in enumerate(self.polygons_list):
                            if zone.contains(Point(head[0], head[1])):
                                self.location_head = self.zone_list_key[index]

                    if center[2] > self.thresh_detct_ci:
                        for index, zone in enumerate(self.polygons_list):
                            if zone.contains(Point(center[0], center[1])):
                                self.location_center = self.zone_list_key[index]

                    cv2.putText(self.cropped_image, str(self.state_name), (width - 400, 17), cv2.FONT_HERSHEY_PLAIN,
                                1.8, (37, 255, 150), cv2.LINE_4)
                    cv2.putText(self.cropped_image, str("spd %.2f m/s" % self.velocity), (0, height - 5),
                                cv2.FONT_HERSHEY_PLAIN, 0.5, (170, 255, 127), cv2.LINE_4)
                    cv2.putText(self.cropped_image, str('head :-{}'.format(self.location_head)), (width - 300, height - 40),
                                cv2.FONT_HERSHEY_PLAIN, 0.5, (200, 255, 180), cv2.LINE_4)
                    cv2.putText(self.cropped_image, str('body :-{}'.format(self.location_center)), (width - 300, height - 10),
                                cv2.FONT_HERSHEY_PLAIN, 0.5, (170, 255, 160), cv2.LINE_4)
                    # Always update prev_coord so velocity stays consistent
                    # between frames, regardless of whether we're recording.
                    self.prev_coord = self.curr_coord

                    # ---- MCU coords: independent of file recording ----
                    if self.mcu_running and self.board is not None:
                        try:
                            self.board.set_coordinates('loc_head', self.location_head)
                            self.board.set_coordinates('loc_center', self.location_center)
                            self.board.set_coordinates('speed', self.velocity)
                            self.board.set_coordinates('new_frame', 1)
                            self.frame_for_mcu += 1
                            if self.state_start_time != 0:
                                _state_time = time.time() - self.state_start_time
                                cv2.putText(self.cropped_image,
                                            str(str(timedelta(seconds=_state_time))[2:10]),
                                            (width - 175, 50), cv2.FONT_HERSHEY_PLAIN,
                                            1.8, (255, 255, 255), cv2.LINE_4)
                        except Exception as exc:
                            print('set_coordinates failed:', exc)

                    # ---- Per-frame TXT log: only needs the file to be open ----
                    if self.write_video_info:
                        self.FrameDataLogger.write_to_file(
                            self, self.tracking_frame, self.frame_for_mcu,
                            self.state_name, round(self.time_stamps),
                            self.velocity, self.location_head, array,
                        )

                    # ---- AVI: lazy-create at first frame with real dims ----
                    if getattr(self, '_avi_pending', False):
                        try:
                            os.makedirs('pymaze_videos', exist_ok=True)
                            avi_name = (str(self.lineEdit_subid.text())
                                        + datetime.now().strftime('-%Y-%m-%d-%H%M%S')
                                        + '.avi')
                            avi_path = os.path.join('pymaze_videos', avi_name)
                            src_h, src_w = self.image.shape[:2]
                            self.output = cv2.VideoWriter(
                                avi_path, cv2.VideoWriter_fourcc(*'XVID'),
                                15, (src_w, src_h),
                            )
                            if self.output.isOpened():
                                self.record_video = True
                            else:
                                print('VideoWriter failed to open:', avi_path)
                                self.output = None
                        except Exception as exc:
                            print('VideoWriter init failed:', exc)
                            self.output = None
                        self._avi_pending = False

                    if self.record_video and self.output is not None:
                        self.output.write(self.image)

                    self.qImg = QImage(self.cropped_image.data, width, height, step, QImage.Format_RGB888)
                    self._show_frame(self.qImg)
                else:
                    if head[2] > self.thresh_detct_ci:
                        cv2.circle(self.cropped_image, (int(head[0]), int(head[1])), self.marker_size, (150, 50, 255),
                                   -1)
                    if center[2] > self.thresh_detct_ci:
                        cv2.circle(self.cropped_image, (int(center[0]), int(center[1])), self.marker_size,
                                   (1, 190, 200),
                                   -1)
                    if tailbase[2] > self.thresh_detct_ci:
                        cv2.circle(self.cropped_image, (int(tailbase[0]), int(tailbase[1])), self.marker_size,
                                   (255, 128, 0), -1)
                    self.qImg = QImage(self.cropped_image.data, width, height, step, QImage.Format_RGB888)
                    self._show_frame(self.qImg)

            else:
                self.qImg = QImage(self.cropped_image.data, width, height, step, QImage.Format_RGB888)
                self._show_frame(self.qImg)
        else:
            font = QtGui.QFont()
            font.setPointSize(15)
            font.setBold(True)
            font.setWeight(100)
            self.VideoSteamer.setStyleSheet("color: rgb(170, 110, 0);")
            self.VideoSteamer.setFont(font)
            self.VideoSteamer.setText('Streaming interrupted check camera or video files!!!!!!!!!')
            try:
                self.output.release()
            except:
                pass

    def closeEvent(self, event):
        # Called when GUI window is closed.
        if self.board:
            self.board.stop_framework()
            self.board.close()
        if self.cam_capture_timer.isActive():
            self.output.release()
            self.cam_capture_timer.stop()
        event.accept()


# --------------------------------------------------------------
#                              Main
# --------------------------------------------------------------
if __name__ == '__main__':
    try:
        print('Starting Application')
        app = QtWidgets.QApplication(sys.argv)
        gui_app = StartMaze()
        gui_app.show()
        app.exec_()
    except RuntimeError as error:
        print('-' * 150)
        print(error)
        print('-' * 150)
    except BaseException as error:
        print('-' * 150)
        print(error)
        print('-' * 150)
    finally:
        print('Exiting Application')

# ------------------------------ END --------------------------
