"""
Infrared Structure Sensor recorder
@author: Keny ORDAZ
@date: 2023.11.28
"""

import datetime
import enum
import pathlib
import time
import sys
import threading
import queue
import multiprocessing

import cv2 as cv
import numpy as np

try:
    from primesense import openni2
except:
    print('No sensor support')


try:
    from PyQt5.QtCore import (
        QSize, Qt, QThread,
        pyqtSignal as Signal, pyqtSlot as Slot,
        QSettings, QPoint,
        QTimer,
    )

    from PyQt5.QtGui import (
        QKeySequence,
        QImage,
        QPixmap,
        QIcon,
        QFont,
        QIntValidator,
    )

    from PyQt5.QtWidgets import (
        QApplication,
        QAction,
        QFileDialog,
        QLayout,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QSizePolicy,
        QSlider,
        QStyle,
        QVBoxLayout,
        QWidget,
        QStatusBar,
    )
except:
    from PySide6.QtCore import (
        QSize, Qt, QThread,
        Signal, Slot,
        QSettings, QPoint,
        QTimer,
    )

    from PySide6.QtGui import (
        QKeySequence,
        QAction,
        QImage,
        QPixmap,
        QIcon,
        QFont,
        QIntValidator,
    )

    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QLayout,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QSizePolicy,
        QSlider,
        QStyle,
        QVBoxLayout,
        QWidget,
        QStatusBar,
    )
# agregar barra espaciadora para grabar... [s]top

class SensorVideoMode(enum.IntEnum):
    SMALL_320x240x30 = 0
    FAST_320x240x60 = 2
    MEDIUM_640x320x30 = 4


class RecordingSession:
    def __init__(self, location, timestamp):
        self.location = location
        self.timestamp = timestamp

class Clock(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.setInterval(100)
        self.tic = None

    def start(self):
        self.timer.start()
        self.tic = datetime.datetime.now()

    def stop(self):
        self.timer.stop()

    def update(self):
        self.toc = datetime.datetime.now()
        ellapsed = self.toc - self.tic
        ellapsed /= datetime.timedelta(seconds=1)
        msg = f"{ellapsed:.1f} s"
        self.setText(msg)

    def reset(self):
        self.setText("0.0 s")


class WebCamThread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.cap = None

    def run(self):
        print('Running thread')
        self.cap = cv.VideoCapture(0)
        while self.status:
            # cascade = cv2.CascadeClassifier(self.trained_file)
            ret, frame = self.cap.read()
            if not ret:
                continue
            if self.currentThread().isInterruptionRequested():
                self.cap.release()
                break
            # Reading frame in gray scale to process the pattern
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # detections = cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
            #                                      minNeighbors=5, minSize=(30, 30))
            detections = []
            # Drawing green rectangle around the pattern
            for (x, y, w, h) in detections:
                pos_ori = (x, y)
                pos_end = (x + w, y + h)
                color = (0, 255, 0)
                cv.rectangle(frame, pos_ori, pos_end, color, 2)

            # Reading the image in RGB to display it
            color_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Creating and scaling QImage
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)

            # Emit signal
            self.updateFrame.emit(scaled_img)
        print("Sensor stream stopped")

class StructureSensorThread(QThread):
    updateFrame = Signal(QImage)
    linux = "/usr/lib/x86_64-linux-gnu"
    initialized = False
    video_mode = SensorVideoMode.FAST_320x240x60

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        if not self.initialized:
            self._init_sensor()

    @classmethod
    def _init_sensor(cls):
        openni2.initialize(cls.linux)
        cls.dev = openni2.Device.open_any()
        cls.ir_video_modes = cls.dev.get_sensor_info(openni2.SENSOR_IR).videoModes
        cls.cr_video_modes = cls.dev.get_sensor_info(openni2.SENSOR_DEPTH).videoModes


        cls.ir_stream = cls.dev.create_ir_stream()
        cls.cr_stream = cls.dev.create_depth_stream()

        cls.ir_stream.set_video_mode(cls.ir_video_modes[cls.video_mode])
        cls.cr_stream.set_video_mode(cls.cr_video_modes[cls.video_mode])

        cls.initialized = True
        print('Sensor ready')

    @classmethod
    def _unload_sensor(cls):
        if cls.initialized:
            cls.ir_stream.stop()
            cls.cr_stream.stop()
            openni2.unload()
            cls.initialized = False
            print('Sensor unloaded')
        else:
            print('Already unloaded')

    @staticmethod
    def img_16bit_2_8bit(image, power=10):
        assert image.dtype == np.uint16
        max_in_sample = 2**power - 1
        max_out_sample = 2**8 - 1
        assert image.max() < max_in_sample
        tmp = image
        # print(tmp.max(), tmp.min())
        tmp = image * (max_out_sample / max_in_sample) + 0.5
        # print(tmp.max(), tmp.min())
        tmp = np.floor(tmp)
        # print(tmp.max(), tmp.min())
        tmp[tmp > 255] = 255
        tmp[tmp < 0] = 0
        tmp = tmp.astype('uint8')
        # print(tmp.max(), tmp.min())
        return tmp

    @staticmethod
    def frame_to_image(frame, to_uint8=True):
        tst = frame.timestamp
        frame_buffer = frame.get_buffer_as_uint16()
        img = np.frombuffer(frame_buffer, dtype=np.uint16)
        img = img.reshape(frame.height, frame.width)
        img = np.copy(img)
        if to_uint8:
            img = StructureSensorThread.img_16bit_2_8bit(img)
        return img, tst 

    def run(self):
        print('Running thread')
        self.ir_stream.start()
        self.cr_stream.start()
        while self.status:
            if self.currentThread().isInterruptionRequested():
                self._unload_sensor()
                break
            stream = openni2.wait_for_any_stream([self.ir_stream, self.cr_stream], 0.2)
            frame, tst = self.frame_to_image(self.ir_stream.read_frame())

            # Creating and scaling QImage
            h, w = frame.shape
            #img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
            img = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
            # scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)

            self.updateFrame.emit(img)
        print("Sensor stream stopped")


class ControlWidget(QWidget):
    icon_size = QSize(64, 64)
    default_path = pathlib.Path.home()
    warning = Signal(str)
    id_pattern = "SJ_{:03}"
    stopping = Signal(datetime.datetime)
    recording_session = Signal(RecordingSession)
    q = queue.Queue()

    def __init__(self, parent):
        self.recording_folder = None
        self.is_recording = False
        self.frame_idx = 0
        super().__init__(parent)
        self.btn_rec = QPushButton("Grabar")
        self.btn_stop = QPushButton("Parar")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_rec)
        btn_layout.addWidget(self.btn_stop)

        icon = QIcon.fromTheme("media-record")
        self.btn_rec.setIcon(icon)
        self.btn_rec.setIconSize(self.icon_size)
        self.btn_rec.setEnabled(False)
        self.btn_rec.clicked.connect(self.on_start_recording)
        self.btn_rec.setShortcut(QKeySequence("Ctrl+R"))

        self.btn_stop.setEnabled(False)
        icon = QIcon.fromTheme("media-playback-stop")
        # icon = self.style().standardIcon(QStyle.SP_MediaStop)
        self.btn_stop.setIcon(icon)
        self.btn_stop.setIconSize(self.icon_size)
        self.btn_stop.clicked.connect(self.on_stop_recording)
        self.btn_stop.setShortcut(QKeySequence("Ctrl+S"))

        self.edt_id = e1 = QLineEdit()
        e1.setMaxLength(3)
        e1.setAlignment(Qt.AlignCenter)
        e1.setFont(QFont("Arial", 20))
        #e1.setInputMask("SJ_99D;0")
        #e1.setInputMask("SJ_99D")
        e1.setValidator(QIntValidator(1, 999))
        #e1.editingFinished.connect(self.valid_id)
        #e1.returnPressed.connect(self.valid_id)
        e1.textEdited.connect(self.enable_recording)
        e1.textChanged.connect(self.enable_recording)

        flo = QFormLayout()
        flo.addRow("ID: ", self.edt_id)

        self.line_edit = QLineEdit(str(self.default_path))
        self.line_edit.setReadOnly(True)
        self.btn_select = QPushButton("Ubicación: ", self)
        self.btn_select.clicked.connect(self.select_recording_folder)
        self.line_edit.textChanged.connect(self.enable_recording)
        loc_layout = QHBoxLayout()
        loc_layout.addWidget(self.btn_select)
        loc_layout.addWidget(self.line_edit)

        self.view = QLabel('sensor', self)
        self.view.setPixmap(QPixmap(640, 480))

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(flo)
        layout.addLayout(loc_layout)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def get_id(self):
        sj_id = self.edt_id.text()
        return sj_id

    def set_id(self, sj_id):
        return self.edt_id.setText(sj_id)

    def valid_id(self, sj_id=None):
        print("in valid", sj_id)
        is_valid = self.edt_id.hasAcceptableInput()
        print(f"{is_valid=}")

    def select_recording_folder(self):
        initial_path = self.get_location()
        #result = QFileDialog.getExistingDirectory(directory=initial_path)
        result = QFileDialog.getExistingDirectory(self, "", initial_path)
        if not result:
            self.warning.emit("Diálogo cancelado.")
            return
        if self.has_valid_location(result):
            self.set_location(result)
        else:
            self.warning.emit(f"La ubicación '{result}' ha sido rechazada.")

    def get_location(self):
        return self.line_edit.text()

    def set_location(self, location):
        return self.line_edit.setText(location)

    def has_valid_location(self, location):
        if not location:
            return False
        location = pathlib.Path(location)
        is_location_ok = location.exists() and location.is_dir() \
                and location.is_relative_to(self.default_path)
        return is_location_ok

    def enable_recording(self):
        is_id_ok = self.edt_id.hasAcceptableInput()
        is_location_ok = self.has_valid_location(self.get_location())
        self.btn_rec.setEnabled(is_id_ok and is_location_ok)

    def get_formatted_id(self):
        return self.id_pattern.format(int(self.get_id()))

    @staticmethod
    def get_formatted_datetime(value):
        return value.strftime("ir_%Y%m%dT%H%M%S")

    def on_start_recording(self):
        location = pathlib.Path(self.get_location())
        subject = self.get_formatted_id()

        now = datetime.datetime.now()
        timestamp = self.get_formatted_datetime(now)
        # Create subject directory and current directory
        # for recording
        subject_folder = location / subject
        subject_folder.mkdir(mode=0o755, exist_ok=True)
        recording_folder = subject_folder / timestamp
        recording_folder.mkdir(mode=0o755, exist_ok=False)
        self.recording_folder = recording_folder
        self.recording_session.emit(
            RecordingSession(recording_folder, now)
        )

        self.btn_rec.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_select.setEnabled(False)
        self.edt_id.setEnabled(False)
        self.is_recording = True
        self.frame_idx = 0

        for _ in range(6):
            threading.Thread(target=self.saver, daemon=True).start()

    def on_stop_recording(self):
        self.is_recording = False
        self.stopping.emit(datetime.datetime.now())
        print(f"{self.frame_idx} frame(s)")
        print(f"{self.q.qsize()} pending frame(s)")
        self.q.join()
        self.warning.emit(f"{self.frame_idx} frame(s)")
        self.btn_rec.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select.setEnabled(True)
        self.edt_id.setEnabled(True)

    @Slot(QImage)
    def set_image(self, image):
        if self.is_recording:
            self.frame_idx += 1
            filename = self.recording_folder / f"frame_{self.frame_idx:06}.png"
            # print("Enqueueing ", filename)
            self.q.put((filename, image))
        if not MainWindow.use_webcam:
            image = image.scaled(640, 480, Qt.KeepAspectRatio)
        self.view.setPixmap(QPixmap.fromImage(image))

    @classmethod
    def saver(cls):
        while True:
            filename, image = cls.q.get()
            result = image.save(str(filename))
            #print("Saving in ", filename, result)
            cls.q.task_done()


class Recorder(QWidget):
    def __init__(self):
        super().__init__()

        # self.th = StructureSensorThread(self)
        self.th = WebCamThread(self)
        #self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.set_image)
        self.data_dir = str(pathlib.Path(__file__).parent)
        self.initUI()

    def initUI(self):

        xpos = 700
        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(self.quitting)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(xpos, 50)

        btn_size = 200, 100
        btn_record = QPushButton('Grabar', self)
        btn_record.clicked.connect(self.start_recording)
        btn_record.resize(*btn_size)
        btn_record.move(xpos, 100)
        self.btn_record = btn_record
        icon = QIcon.fromTheme("media-record")
        self.btn_record.setIcon(icon)
        self.btn_record.setIconSize(QSize(64, 64))
        
        btn_stop = QPushButton('Detener', self)
        btn_stop.clicked.connect(self.stop_recording)
        btn_stop.resize(*btn_size)
        btn_stop.move(xpos, 350)
        self.btn_stop = btn_stop
        btn_stop.setEnabled(False)
        icon = QIcon.fromTheme("media-playback-stop")
        # icon = self.style().standardIcon(QStyle.SP_MediaStop)
        self.btn_stop.setIcon(icon)

        self.view = QLabel('sensor', self)
        self.view.resize(640, 480)
        self.view.move(10, 10)

        self.data_dir_wgt = QLineEdit(self.data_dir, self)
        self.data_dir_wgt.setReadOnly(True)
        self.data_dir_wgt.resize(500, 30)
        self.data_dir_wgt.move(10, 500)

        self.rec_id = RecordingIdentification(self)
        # self.rec_id.resize(500, 30)
        self.rec_id.move(10, 550)

        btn_select = QPushButton('Carpeta...', self)
        btn_select.clicked.connect(self.select_recording_folder)
        btn_select.resize(*btn_size)
        btn_select.move(xpos, 500)
        self.btn_select = btn_select
        btn_select.setEnabled(True)

        # self.setGeometry(300, 300, 950, 750)
        self.setWindowTitle('Infra Red recorder')
        self.show()
        self.th.start()
        QApplication.instance().lastWindowClosed.connect(self.cleanup)

    def select_recording_folder(self):
        result = QFileDialog.getExistingDirectory()
        self.data_dir = result
        self.data_dir_wgt.setText(result)

    def start_recording(self):
        print("Grabando... \u23fa")
        self.btn_stop.setEnabled(True)
        self.btn_record.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.show_image()
        result = QFileDialog.getExistingDirectory()
        print(result)

    def stop_recording(self):
        print("Deteniendo la grabación... \u23f9")
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select.setEnabled(True)
        self.th.requestInterruption()

    def show_image(self):
        #self.image = cv.imread('../data/frame_29.png')
        #self.image = QImage(self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format_RGB888).rgbSwapped()
        #self.view.setPixmap(QPixmap.fromImage(self.image))
        pass

    @Slot(QImage)
    def set_image(self, image):
        self.view.setPixmap(QPixmap.fromImage(image))
        sj_id = self.rec_id.get_id()
        print(f"{sj_id=}")

    def cleanup(self):
        self.th.requestInterruption()
        self.th.wait()

    def quitting(self):
        self.cleanup()
        QApplication.instance().quit()


class MainWindow(QMainWindow):
    company = "cinvestav"
    use_webcam = True

    def __init__(self, appname):
        super().__init__()
        self.appname = appname
        self.setWindowTitle(self.appname)

        # Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.exit_app)

        self.cfg_wgt = ControlWidget(self)
        self.cfg_wgt.warning.connect(self.show_message)
        # self.cfg_wgt.recording.connect(self.start_recording)
        self.cfg_wgt.recording_session.connect(self.start_recording)
        self.cfg_wgt.stopping.connect(self.stop_recording)

        self.setCentralWidget(self.cfg_wgt)

        self.status_bar = QStatusBar()
        self.ctimer = Clock(self.status_bar)
        self.status_bar.addPermanentWidget(self.ctimer)
        self.setStatusBar(self.status_bar)

        self.read_settings()
        QApplication.instance().lastWindowClosed.connect(self.cleanup)

        self.layout().setSizeConstraint(QLayout.SetFixedSize)
        self.statusBar().setSizeGripEnabled(False)
        self.show()

        if self.use_webcam:
            self.th = WebCamThread(self)
        else:
            self.th = StructureSensorThread(self)
        self.th.updateFrame.connect(self.cfg_wgt.set_image)
        self.th.start()

    @Slot()
    def exit_app(self, checked=None):
        self.cleanup()
        self.close()
        QApplication.quit()

    def write_settings(self):
        sj_id = self.cfg_wgt.get_id()
        location = self.cfg_wgt.get_location()
        settings = QSettings(self.company, self.appname)
        # print(settings.fileName())
        settings.beginGroup("MainWindow")
        settings.setValue("size", self.size())
        settings.setValue("pos", self.pos())
        settings.endGroup()

        settings.beginGroup("LastRecording")
        settings.setValue("subject_id", sj_id)
        settings.setValue("location", location)
        settings.endGroup()

    def read_settings(self):
        settings = QSettings(self.company, self.appname)
        settings.beginGroup("MainWindow")
        self.resize(settings.value("size", QSize(640, 400)))
        self.move(settings.value("pos", QPoint(400, 400)))
        settings.endGroup()

        settings.beginGroup("LastRecording")
        self.set_subject_id(settings.value("subject_id", "999"))
        self.set_location(settings.value("location", str(self.cfg_wgt.default_path)))
        settings.endGroup()

        self.show_message("Configuración cargada.", 2000)

    def set_subject_id(self, sj_id):
        self.cfg_wgt.set_id(sj_id)

    def set_location(self, location):
        self.cfg_wgt.set_location(location)

    def cleanup(self):
        self.th.requestInterruption()
        self.write_settings()
        self.th.wait()

    def show_message(self, msg, duration=3000):
        self.status_bar.showMessage(msg, duration)

    def start_recording(self, session):
        self.ctimer.reset()
        self.ctimer.start()
        msg = session.timestamp
        self.show_message(f"Recording session: {session.location}")

    def stop_recording(self, msg):
        self.ctimer.stop()
        # fmsg = ControlWidget.get_formatted_datetime(msg)
        self.show_message(f"Recording stopped: {msg}")


def main():

    #try:
    if True:
        app = QApplication(sys.argv)
        win = MainWindow("recorder")
        #win = Recorder()

        result = app.exec()
        #StructureSensorThread._unload_sensor()
        sys.exit(result)
    #except Exception as exc:
    else:
        print(exc)
        #StructureSensorThread._unload_sensor()


if __name__ == '__main__':
    main()

