import sys
import mss
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QColor
from PIL import Image
import numpy as np


class ScreenSelector(QWidget):
    def __init__(self, lock):
        super().__init__()
        self.lock = lock
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = None
        self.initUI()

    def initUI(self):
        self.setMouseTracking(True)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.5)
        self.setWindowTitle('Select Screen Area')
        self.showFullScreen()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.selection_start = event.globalPos()
            self.selection_rect = None

    def mouseMoveEvent(self, event):
        if self.selection_start and event.buttons() == QtCore.Qt.LeftButton:
            self.selection_rect = QtCore.QRect(self.selection_start, event.globalPos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.selection_rect:
            self.selection_end = event.globalPos()
            self.update()
            self.close()
            threading.Thread(target=self.capture_screen, args=(self.selection_rect,)).start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.selection_rect:
            painter.setPen(QPen(QColor(255, 0, 0), 5, QtCore.Qt.SolidLine))
            painter.drawRect(self.selection_rect.normalized())

        if self.selection_start:
            painter.setPen(QPen(QColor(0, 255, 0), 10, QtCore.Qt.SolidLine))
            painter.drawPoint(self.selection_start)

        if self.selection_end:
            painter.setPen(QPen(QColor(0, 0, 255), 10, QtCore.Qt.SolidLine))
            painter.drawPoint(self.selection_end)

    def capture_screen(self, rect):
        """Capture the selected screen area and save as a PNG file."""
        if rect.isNull() or rect.width() == 0 or rect.height() == 0:
            print("选择的矩形无效。")
            return

        try:
            monitor = {
                "top": int(rect.top()),
                "left": int(rect.left()),
                "width": int(rect.width()),
                "height": int(rect.height())
            }

            with mss.mss() as sct:
                screenshot = sct.grab(monitor)
                screenshot_path = "screenshot.png"
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=screenshot_path)

                # Adjust the size of the screenshot
                img = Image.open(screenshot_path)
                img = img.resize((1920, 1080), Image.Resampling.LANCZOS)
                img.save(screenshot_path)

                print(f"截图已保存并调整大小到 {screenshot_path}")
        finally:
            self.lock.release()  # Release the lock after capturing


def start_screen_capture(lock):
    app = QApplication(sys.argv)
    selector = ScreenSelector(lock)
    selector.show()
    sys.exit(app.exec_())
