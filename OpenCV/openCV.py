import sys
import mss
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QColor


class ScreenSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.selection_start = None
        self.selection_rect = None
        self.initUI()

    def initUI(self):
        self.setMouseTracking(True)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)  # Keep window on top
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # Transparent background
        self.setWindowOpacity(0.2)  # Semi-transparent window
        # self.setGeometry(100, 100, 1300, 1200)  # Initial window size
        self.setWindowTitle('Select Screen Area')
        self.showFullScreen()  # Make the window full screen

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
            self.capture_screen(self.selection_rect)
            self.close()

    def paintEvent(self, event):
        # 清除之前的绘制
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 只绘制矩形边框
        if self.selection_rect:
            painter.setPen(QPen(QColor(255, 0, 0), 5, QtCore.Qt.SolidLine))
            painter.drawRect(self.selection_rect.normalized())

        # 让其他区域保持原样
        painter.end()

    # def capture_screen(self, rect):
    #     with mss.mss() as sct:
    #         monitor = {"top": rect.top(), "left": rect.left(), "width": rect.width(), "height": rect.height()}
    #         screenshot = sct.grab(monitor)
    #         img = QtGui.QImage(screenshot.rgb, screenshot.width, screenshot.height, QtGui.QImage.Format_RGB32)
    #         screenshot_path = "screenshot.png"
    #         QtGui.QImageWriter(screenshot_path, "png").write(img)
    #         print(f"Screenshot saved to {screenshot_path}")

    def capture_screen(self, rect):
        print(rect)
        if rect.isNull() or rect.width() == 0 or rect.height() == 0:
            print("选择的矩形无效。")
            return

        try:
            # 直接使用原始 rect 值
            monitor = {
                # "top": int(rect.top()),
                # "left": int(rect.left()),
                # "width": int(rect.width()),
                # "height": int(rect.height())
                "top": 0,
                "left": 0,
                "width": int(rect.width()),
                "height": int(rect.height())
            }
            screenshot = mss.mss().grab(monitor)
            print(screenshot)
            img = QtGui.QImage(screenshot.rgb, screenshot.width, screenshot.height, QtGui.QImage.Format_RGB888)

            screenshot_path = "screenshot.png"
            if img.save(screenshot_path, "PNG"):
                print(f"截图已保存到 {screenshot_path}")
            else:
                print("保存截图失败。")
        except Exception as e:
            print(f"发生错误：{e}")


def main():
    app = QApplication(sys.argv)
    selector = ScreenSelector()
    selector.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
