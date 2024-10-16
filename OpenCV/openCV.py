import sys
import mss
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QColor


class ScreenSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = None
        self.initUI()

    def initUI(self):
        self.setMouseTracking(True)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)  # Keep window on top
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # Transparent background
        self.setWindowOpacity(0.5)  # Semi-transparent window
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
            self.selection_end = event.globalPos()  # 记录终点
            self.update()  # 更新画面，绘制起点和终点
            self.close()
            self.capture_screen(self.selection_rect)  # 调用截图函数

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制矩形边框
        if self.selection_rect:
            painter.setPen(QPen(QColor(255, 0, 0), 5, QtCore.Qt.SolidLine))
            painter.drawRect(self.selection_rect.normalized())

        # 绘制起点
        if self.selection_start:
            painter.setPen(QPen(QColor(0, 255, 0), 10, QtCore.Qt.SolidLine))  # 绿色点
            painter.drawPoint(self.selection_start)

        # 绘制终点
        if self.selection_end:
            painter.setPen(QPen(QColor(0, 0, 255), 10, QtCore.Qt.SolidLine))  # 蓝色点
            painter.drawPoint(self.selection_end)

        painter.end()

    def capture_screen(self, rect):
        """Capture the selected screen area and save as a PNG file."""
        if rect.isNull() or rect.width() == 0 or rect.height() == 0:
            print("选择的矩形无效。")
            return

        try:
            # 使用原始 rect 值
            monitor = {
                "top": int(rect.top()),
                "left": int(rect.left()),
                "width": int(rect.width()),
                "height": int(rect.height())
            }

            with mss.mss() as sct:
                screenshot = sct.grab(monitor)

                # Save the screenshot
                screenshot_path = "screenshot.png"
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=screenshot_path)
                print(f"截图已保存到 {screenshot_path}")
        except Exception as e:
            print(f"发生错误：{e}")


def main():
    app = QApplication(sys.argv)
    selector = ScreenSelector()
    selector.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
