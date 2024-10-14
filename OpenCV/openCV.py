import cv2
import numpy as np


def capture_screen(region=None):
    """
    Capture screen without cursor.
    :param region: None or a tuple of (x, y, width, height)
    :return: a screenshot as an image
    """
    screenshot = cv2.VideoCapture(0)
    screenshot.open(0)
    _, frame = screenshot.read()
    screenshot.release()
    if region is not None:
        x, y, w, h = region
        frame = frame[y:y + h, x:x + w]
    return frame
