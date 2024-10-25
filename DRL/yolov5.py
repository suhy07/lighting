import torch
import cv2
import numpy as np
import sys
import os
import pathlib

# 设置Windows路径兼容性
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将 YOLOv5 添加到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5')))

# 导入模型类
from models.common import DetectMultiBackend

# 加载 YOLOv5 模型
model_path = r'..\yolov5\runs\train\yolov5_lighting10\weights\best.pt'
model = DetectMultiBackend(weights=model_path, device=device)
model.eval()

# 图片预处理函数
def letterbox(img, new_shape=(1920, 1920), color=(114, 114, 114), auto=True, scaleFill=False, stride=32):
    shape = img.shape[:2]  # 当前形状 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 计算缩放比例
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))  # 新形状 [width, height]
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充
    if auto:  # 根据比例填充
        dw, dh = dw % stride, dh % stride
    dw //= 2  # 除以2，左右填充
    dh //= 2  # 除以2，上下填充
    img = cv2.resize(img, new_unpad)  # 调整图像大小
    new_image = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return new_image

def process_image(image_path, img_size=(1920, 1920)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    img = letterbox(img, new_shape=img_size)
    img = img.transpose(2, 0, 1)  # 转换为[通道, 高度, 宽度]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 归一化到0-1之间
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img

# 特征提取函数
def extract_features(image_path, model, img_size=(1920, 1920)):
    img = process_image(image_path, img_size)
    with torch.no_grad():
        features = model(img, augment=False, visualize=False)[0]  # features为多个尺度的特征图
    return features  # 直接返回特征图

# 主程序
if __name__ == "__main__":
    image_path = r'..\image\frame_0007.jpg'  # 替换为你的图片路径
    features = extract_features(image_path, model, img_size=(1920, 1920))
    print("特征图形状:", features.shape)
    features_numpy = features.cpu().numpy()
    # 这里可以继续你的DRL逻辑