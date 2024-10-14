import cv2
import os

filename = '1.mp4'
# 视频文件路径
video_path = f'../Video/{filename}'
# 图片保存路径
image_folder = '../Output'

# 确保图片保存路径存在
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open Video.")
    exit()

# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in Video: {total_frames}")

# 读取视频帧并保存为图片
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Reached end of Video or failed to read frame.")
        break
    # 构建图片文件名
    image_name = os.path.join(image_folder, f'frame_{frame_id:04d}.jpg')
    # 保存图片
    cv2.imwrite(image_name, frame)
    print(f"Saved {image_name}")  # 打印保存的图片文件名
    frame_id += 1

# 释放资源
cap.release()
print(f'图片已保存到 {image_folder}')