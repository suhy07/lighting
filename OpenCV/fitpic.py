import os
from PIL import Image
from tqdm import tqdm

def resize_and_pad(img_path, output_path, target_size=(1080, 1920), color=(255, 255, 255)):
    img = Image.open(img_path).convert("RGBA")  # 转换为 RGBA 模式以保持透明度
    img_width, img_height = img.size

    # 创建新画布，背景颜色为白色，并保持透明度
    new_img = Image.new('RGBA', target_size, (255, 255, 255, 0))  # 透明背景

    # 计算填充的位置
    left = target_size[0] - img_width
    top = target_size[1] - img_height

    # 确保左和上边界不小于0
    left = max(0, left)
    top = max(0, top)

    # 粘贴原始图片
    new_img.paste(img, (left, top), img)  # 使用 img 作为掩码，以保留透明度

    # 保存输出图片
    new_img.convert("RGB").save(output_path)  # 转换为 RGB 保存，去掉透明度

def process_directory(directory, output_directory, target_size=(1080, 1920), color=(255, 255, 255)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                all_files.append(os.path.join(root, file))

    for img_path in tqdm(all_files, desc='Processing images'):
        relative_path = os.path.relpath(img_path, directory)
        output_subdir = os.path.join(output_directory, os.path.dirname(relative_path))
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        output_path = os.path.join(output_subdir, os.path.basename(img_path))
        try:
            resize_and_pad(img_path, output_path, target_size, color)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

source_directory = 'Z:\\AI素材'
output_directory = 'Z:\\AI素材_Processed'

process_directory(source_directory, output_directory)
