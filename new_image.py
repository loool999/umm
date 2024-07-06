from PIL import Image
from collections import Counter
import cv2
import numpy as np
import os

input_image_path = 'image.jpg'
output_image_path = 'output.jpg'

def create_blank_image(input_image_path, output_image_path):
    with Image.open(input_image_path) as img:
        img = img.convert('RGB')
        pixels = list(img.getdata())
        pixel_counts = Counter(pixels)
        most_frequent_pixel = pixel_counts.most_common(1)[0][0]
        new_img = Image.new('RGB', img.size, most_frequent_pixel)
        new_img.save(output_image_path)

create_blank_image(input_image_path, output_image_path)

def calculate_similarity(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    similarity_percentage = min(100, 100 * (psnr / 48))
    return similarity_percentage

def create_colormap():
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        colormap[i, 0, 2] = int(255 * (i / 255.0))
        colormap[i, 0, 1] = int(255 * ((255 - i) / 255.0))
    return colormap

def visualize_color_difference(image1_path, image2_path, output_dir):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    similarity_score = calculate_similarity(img1, img2)
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    colormap = create_colormap()
    diff_color = cv2.applyColorMap(diff_gray, colormap)

    result = cv2.addWeighted(img1, 0.7, diff_color, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, f"Similarity: {similarity_score:.2f}%", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
#   cv2.imwrite(os.path.join(output_dir, 'original1.png'), img1)
#   cv2.imwrite(os.path.join(output_dir, 'original2.png'), img2)
    cv2.imwrite(os.path.join(output_dir, 'difference_map.png'), diff_color)
#   cv2.imwrite(os.path.join(output_dir, 'result.png'), result)

    print(f"Images saved in {output_dir}")

    return similarity_score

score = visualize_color_difference(input_image_path, output_image_path, 'output_images')
print(f"Similarity score: {score:.2f}%")
