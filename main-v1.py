from PIL import Image
from collections import Counter
import cv2
import numpy as np
import os
import random

input_image_path = 'image.jpg'
output_image_path = 'output.jpg'
object_folder = 'object'
process_folder = 'process'
output_v1_image_path = os.path.join(process_folder, 'output-v1.png')  # Change to PNG

def create_blank_image(input_image_path, output_image_path):
    with Image.open(input_image_path) as img:
        img = img.convert('RGB')
        pixels = list(img.getdata())
        pixel_counts = Counter(pixels)
        most_frequent_pixel = pixel_counts.most_common(1)[0][0]
        new_img = Image.new('RGB', img.size, most_frequent_pixel)
        new_img.save(output_image_path)

def paste_object_image(base_image_path, object_folder, output_v1_image_path):
    with Image.open(base_image_path) as base_img:
        base_img = base_img.convert('RGBA')
        object_image_path = random.choice([os.path.join(object_folder, f) for f in os.listdir(object_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        with Image.open(object_image_path) as obj_img:
            obj_img = obj_img.convert('RGBA')
            # Resize object image if it's larger than the base image
            if obj_img.width > base_img.width or obj_img.height > base_img.height:
                scale_factor = min(base_img.width / obj_img.width, base_img.height / obj_img.height) * 0.5
            else:
                scale_factor = random.uniform(0.1, 0.5)
            new_size = (int(obj_img.width * scale_factor), int(obj_img.height * scale_factor))
            obj_img = obj_img.resize(new_size, Image.LANCZOS)
            # Rotate object image
            angle = random.uniform(0, 360)
            obj_img = obj_img.rotate(angle, expand=True)
            # Ensure the object image fits within the base image
            max_x = max(0, base_img.width - obj_img.width)
            max_y = max(0, base_img.height - obj_img.height)
            paste_position = (random.randint(0, max_x), random.randint(0, max_y))
            base_img.paste(obj_img, paste_position, obj_img)
            base_img.save(output_v1_image_path)

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
    cv2.imwrite(os.path.join(output_dir, 'difference_map.png'), diff_color)
    return similarity_score

def process_images():
    create_blank_image(input_image_path, output_image_path)
    if not os.path.exists(process_folder):
        os.makedirs(process_folder)
    last_similarity_score = 0
    for _ in range(10):  # Adjust the number of iterations as needed
        paste_object_image(output_image_path, object_folder, output_v1_image_path)
        score = visualize_color_difference(input_image_path, output_v1_image_path, process_folder)
        if score > last_similarity_score:
            last_similarity_score = score
            best_output_path = os.path.join(process_folder, 'best_output.png')  # Change to PNG
            os.rename(output_v1_image_path, best_output_path)
            print(f"New best image saved with similarity score: {score:.2f}%")
        else:
            os.remove(output_v1_image_path)
            print(f"Discarded image with similarity score: {score:.2f}%")

process_images()
