from PIL import Image, ImageEnhance
from collections import Counter
import cv2
import numpy as np
import os
import random

input_image_path = 'image.jpg'
output_image_path = 'output.jpg'
object_folder = 'object'
process_folder = 'process'
phase_folder = 'phase-1'
output_v1_image_path = os.path.join(process_folder, 'output-v1.png')  # Change to PNG

def create_blank_image(input_image_path, output_image_path):
    with Image.open(input_image_path) as img:
        img = img.convert('RGB')
        pixels = list(img.getdata())
        pixel_counts = Counter(pixels)
        most_frequent_pixel = pixel_counts.most_common(1)[0][0]
        new_img = Image.new('RGB', img.size, most_frequent_pixel)
        new_img.save(output_image_path)

def adjust_object_image_color(obj_img, base_img, paste_position):
    # Create a mask to isolate the area where the object image will be pasted
    mask = Image.new('L', base_img.size, 0)
    mask.paste(obj_img.split()[-1], paste_position)

    # Extract the region of interest (ROI) from the base image
    base_img_crop = base_img.crop((paste_position[0], paste_position[1], paste_position[0] + obj_img.width, paste_position[1] + obj_img.height))

    # Get the most frequent color in the ROI
    base_pixels = list(base_img_crop.getdata())
    base_pixel_counts = Counter(base_pixels)
    most_frequent_base_pixel = base_pixel_counts.most_common(1)[0][0]

    # Adjust the object image color to the most frequent color in the ROI
    obj_img_adjusted = Image.new("RGBA", obj_img.size, most_frequent_base_pixel)
    obj_img_adjusted.paste(obj_img, (0, 0), obj_img)

    return obj_img_adjusted

def paste_object_image(base_image_path, object_folder, output_v1_image_path):
    with Image.open(base_image_path) as base_img:
        base_img = base_img.convert('RGBA')
        object_image_path = random.choice([os.path.join(object_folder, f) for f in os.listdir(object_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        with Image.open(object_image_path) as obj_img:
            obj_img = obj_img.convert('RGBA')
            # Resize object image if it's larger than the base image
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
            # Adjust object image color
            obj_img = adjust_object_image_color(obj_img, base_img, paste_position)
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

def process_images(input_image_path, output_image_path, object_folder, process_folder, phase_folder, num_phases, iterations):
    create_blank_image(input_image_path, output_image_path)
    if not os.path.exists(process_folder):
        os.makedirs(process_folder)
    if not os.path.exists(phase_folder):
        os.makedirs(phase_folder)
    last_similarity_score = 0
    base_image_path = output_image_path
    for phase in range(num_phases):
        print(f"Phase {phase + 1}")
        best_output_path = None
        for _ in range(iterations):  # Adjust the number of iterations as needed
            paste_object_image(base_image_path, object_folder, output_v1_image_path)
            score = visualize_color_difference(input_image_path, output_v1_image_path, process_folder)
            if score > last_similarity_score:
                last_similarity_score = score
                if best_output_path:
                    os.remove(best_output_path)
                best_output_path = os.path.join(process_folder, f'best_output_phase_{phase + 1}.png')
                os.rename(output_v1_image_path, best_output_path)
                print(f"New best image saved with similarity score: {score:.2f}%")
            else:
                os.remove(output_v1_image_path)
                print(f"Discarded image with similarity score: {score:.2f}%")
        if best_output_path and os.path.exists(best_output_path):
            phase_best_output_path = os.path.join(phase_folder, f'best_output_phase_{phase + 1}.png')
            os.rename(best_output_path, phase_best_output_path)
            base_image_path = phase_best_output_path
        else:
            print(f"No new best image found in phase {phase + 1}")

process_images(input_image_path, output_image_path, object_folder, process_folder, phase_folder, num_phases=15, iterations=40)
