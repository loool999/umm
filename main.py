import cv2
import numpy as np
import os
from heapq import heappush, heappop
import uuid
from tqdm import tqdm

def calculate_similarity(img1, img2):
    diff = cv2.absdiff(img1, img2)
    similarity = 1 - (np.sum(diff) / (img1.shape[0] * img1.shape[1] * 255 * 3))
    similarity_percentage = similarity * 100
    return similarity_percentage

def insert_object(background, object_img, x, y, scale, rotation):
    h, w = object_img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, rotation, scale)
    rotated = cv2.warpAffine(object_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    yh, xw = rotated.shape[:2]
    if x + xw > background.shape[1] or y + yh > background.shape[0] or x < 0 or y < 0:
        return background

    if rotated.shape[2] == 4:  # If the image has an alpha channel
        alpha_channel = rotated[:,:,3] / 255.0
        for c in range(0, 3):
            background[y:y+yh, x:x+xw, c] = background[y:y+yh, x:x+xw, c] * (1 - alpha_channel) + rotated[:,:,c] * alpha_channel
    else:  # If the image doesn't have an alpha channel
        background[y:y+yh, x:x+xw] = rotated
    
    return background

def optimize_insertion(background_path, object_folder, output_folder, done_folder, max_iterations=1000, top_k=10):
    original_background = cv2.imread(background_path)
    if original_background is None:
        print(f"Error: Could not read background image")
        return

    target = np.ones_like(original_background) * 255

    for folder in [output_folder, done_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    top_results = []
    best_score = 0
    best_image = target.copy()

    object_files = sorted(os.listdir(object_folder))
    
    with tqdm(total=max_iterations, desc="Processing", unit="iteration") as pbar:
        for iteration in range(max_iterations):
            obj_filename = np.random.choice(object_files)
            obj_path = os.path.join(object_folder, obj_filename)
            object_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            if object_img is None:
                print(f"Error: Could not read object image {obj_path}")
                continue

            x = np.random.randint(0, original_background.shape[1])
            y = np.random.randint(0, original_background.shape[0])
            scale = np.random.uniform(0.5, 1.5)
            rotation = np.random.uniform(0, 360)

            modified = insert_object(best_image.copy(), object_img, x, y, scale, rotation)
            score = calculate_similarity(original_background, modified)

            if score > best_score:
                best_score = score
                best_image = modified.copy()

                result_id = str(uuid.uuid4())
                if len(top_results) < top_k:
                    heappush(top_results, (score, result_id, obj_filename, x, y, scale, rotation))
                elif score > top_results[0][0]:
                    heappop(top_results)
                    heappush(top_results, (score, result_id, obj_filename, x, y, scale, rotation))

                # Save the current best result
                cv2.imwrite(os.path.join(output_folder, f"best_score_{score:.2f}.png"), best_image)

            pbar.update(1)
            pbar.set_postfix({"Best Score": f"{best_score:.2f}%"})

            if best_score >= 99.99:  # If we've reached near-perfect similarity, stop
                break

    # Save top k results
    for i, (score, result_id, obj_filename, x, y, scale, rotation) in enumerate(sorted(top_results, reverse=True), 1):
        output_path = os.path.join(output_folder, f"top_{i}_score_{score:.2f}.png")
        cv2.imwrite(output_path, best_image)
        print(f"Saved {output_path} (Object: {obj_filename}, X: {x}, Y: {y}, Scale: {scale}, Rotation: {rotation})")

    # Save highest score image in 'done' folder
    if top_results:
        best_score, _, best_obj, _, _, _, _ = max(top_results)
        done_path = os.path.join(done_folder, f"best_score_{best_score:.2f}_{best_obj}.png")
        cv2.imwrite(done_path, best_image)
        print(f"Saved best result: {done_path}")

# Usage
background_path = 'Screenshot 2024-06-12 5.39.30 PM.png'  # Replace with your background image path
object_folder = 'object'  # Replace with your object folder path
output_folder = 'output_images'  # Replace with your desired output folder
done_folder = 'done'  # Replace with your desired 'done' folder

optimize_insertion(background_path, object_folder, output_folder, done_folder)
