import cv2
import numpy as np
import os

def calculate_similarity(img1, img2):
    # Calculate Mean Squared Error
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # Convert PSNR to a percentage (0-100)
    similarity_percentage = min(100, 100 * (psnr / 48))  # 48 is chosen as a reasonable max PSNR
    return similarity_percentage

def visualize_color_difference(image1_path, image2_path, output_dir):
    # Read the images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Ensure images are the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Calculate similarity score
    similarity_score = calculate_similarity(img1, img2)

    # Calculate the per-pixel difference
    diff = cv2.absdiff(img1, img2)

    # Create a mask where differences are significant
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th = 30  # You can adjust this threshold
    mask = cv2.threshold(mask, th, 255, cv2.THRESH_BINARY)[1]

    # Create a colored mask
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    # Apply the binary mask to the colored difference image
    diff_color_masked = cv2.bitwise_and(diff_color, diff_color, mask=mask)

    # Blend the original image with the difference map
    result = cv2.addWeighted(img1, 0.7, diff_color_masked, 0.3, 0)

    # Add similarity score to the result image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, f"Similarity: {similarity_score:.2f}%", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cv2.imwrite(os.path.join(output_dir, 'original1.png'), img1)
    cv2.imwrite(os.path.join(output_dir, 'original2.png'), img2)
    cv2.imwrite(os.path.join(output_dir, 'difference_map.png'), diff_color_masked)
    cv2.imwrite(os.path.join(output_dir, 'result.png'), result)

    print(f"Images saved in {output_dir}")
    print(f"Similarity score: {similarity_score:.2f}%")

    return similarity_score

# Usage
score = visualize_color_difference('44e8940e450d6b8e372c190e02596d1d.jpg', '44e8940e450d6b8e372c190e02596d1d.png', 'output_images')
print(f"Final similarity score: {score:.2f}%")