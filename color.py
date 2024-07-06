from PIL import Image
import numpy as np
from collections import Counter

def top_n_colors(image_path, n=10):
    # Open the image
    img = Image.open(image_path)
    
    # Convert the image to RGB if it is not already
    img = img.convert('RGB')
    
    # Convert the image data to a numpy array
    np_img = np.array(img)
    
    # Reshape the array to be a list of pixels
    pixels = np_img.reshape(-1, 3)
    
    # Convert the list of pixels to a list of tuples
    pixels = [tuple(pixel) for pixel in pixels]
    
    # Count the frequency of each color
    color_counts = Counter(pixels)
    
    # Find the top n most common colors
    top_n_colors = color_counts.most_common(n)
    
    return top_n_colors

# Example usage
image_path = 'color.jpg'
top_colors = top_n_colors(image_path, n=10)
for i, color in enumerate(top_colors, 1):
    print(f"Rank {i}: Color {color[0]} with {color[1]} occurrences")
