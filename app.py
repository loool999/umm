# /app.py

from flask import Flask, request, render_template_string
import numpy as np
import cv2
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Difference Heatmap</title>
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container">
        <h1 class="my-4">Upload Images to Compare</h1>
        <form action="/" method="post" enctype="multipart/form-data" class="mb-4">
            <div class="form-group">
                <label for="image1">First Image:</label>
                <input type="file" name="image1" class="form-control" required>
            </div>
            <div class="form-group">
                <label for="image2">Second Image (optional):</label>
                <input type="file" name="image2" class="form-control">
            </div>
            <button type="submit" class="btn btn-primary">Upload</button>
        </form>
        {% if uploaded %}
            <h2 class="my-4">Heatmap Result</h2>
            <p class="lead">Score: {{ score }}</p>
            <img src="data:image/png;base64,{{ heatmap_image }}" alt="Heatmap Image" class="img-fluid">
            <br><br>
            <a href="/" class="btn btn-secondary">Upload Another</a>
        {% endif %}
    </div>
</body>
</html>
"""

app = Flask(__name__)

def create_blank_image(width, height):
    return np.zeros((height, width, 3), np.uint8) + 255

def calculate_difference(img1, img2):
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return diff, diff_gray

def generate_detailed_heatmap(diff_gray):
    max_diff = 255
    colormap = plt.get_cmap('RdYlGn_r')
    norm_diff = diff_gray / max_diff
    heatmap_color = colormap(norm_diff)
    heatmap_color = (heatmap_color[:, :, :3] * 255).astype(np.uint8)
    return heatmap_color

def calculate_score(heatmap_color):
    green_points = np.sum(np.all(heatmap_color == [0, 255, 0], axis=2)) * 1.0
    orange_points = np.sum(np.all(heatmap_color == [255, 165, 0], axis=2)) * 0.5
    red_points = np.sum(np.all(heatmap_color == [255, 0, 0], axis=2)) * 0.0
    score = green_points + orange_points + red_points
    return score

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img1 = Image.open(request.files['image1']).convert('RGB')
        img2 = request.files['image2']
        img2 = Image.open(img2).convert('RGB') if img2 else None
        
        img1 = np.array(img1)
        img2 = np.array(img2) if img2 else create_blank_image(img1.shape[1], img1.shape[0])
        
        diff, diff_gray = calculate_difference(img1, img2)
        heatmap_color = generate_detailed_heatmap(diff_gray)
        score = calculate_score(heatmap_color)
        heatmap_base64 = image_to_base64(heatmap_color)
        
        return render_template_string(HTML_TEMPLATE, heatmap_image=heatmap_base64, score=score, uploaded=True)
    
    return render_template_string(HTML_TEMPLATE, uploaded=False)

if __name__ == '__main__':
    app.run(debug=True)
