from PIL import Image

# Load the transparent image
image_path = 'object/wavy.png'
image = Image.open(image_path).convert("RGBA")

# Create a tint color (e.g., red with full opacity)
tint_color = (255, 0, 0, 255)  # (R, G, B, A)

# Create a new image with the same size and the tint color
tint_layer = Image.new("RGBA", image.size, tint_color)

# Extract the alpha channel from the original image
alpha = image.split()[3]

# Composite the images using the alpha channel as a mask
tinted_image = Image.composite(tint_layer, image, alpha)

# Save the result
tinted_image.save('tinted_image.png')
