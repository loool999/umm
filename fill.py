from PIL import Image

def create_filled_image(input_image_path, output_image_path, color):
    # Open the input image to get its size
    with Image.open(input_image_path) as input_image:
        width, height = input_image.size

    # Create a new image with the same size as the input image and the specified color
    filled_image = Image.new("RGB", (width, height), color)

    # Save the new image to the specified output path
    filled_image.save(output_image_path)

# Define the color and input/output image paths
color = (134, 168, 178)  # The specified color
input_image_path = "image.jpg"  # Path to the input image
output_image_path = "filled.jpg"  # Path to save the filled image

# Create the filled image
create_filled_image(input_image_path, output_image_path, color)
