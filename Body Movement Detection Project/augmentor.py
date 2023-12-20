import Augmentor
from PIL import Image, ImageEnhance

# Define the path to the directory containing your original images
original_images_dir = 'Dataset/Loops/Open Loops/'

# Define the path where augmented images will be saved
output_dir = 'Dataset/Loops/ Loops/'

# Create a pipeline for data augmentation
p = Augmentor.Pipeline(original_images_dir, output_directory=output_dir)

# Define the augmentation operations you want to apply
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)

# Function to enhance image contrast
def enhance_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    factor = 1.3  # Adjust the contrast factor as needed
    return enhancer.enhance(factor)

# Apply contrast enhancement
p.add_operation(enhance_contrast)

# Set the number of augmented images you want to generate
num_augmented_images = 100  # Change as needed

# Execute the augmentation process
p.sample(num_augmented_images)
