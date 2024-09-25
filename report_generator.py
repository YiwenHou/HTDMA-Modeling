import os
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from collections import defaultdict
from tqdm import tqdm
def resize_and_fit_image(img, target_width, target_height):
    """Resize the image to fit within the target width and height while preserving aspect ratio."""
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        # Image is wider than the target cell, fit by width
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        # Image is taller than the target cell, fit by height
        new_height = target_height
        new_width = int(target_height * img_ratio)

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def stitch_images(image_paths, grid_size=(2, 3)):
    """Stitch 6 images into a grid of size grid_size (rows x columns) and return the stitched image."""
    images = [
        Image.open(image_path) if image_path else Image.new('RGB', (600, 400), (255, 255, 255))
        for image_path in image_paths
    ]

    # Define grid cell size
    grid_width = 1200  # Width of the entire grid (adjust as needed)
    grid_height = 1800  # Height of the entire grid (adjust as needed)
    cell_width = grid_width // grid_size[0]
    cell_height = grid_height // grid_size[1]

    # Create a blank image to contain the grid
    stitched_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    # Paste images into the grid
    for i, img in enumerate(images):
        resized_img = resize_and_fit_image(img, cell_width, cell_height)

        # Calculate x and y positions
        x_offset = (i % grid_size[0]) * cell_width
        y_offset = (i // grid_size[0]) * cell_height

        # Paste the resized image into the grid
        stitched_image.paste(resized_img, (x_offset, y_offset))

    return stitched_image

def create_pdf_with_stitched_images(image_groups, output_pdf_path):
    """Create a PDF with stitched images grouped by date, organized by data type."""
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4

    date_keys = list(image_groups.keys())

    for i in tqdm(range(0, len(date_keys), 2)):
        # For every two dates, create one page (two columns)
        column1_images = image_groups[date_keys[i]]  # First date
        column2_images = image_groups[date_keys[i + 1]] if i + 1 < len(date_keys) else [None, None, None]  # Second date or empty

        # Interleave the two columns so that each column has 3 rows, 2 columns on the page
        images_for_page = []
        for row in range(3):  # There are 3 rows per column
            images_for_page.append(column1_images[row])  # Add the image for column 1, row X
            images_for_page.append(column2_images[row])  # Add the image for column 2, row X

        # Stitch images into a single 2x3 grid
        stitched_image = stitch_images(images_for_page)

        # Save the stitched image temporarily
        temp_image_path = f"stitched_page_{i}.jpg"
        stitched_image.save(temp_image_path)

        # Draw the stitched image on the A4 page
        c.drawImage(temp_image_path, inch / 2, inch / 2, width - inch, height - inch, preserveAspectRatio=True)
        c.showPage()

        # Remove temporary stitched image file
        os.remove(temp_image_path)

    c.save()

def organize_images_by_date(image_paths):
    """Organize images into a dictionary sorted by date with empty slots for missing data types."""
    image_dict = defaultdict(lambda: [None, None, None])  # Default list for 3 data types (None if missing)
    
    # Data type index mapping
    data_type_mapping = {'chi': 0, 'acsm': 1, 'kpdf': 2}
    
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        date_part, data_type_part = filename.split('_')[0], filename.split('_')[1].split('.')[0]
        
        # Populate dictionary with the images in correct positions based on data type
        if data_type_part in data_type_mapping:
            data_type_index = data_type_mapping[data_type_part]
            image_dict[date_part][data_type_index] = image_path
    
    # Sort images by date
    sorted_image_dict = dict(sorted(image_dict.items()))
    return sorted_image_dict

def get_images_from_directory(directory):
    """Get all image file paths from the specified directory."""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # Add more extensions if needed
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.lower().endswith(image_extensions)]

# Absolute path to the directory (use os.path.expanduser to handle the '~' for home directory)
image_directory = os.path.expanduser("~/Documents/images")

# Get all images from the directory
image_paths = get_images_from_directory(image_directory)

# Organize images by date and data type
image_groups = organize_images_by_date(image_paths)

# Create a PDF with stitched images grouped by date and data type
output_pdf_path = os.path.expanduser("~/Documents/output_with_grouped_images.pdf")
print(len(image_groups))
create_pdf_with_stitched_images(image_groups, output_pdf_path)

