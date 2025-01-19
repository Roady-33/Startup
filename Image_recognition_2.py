# Model is gemaakt met behulp van ChatGPT
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Define the color meanings in RGB format
color_meanings = {
    (0, 0, 0): "Paalkuil",
    (194, 193, 195): "Paalkuil",
    (237, 199, 149): "Haardkuil, Vooraadkuil",
    (219, 152, 101): "Haardkuil, Vooraadkuil",
    (238, 111, 62): "Standspoor wand of stalscheiding",
    (249, 190, 146): "Standspoor wand of stalscheiding",
    (233, 193, 90): "Vlechtwerkwerk wand of stalscheiding",
    (194, 193, 118): "Planken van wand of vloer",
    (217, 215, 142): "Planken van wand of vloer",
    (194, 192, 143): "Verdiepte vloer, potstal",
    (213, 211, 162): "Verdiepte vloer, potstal",
    (0, 165, 84): "Plag, plaggenwand",
    (0, 144, 72): "Plag, plaggenwand",
    (255, 99, 99): "Hulplijn wand",
    (51, 150, 51): "Hulplijn dragende structuur",
    (99, 99, 255): "Ingangkuil, drempelkuil, stalgoot",
    (228, 112, 61): "Pallisade",
    (240, 128, 78): "Aarden wal",
    (111, 158, 204): "Waterput",
    (0, 229, 229): "Sloot of greppel",
    (255, 255, 191): "Overige kuilen",
    (255, 255, 255): "Achtergrond"
}

# Define expected dimension ranges for each house type
house_dimensions = {
    "Wijster A": {"length": (15, 25), "width": (5, 6)},
    "Midlaren": {"length": (25, 42), "width": (5.5, 6.5)},
    "Peelo A": {"length": (21, 36), "width": (6, 7)},
    "Noordbarge": {"length": (10, 21), "width": (5, 6.5)},
    "Fochteloo B": {"length": (12, 27), "width": (5, 6)},
    "Wijster B": {"length": (22, 36), "width": (6)},
    "Wijster C": {"length": (18, 22), "width": (6, 7)},
    "Peelo B": {"length": (13, 24), "width": (5, 6)},
}

# Function to find the closest color
def closest_color(pixel, color_list):
    return min(color_list.keys(), key=lambda c: np.linalg.norm(np.array(c) - np.array(pixel)))

# Function to map colors with tolerance
def map_colors_with_tolerance(image_path, color_list, tolerance=10):
    img = Image.open(image_path).convert('RGB')
    pixels = img.getdata()
    img_colors = {}
    
    for pixel in pixels:
        closest = closest_color(pixel, color_list)
        if np.linalg.norm(np.array(closest) - np.array(pixel)) <= tolerance:
            meaning = color_list[closest]
            img_colors[meaning] = img_colors.get(meaning, 0) + 1
    
    return img_colors

# Function to calculate dimensions based on scale
def calculate_dimensions(image, scale):
    width_pixels, height_pixels = image.size
    try:
        # Extract the numeric scale factor from the string "1:200"
        scale_factor = int(scale.split(":")[1])
    except (IndexError, ValueError):
        raise ValueError(f"Invalid scale format: {scale}. Expected format is '1:200'.")
    
    real_width = width_pixels / scale_factor
    real_height = height_pixels / scale_factor
    return real_width, real_height


# Function to check if dimensions are within expected range
def validate_dimensions(label, real_width, real_height, dimension_ranges):
    if label in dimension_ranges:
        length_range = dimension_ranges[label]["length"]
        width_range = dimension_ranges[label]["width"]
        within_length = length_range[0] <= real_height <= length_range[1]
        within_width = width_range[0] <= real_width <= width_range[1]
        return within_length and within_width
    return False

# Function to extract features from a single image
def extract_features(image_path, color_list, label, scale="1:200", tolerance=10):
    img = Image.open(image_path)
    color_distribution = map_colors_with_tolerance(image_path, color_list, tolerance)
    total_pixels = sum(color_distribution.values())
    
    # Normalize the counts to proportions
    features = {key: color_distribution.get(key, 0) / total_pixels for key in color_list.values()}
    
    # Calculate dimensions
    real_width, real_height = calculate_dimensions(img, scale)
    features['real_width'] = real_width
    features['real_height'] = real_height
    features['scale'] = scale
    
    # Validate dimensions
    dimensions_valid = validate_dimensions(label, real_width, real_height, house_dimensions)
    features['dimensions_valid'] = dimensions_valid
    return features

def process_images_in_folder(folder_path, color_list, output_csv, tolerance=10):
    data = []
    # Define the scale and dimensions for each house type
    scale_mapping = {
        "Wijster A": "1:200",  # Scale for Wijster A is 1:200
        "Midlaren": "1:200",
        "Peelo A": "1:200",
        "Noordbarge": "1:200",
        "Fochteloo B": "1:200",
        "Wijster C": "1:200",
        "Wijster B": "1:200",
        "Peelo B": "1:200"
    }
    
    dimensions_mapping = {
        "Wijster A": {"min_length": 15, "max_length": 25, "min_width": 5, "max_width": 6},
        "Midlaren": {"min_length": 25, "max_length": 42, "min_width": 5.5, "max_width": 6.5},
        "Peelo A": {"min_length": 21, "max_length": 36, "min_width": 6, "max_width": 7},
        "Noordbarge": {"min_length": 10, "max_length": 21, "min_width": 5, "max_width": 6.5},
        "Fochteloo B": {"min_length": 12, "max_length": 27, "min_width": 5, "max_width": 6},
        "Wijster C": {"min_length": 18, "max_length": 22, "min_width": 6, "max_width": 7},
        "Peelo B": {"min_length": 13, "max_length": 24, "min_width": 5, "max_width": 6},
        "Wijster B": {"min_length": 22, "max_length": 36, "min_width": 6, "max_width": 6},
    }
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)  # Use folder name as label
                print(f"Processing {image_path}...")

                # Get the scale for the current house type
                scale = scale_mapping.get(label, "1:200")  # Default to "1:200" if no scale is provided

                # Extract features, explicitly passing the scale
                features = extract_features(image_path, color_list, label, scale=scale, tolerance=tolerance)
                
                features['label'] = label
                features['image_path'] = image_path  # Include image path for reference

                # Add dimensions
                dimensions = dimensions_mapping.get(label, None)
                if dimensions:
                    features.update(dimensions)

                data.append(features)
    
    # Convert the data into a DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete. Data saved to {output_csv}")

# Example usage
folder_path = r"C:\Users\ecgam\Desktop\Startup_met_code_werkend_model\Startup"  # Folder containing images
output_csv = r"C:\Users\ecgam\Desktop\Startup_met_code_werkend_model\Startup\Startup\features.csv"
process_images_in_folder(folder_path, color_meanings, output_csv)
