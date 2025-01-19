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

# Function to extract features from a single image
def extract_features(image_path, color_list, tolerance=10):
    color_distribution = map_colors_with_tolerance(image_path, color_list, tolerance)
    total_pixels = sum(color_distribution.values())
    # Normalize the counts to proportions
    features = {key: color_distribution.get(key, 0) / total_pixels for key in color_list.values()}
    return features

# Function to process all images in a folder and subfolders
def process_images_in_folder(folder_path, color_list, output_csv, tolerance=10):
    data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)  # Use folder name as label
                print(f"Processing {image_path}...")

                # Extract features
                features = extract_features(image_path, color_list, tolerance)
                features['label'] = label
                features['image_path'] = image_path  # Include image path for reference
                data.append(features)

                # Optional: Display the image
                #img = Image.open(image_path)
                #plt.imshow(img)
                #plt.axis('off')
                #plt.show()
    
    # Convert the data into a DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete. Data saved to {output_csv}")

# Example usage
folder_path = r"C:\Users\ecgam\Documents\Minor_HU\Startup\Startup"  # Folder containing images
output_csv = r"C:\Users\ecgam\Documents\Minor_HU\Startup\Startup\features.csv"
process_images_in_folder(folder_path, color_meanings, output_csv)
