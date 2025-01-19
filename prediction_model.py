# Model is gemaakt met behulp van ChatGPT
import joblib
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model_path = r"D:\3. Minor HU\Startup_met_code_werkend_model\Startup_met_code_werkend_model\Startup\Startup\trained_model.pkl"
model = joblib.load(model_path)
print("Model loaded successfully!")

# Define the color meanings (RGB values and their meanings)
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

# Define the scale and dimensions for each house type
scale_mapping = {
    "Wijster A": "1:200",
    "Midlaren": "1:200",
    "Peelo A": "1:200",
    "Noordbarge": "1:200",
    "Fochteloo B": "1:200",
}

dimensions_mapping = {
    "Wijster A": {"min_length": 15, "max_length": 25, "min_width": 5, "max_width": 6},
    "Midlaren": {"min_length": 25, "max_length": 42, "min_width": 5.5, "max_width": 6.5},
    "Peelo A": {"min_length": 21, "max_length": 36, "min_width": 6, "max_width": 7},
    "Noordbarge": {"min_length": 10, "max_length": 21, "min_width": 5, "max_width": 6.5},
    "Fochteloo B": {"min_length": 12, "max_length": 27, "min_width": 5, "max_width": 6},
}

# Function to find the closest color
def closest_color(pixel, color_list):
    return min(color_list.keys(), key=lambda c: np.linalg.norm(np.array(c) - np.array(pixel)))

# Function to extract features from an image
def extract_features(image_path, color_list, tolerance, scale_mapping, dimensions_mapping):
    img = Image.open(image_path).convert('RGB')
    pixels = img.getdata()
    features = {meaning: 0 for meaning in color_list.values()}
    total_pixels = len(pixels)

    # Count color occurrences
    for pixel in pixels:
        closest = closest_color(pixel, color_list)
        if np.linalg.norm(np.array(closest) - np.array(pixel)) <= tolerance:
            meaning = color_list[closest]
            features[meaning] += 1

    # Normalize the features (proportion of each color)
    for key in features:
        features[key] /= total_pixels

    # Get image dimensions
    img_width, img_height = img.size

    # Get scale
    label = os.path.basename(os.path.dirname(image_path))  # Folder name as label
    scale = scale_mapping.get(label, None)
    if scale:
        scale_factor = float(scale.split(":")[1])
        real_width = img_width / scale_factor
        real_height = img_height / scale_factor
    else:
        real_width = 0
        real_height = 0

    # Add real dimensions
    features['real_width'] = real_width
    features['real_height'] = real_height

    # Validate dimensions
    dimensions = dimensions_mapping.get(label, None)
    if dimensions:
        valid_width = dimensions['min_width'] <= real_width <= dimensions['max_width']
        valid_height = dimensions['min_length'] <= real_height <= dimensions['max_length']
        features['dimensions_valid'] = int(valid_width and valid_height)
        features['min_length'] = dimensions['min_length']
        features['max_length'] = dimensions['max_length']
        features['min_width'] = dimensions['min_width']
        features['max_width'] = dimensions['max_width']
    else:
        features['dimensions_valid'] = 0
        features['min_length'] = 0
        features['max_length'] = 0
        features['min_width'] = 0
        features['max_width'] = 0

    # Add scale
    features['scale'] = float(scale.split(":")[1]) if scale else 0

    return features

# Function to align features with model requirements
def align_features(features, model):
    required_features = model.feature_names_in_
    aligned_features = {key: features.get(key, 0) for key in required_features}
    return pd.DataFrame([aligned_features])

# Function to display an image
def display_image(image_path, title="Image"):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to predict and display results
def predict_and_display(image_path, model, color_list, scale_mapping, dimensions_mapping, tolerance=10):
    print(f"Processing {image_path}...")

    # Extract features
    features = extract_features(image_path, color_list, tolerance, scale_mapping, dimensions_mapping)

    # Align features with the model
    feature_df = align_features(features, model)

    # Predict the type
    predicted_label = model.predict(feature_df)[0]
    print(f"Predicted Type: {predicted_label}")

    # Display the uploaded image
    display_image(image_path, title="Uploaded Image")

    # Display the reference image for the predicted type
    reference_images = {
        "Wijster A": r"D:\3. Minor HU\Startup_met_code_werkend_model\Startup_met_code_werkend_model\Startup\Startup\Wijster A",
        "Midlaren": r"D:\3. Minor HU\Startup_met_code_werkend_model\Startup_met_code_werkend_model\Startup\Startup\Midlaren",
        "Peelo A": r"D:\3. Minor HU\Startup_met_code_werkend_model\Startup_met_code_werkend_model\Startup\Startup\Peelo A",
        "Noordbarge": r"D:\3. Minor HU\Startup_met_code_werkend_model\Startup_met_code_werkend_model\Startup\Startup\Noordbarge",
        "Fochteloo B": r"D:\3. Minor HU\Startup_met_code_werkend_model\Startup_met_code_werkend_model\Startup\Startup\Fochteloo B",
    }

    if predicted_label in reference_images:
        reference_image_path = reference_images[predicted_label]
        display_image(reference_image_path, title=f"Reference Image: {predicted_label}")
    else:
        print(f"No reference image available for {predicted_label}")

# Test the function
image_path = r"D:\3. Minor HU\Startup_met_code_werkend_model\Startup_met_code_werkend_model\Startup\Test_images\1.png"  # Update with the path to your test image
predict_and_display(image_path, model, color_meanings, scale_mapping, dimensions_mapping, tolerance=10)
