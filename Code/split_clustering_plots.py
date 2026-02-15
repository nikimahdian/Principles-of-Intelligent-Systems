"""
Split question2_clustering_analysis.png into 4 separate images
"""

import matplotlib.pyplot as plt
from PIL import Image
import os

def split_clustering_analysis_image():
    """Split the 2x2 subplot image into 4 separate images"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    img_dir = os.path.join(project_root, 'Report', 'img')
    
    input_image = os.path.join(img_dir, 'question2_clustering_analysis.png')
    
    if not os.path.exists(input_image):
        print(f"Error: {input_image} not found!")
        return
    
    # Load the image
    img = Image.open(input_image)
    width, height = img.size
    
    print(f"Original image size: {width} x {height}")
    
    # Calculate dimensions for each subplot (2x2 grid)
    subplot_width = width // 2
    subplot_height = height // 2
    
    print(f"Each subplot size: {subplot_width} x {subplot_height}")
    
    # Define crop boxes for each subplot
    # Top-left: (left, top, right, bottom)
    crops = {
        'kmeans_inertia_silhouette': (0, 0, subplot_width, subplot_height),  # Top-left
        'agglomerative_linkage': (subplot_width, 0, width, subplot_height),  # Top-right
        'dbscan_heatmap': (0, subplot_height, subplot_width, height),  # Bottom-left
        'clustering_comparison': (subplot_width, subplot_height, width, height)  # Bottom-right
    }
    
    # Crop and save each subplot
    for name, box in crops.items():
        cropped = img.crop(box)
        output_path = os.path.join(img_dir, f'question2_{name}.png')
        cropped.save(output_path, 'PNG', dpi=(300, 300))
        print(f"Saved: {output_path} (size: {cropped.size})")
    
    print("\nAll subplots saved successfully!")

if __name__ == "__main__":
    split_clustering_analysis_image()



