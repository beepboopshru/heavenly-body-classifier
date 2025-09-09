import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def explore_dataset(data_path='data'):
    """
    Explores the celestial image dataset by counting images in each category
    and displaying a random sample from each.
    """
    # The dataset from Kaggle might have a nested folder structure.
    # Let's find the correct base directory which contains the class folders.
    # Common nested folder names are 'New_images' or similar.
    
    base_path = data_path
    # List contents of the data_path to find the actual image folder
    contents = os.listdir(data_path)
    # A common pattern is a single folder inside the main data directory
    if len(contents) == 1 and os.path.isdir(os.path.join(data_path, contents[0])):
        base_path = os.path.join(data_path, contents[0])

    print(f"Exploring dataset at: {base_path}")

    # Get the category names from the folder names within the base_path
    try:
        categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if not categories:
            print(f"Error: No category subdirectories found in '{base_path}'.")
            print("Please ensure your data is unzipped correctly, e.g., 'data/New_images/galaxy', 'data/New_images/star' etc.")
            return
    except FileNotFoundError:
        print(f"Error: Directory not found at '{base_path}'. Make sure your data is in the correct folder.")
        return

    print("--- Image Counts per Category ---")
    category_counts = {}
    for category in categories:
        category_path = os.path.join(base_path, category)
        num_files = len(os.listdir(category_path))
        category_counts[category] = num_files
        print(f"- {category}: {num_files} images")
    
    print("\n--- Displaying Sample Images ---")
    
    # Set up a figure to display one image from each category
    plt.figure(figsize=(15, 5))
    
    for i, category in enumerate(categories):
        plt.subplot(1, len(categories), i + 1)
        
        category_path = os.path.join(base_path, category)
        
        # Get a random image from the category folder
        random_image_name = random.choice(os.listdir(category_path))
        image_path = os.path.join(category_path, random_image_name)
        
        # Open the image using Pillow and display it
        with Image.open(image_path) as img:
            plt.imshow(img)
            plt.title(f"Category: {category}\nSize: {img.size}")
            plt.axis('off')
            
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Define the name of the folder containing your image categories.
    data_folder = 'Cutout Files'
    
    # Check if the folder exists in the same directory as this script.
    if os.path.exists(data_folder):
        explore_dataset(data_path=data_folder)
    else:
        print(f"Error: '{data_folder}' directory not found.")
        print("Please make sure you have your image folder in the project directory.")

