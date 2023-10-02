import os
import imageio
from tqdm import tqdm

def create_gif_from_folder(folder_path, output_gif_path):
    # Get a list of all image filenames in the folder
    image_filenames = sorted(os.listdir(folder_path))

    # Filter out non-image files
    image_filenames = [filename for filename in image_filenames if filename.endswith('.png') or filename.endswith('.jpg')]

    # Create a list to hold the images
    images = []

    # Open each image and append it to the images list
    for filename in image_filenames:
        image_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(image_path))

    # Write the images to a gif
    imageio.mimsave(output_gif_path, images)

def create_gifs_from_subfolders(main_folder_path, output_folder_path):
    # Iterate over all subdirectories in the main folder
    for subfolder in tqdm(os.listdir(main_folder_path)):
        subfolder_path = os.path.join(main_folder_path, subfolder)

        # Check if it's indeed a subdirectory
        if os.path.isdir(subfolder_path):
            main_name = main_folder_path.split('/')[-2]
            # Create the output GIF filename
            output_gif_filename = f'{main_name}_{subfolder}.gif'
            output_gif_path = os.path.join(output_folder_path, output_gif_filename)

            # Create a GIF from the images in the subfolder
            create_gif_from_folder(subfolder_path, output_gif_path)

# Define your main directory and the output directory
main_directory = 'data/BOP/lmo/test_video_pre'
output_directory = 'p1_gif'

create_gifs_from_subfolders(main_directory, output_directory)
