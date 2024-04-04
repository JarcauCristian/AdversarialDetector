import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


def add_noise_to_image(image_path: str, noise_level: float = 0.1) -> Image:
    """Add random noise to an image.

    Args:
        image_path (str): The path to the input image.
        noise_level (float): The intensity of the noise to add, as a fraction of the maximum value.

    Returns:
        PIL.Image: A new image object with noise added.
    """
    # Load the image
    img = Image.open(image_path).convert('RGB')

    # Convert image to numpy array
    img_array = np.asarray(img)

    # Generate random noise
    noise = np.random.randn(*img_array.shape) * 255 * noise_level

    # Add the noise to the image
    noisy_img_array = img_array + noise

    # Ensure values are within the proper range
    noisy_img_array = np.clip(noisy_img_array, 0, 255)

    # Convert back to an image
    noisy_img = Image.fromarray(noisy_img_array.astype('uint8'), 'RGB')

    return noisy_img


def generate_images(base_path: str) -> None:
    for path in Path(base_path).glob("*"):
        new_path = str(path).split("/")[0] + "/" + str(path).split("/")[-1].split(".")[0] + "_attacked.png"

        attacked_image = add_noise_to_image(path)

        attacked_image.save(new_path)


def generate_labels(base_path: str, label_file: str):
    df = []
    for path in Path(base_path).glob("*"):
        if "attacked" in str(path):
            df.append({
                "image_name": str(path).split("/")[-1],
                "label": 1
            })
        else:
            df.append({
                "image_name": str(path).split("/")[-1],
                "label": 0
            })
    
    df = pd.DataFrame(df)
    df.to_csv(label_file, index=False)
            

if __name__ == '__main__':
    # generate_images("./TEST")
    generate_labels("./TEST", "attacked_image_labels.csv")
