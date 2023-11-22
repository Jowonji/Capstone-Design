import math

import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imutils import paths
from tensorflow import keras

def plot_images(images, title=None, save_path=None):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        if title is not None:
            plt.title(title)
        plt.imshow(images[i])
        plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path)

ckpt_path = "./dreambooth-person6.h5"

unique_id = "sks"
class_label = "person"

resolution = 384

# Initialize a new Stable Diffusion model.
dreambooth_model = keras_cv.models.StableDiffusion(
    img_width=resolution, img_height=resolution, jit_compile=True
)
dreambooth_model.diffusion_model.load_weights(ckpt_path)

# Note how the unique identifier and the class have been used in the prompt.
prompt = f"A photo of {unique_id} {class_label} with short curly perm hairstyle"
num_imgs_to_gen = 3

images_dreamboothed = dreambooth_model.text_to_image(prompt, batch_size=num_imgs_to_gen)
plot_images(images_dreamboothed, prompt, save_path='./hair15')