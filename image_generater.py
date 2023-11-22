from tqdm import tqdm
import numpy as np
import hashlib
import keras_cv
import PIL
import os
import tensorflow as tf

class_images_dir = "class-cat"
os.makedirs(class_images_dir, exist_ok=True)


model = keras_cv.models.StableDiffusion(img_width=384, img_height=384, jit_compile=True)

class_prompt = "A photo of cat"
num_imgs_to_generate = 20
for i in tqdm(range(num_imgs_to_generate)):
    images = model.text_to_image(
        class_prompt,
        batch_size=3
    )
    idx = np.random.choice(len(images))
    selected_image = PIL.Image.fromarray(images[idx])
    hash_image = hashlib.sha1(selected_image.tobytes()).hexdigest()
    image_filename = os.path.join(class_images_dir, f"{hash_image}.jpg")
    selected_image.save(image_filename)