import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os


csv_path = './data/original/plates.csv'
image_dir = './data/original/photos'
flipped_dir = './data/original/flipped_photos'
noise_dir = './data/original/noise_photos'
flipped_noise_dir = './data/original/flipped_noise_photos'


def read_plates_csv(csv_path):
    # Load original CSV to pandas dataframe
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        name = row['name']
        xtl, ytl, xbr, ybr = row['xtl'], row['ytl'], row['xbr'], row['ybr']
        img_width, img_height = row['img_width'], row['img_height']

        yield name, xtl, ytl, xbr, ybr, img_width, img_height


def save_flipped_images(image_dir, flipped_dir, csv_path):    
    os.makedirs(flipped_dir, exist_ok=True) 
    flipped_rows = []
    for name, xtl, ytl, xbr, ybr, img_width, img_height in read_plates_csv(csv_path):
        # Load and flip image
        img_path = os.path.join(image_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Image {name} not found!")
            continue

        flipped_img = cv2.flip(img, 1)
        flipped_name = f'flip_{name}'
        flipped_path = os.path.join(flipped_dir, flipped_name)
        cv2.imwrite(flipped_path, flipped_img)

        # Flipped bbox coordinates
        bbox_width = xbr - xtl
        bbox_height = ybr - ytl
        new_xtl = img_width - xtl - bbox_width
        new_xbr = new_xtl + bbox_width
        new_ytl = ybr - bbox_height
        new_ybr = new_ytl + bbox_height

        flipped_rows.append({
            'name': flipped_name,
            'xtl': round(new_xtl, 2),
            'ytl': round(new_ytl, 2),
            'xbr': round(new_xbr, 2),
            'ybr': round(new_ybr, 2),
            'img_width': img_width,
            'img_height': img_height
        })

    # Save flipped data to CSV
    flipped_df = pd.DataFrame(flipped_rows)
    flipped_df.to_csv('./data/original/flipped_plates.csv', index=False)
    print(f"Flipping complete. CSV saved, total flipped rows: {len(flipped_rows)}")


def add_noise_and_brightness(image_np):
    # Convert image to float32 and normalize to [0.0, 1.0]
    image = tf.convert_to_tensor(image_np, dtype=tf.float32) / 255.0

    # Apply random brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Add Gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = image + noise

    # Keep values in [0.0, 1.0]
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Convert back to uint8 format (0–255) for saving with OpenCV
    return (image * 255).numpy().astype("uint8")


def save_noisy_images(image_dir, noise_dir, csv_path, new_csv_path):
    os.makedirs(noise_dir, exist_ok=True)
    noisy_rows = []
    for name, xtl, ytl, xbr, ybr, img_width, img_height in read_plates_csv(csv_path):
        image_path = os.path.join(image_dir, name)
        original_image = cv2.imread(image_path)

        if original_image is None:
            print(f"Warning: Image {name} not found!")
            continue

        # Apply noise and brightness
        noisy_image = add_noise_and_brightness(original_image)

        # Save noisy image
        new_name = f"noise_{name}"
        save_path = os.path.join(noise_dir, new_name)
        cv2.imwrite(save_path, noisy_image)

        # Save new image name with the same bounding boxes coordinates
        noisy_rows.append({
            'name': new_name,
            'xtl': xtl,
            'ytl': ytl,
            'xbr': xbr,
            'ybr': ybr,
            'img_width': img_width,
            'img_height': img_height
        })        

    # Save noised data to CSV
    pd.DataFrame(noisy_rows).to_csv(new_csv_path, index=False)
    print(f"Adding noise and brightnes complete. CSV saved, total flipped rows: {len(noisy_rows)}.")


save_flipped_images(image_dir, flipped_dir, csv_path)
save_noisy_images(image_dir, noise_dir, csv_path, './data/original/noise_plates.csv')
save_noisy_images(flipped_dir, flipped_noise_dir, './data/original/flipped_plates.csv', './data/original/flipped_noise_plates.csv')