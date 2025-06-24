from xml.dom.minidom import parse
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import csv


# Paths: existing .xml file and images dir, new .csv file
xml_path = './data/original/annotations.xml'
csv_path = './data/original/plates.csv'
image_dir = './data/original/photos'

# Paths for new data
flipped_dir = './data/original/flipped_photos'
noise_dir = './data/original/noise_photos'
flipped_noise_dir = './data/original/flipped_noise_photos'
processed_dir = './data/processed'

# List of new paths
folders = [flipped_dir, noise_dir, flipped_noise_dir, processed_dir]


# ==================== DATA AUGMENTATION ====================
def create_folders(folders):
    for folder in folders:
        if folder and not os.path.exists(folder):
            os.mkdir(folder)


# Write the plates location from .xml to .csv ('name', 'x_top_left', 'y_top_left', 'x_bottom_right', 'y_bottom_right')
def write_to_csv(xml_path, csv_path):
    dom = parse(xml_path)
    images = dom.getElementsByTagName('image')
    
    csvfile = open(csv_path, 'w', newline='')
    fieldnames = ['name', 'xtl', 'ytl', 'xbr', 'ybr', 'img_width', 'img_height']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for image in images:
        # Getg value from xml file
        name = image.getAttribute('name')
        xtl = float(image.getElementsByTagName('box')[0].getAttribute('xtl'))
        ytl = float(image.getElementsByTagName('box')[0].getAttribute('ytl'))
        xbr = float(image.getElementsByTagName('box')[0].getAttribute('xbr'))
        ybr = float(image.getElementsByTagName('box')[0].getAttribute('ybr'))
        img_width = float(image.getAttribute('width'))
        img_height = float(image.getAttribute('height'))

        writer.writerow({'name': name, 'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr, 'img_width': img_width, 'img_height': img_height})


def read_plates_csv(csv_path):
    # Load original CSV to pandas dataframe
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        name = row['name']
        xtl, ytl, xbr, ybr = row['xtl'], row['ytl'], row['xbr'], row['ybr']
        img_width, img_height = row['img_width'], row['img_height']

        yield name, xtl, ytl, xbr, ybr, img_width, img_height


def save_flipped_images(image_dir, flipped_dir, csv_path):
    flipped_rows = []
    for name, xtl, ytl, xbr, ybr, img_width, img_height in read_plates_csv(csv_path):
        # Load and flip image
        img_path = os.path.join(image_dir, name)
        img = cv2.imread(img_path)

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
    print(f"Flipping complete. CSV saved, total flipped rows: {len(flipped_rows)}.")


def add_noise_and_brightness(image):
    # Convert image to float32 and normalize to [0.0, 1.0]
    noisy_image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0

    # Apply random brightness and contrast
    noisy_image = tf.image.random_brightness(noisy_image, max_delta=0.2)
    noisy_image = tf.image.random_contrast(noisy_image, lower=0.8, upper=1.2)

    # Add Gaussian noise
    noise = tf.random.normal(shape=tf.shape(noisy_image), mean=0.0, stddev=0.05)
    noisy_image = noisy_image + noise

    # Keep values in range [0.0, 1.0]
    noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)

    # Convert back to uint8 format (0–255) for saving with OpenCV
    return (noisy_image * 255).numpy().astype("uint8")


def save_noisy_images(image_dir, noise_dir, csv_path, new_csv_path):
    noisy_rows = []
    for name, xtl, ytl, xbr, ybr, img_width, img_height in read_plates_csv(csv_path):
        image_path = os.path.join(image_dir, name)
        original_image = cv2.imread(image_path)

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

    # Save noisy data to CSV
    pd.DataFrame(noisy_rows).to_csv(new_csv_path, index=False)
    print(f"Adding noise and brightnes complete. CSV saved, total noisy rows: {len(noisy_rows)}.")


# ==================== EXECUTABLE ====================
def main():
    write_to_csv(xml_path, csv_path)
    create_folders(folders)
    save_flipped_images(image_dir, flipped_dir, csv_path)
    save_noisy_images(image_dir, noise_dir, csv_path, './data/original/noise_plates.csv')
    save_noisy_images(flipped_dir, flipped_noise_dir, './data/original/flipped_plates.csv', './data/original/flipped_noise_plates.csv')


if __name__ == "__main__":
    main()