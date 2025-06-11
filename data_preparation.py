from xml.dom.minidom import parse
import csv
import os
import random
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# LOADING DATA TO CSV
# writing the plates location from .xml to .csv ('name', 'x_top_left', 'y_top_left', 'x_bottom_right', 'y_bottom_right')
def write_to_csv(plates):
    csvfile = open('csv_plates.csv', 'w', newline='')
    fieldnames = ['name', 'xtl', 'ytl', 'xbr', 'ybr', 'img_width', 'img_height']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for plate in plates:
        # getting value from xml file
        name = plate.getAttribute('name')
        xtl = float(plate.getElementsByTagName('box')[0].getAttribute('xtl'))
        ytl = float(plate.getElementsByTagName('box')[0].getAttribute('ytl'))
        xbr = float(plate.getElementsByTagName('box')[0].getAttribute('xbr'))
        ybr = float(plate.getElementsByTagName('box')[0].getAttribute('ybr'))
        img_width = float(plate.getAttribute('width'))
        img_height = float(plate.getAttribute('height'))

        writer.writerow({'name': name, 'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr, 'img_width': img_width, 'img_height': img_height})

xml = parse('./data/annotations.xml')
plates = xml.getElementsByTagName('image')
# write_to_csv_normalized(plates)
# write_to_csv(plates)

# Create the list of dict [{'X: path, 'Y': xtl, ytl, xbr, ybr, width, height}]
def data_list(data_img_path):
    # data_img_path = "./data/photos/"
    images = os.listdir(data_img_path)
    all_data = []
    
    for img in images:
        img_path = os.path.join(data_img_path, img)

        with open('csv_plates.csv') as file:
            reader_file = csv.reader(file)
            for row in reader_file:
                if row[0] == img:
                    all_data.append({"X": img_path, "Y": row[1:]})
    return all_data
# all_data = data_list()


# IMAGE PRPROCESSING
# Image resising, normalize bbox data (0-1 value), output [{'X': img_array(for color), 'Y': xtl_norm, ytl_norm, xbr_norm, ybr_norm}]
def img_set_size(all_data, new_width, new_height):
    all_data_resized = []
    number_of_examples = len(all_data)
    processing_part = int(number_of_examples / 5)

    for i, example in enumerate(all_data):
        if (i + 1) % processing_part == 0:
            print(f'processing {i + 1}')

        # Load and resize image
        img = cv.imread(example['X'])
        resized_img = cv.resize(img, (new_width, new_height))

        # Normalize bbox values (0-1 value)
        xtl, ytl, xbr, ybr, img_width, img_height = map(float, example['Y'])
        xtl_norm = xtl / img_width
        ytl_norm = ytl / img_height
        xbr_norm = xbr / img_width
        ybr_norm = ybr / img_height

        # Save resized example
        all_data_resized.append({'X': resized_img, 'Y': [xtl_norm, ytl_norm, xbr_norm, ybr_norm], 'original_size': (img_width, img_height)})

    return all_data_resized

# Image to grayscale, shuffle final list, output [{'X': img_array(for grayscale), 'Y': xtl_norm, ytl_norm, xbr_norm, ybr_norm}]
def data_list_processed(all_data_resized):
    all_data_processed = []

    for img in all_data_resized:
        img_gray = cv.cvtColor(img['X'], cv.COLOR_BGR2GRAY)
        all_data_processed.append({"X": img_gray, "Y": img['Y'], "original_size": img['original_size']})

    # Shuffle the resized and converted to grayscale data list    
    random.seed(42)
    random.shuffle(all_data_processed)
    return all_data_processed

# Prepare the data list for CNN input, output X = [img_array(for grayscale)], Y = [xtl_norm, ytl_norm, xbr_norm, ybr_norm]
def normalize_data_input(all_data_processed):
    X = np.array([example['X'] for example in all_data_processed])
    Y = np.array([example['Y'] for example in all_data_processed])
    original_size = [example['original_size'] for example in all_data_processed]

    # normalize pixel value (0-1 value)
    X = np.array(X, dtype=np.float32) / 255.0
    Y = np.array(Y, dtype=np.float32)

    # 2D shape in grayscale, adding new dimension
    X = X[..., np.newaxis]

    print(f"Input feature (X): {X.shape}, target (Y): {Y.shape}")
    return X, Y, original_size


# Prepare sets: training - 70%, test - 15%, validation - 15%
def data_split(X, Y, original_size):
    split_ratio = 0.7
    split_idx = int(np.ceil(len(Y)*split_ratio))
  
    X_train = X[:split_idx,:,:,:]
    Y_train = Y[:split_idx,:]
    original_train = original_size[:split_idx]

    X_test_and_val = X[split_idx:,:,:,:]
    Y_test_and_val = Y[split_idx:,:]
    original_test_and_val = original_size[split_idx:]  
    

    half = int(len(Y_test_and_val)//2)
    X_test = X_test_and_val[:half,:,:,:]
    Y_test = Y_test_and_val[:half,:]
    original_test = original_test_and_val[:half]

    X_val = X_test_and_val[half:,:,:,:]    
    Y_val = Y_test_and_val[half:,:]
    original_val = original_test_and_val[half:]

    print(f"Number of examples in train set: {len(X_train)}\nNumber of examples in test set: {len(X_test)}\nNumber of examples in validation set: {len(X_val)}")
    return X_train, Y_train, X_test, Y_test, X_val, Y_val, original_train, original_test, original_val


def output_array_tolist(predicted_array):
    predicted_values = predicted_array.flatten().tolist()
    return predicted_values


def rescale_bbox(predicted_values, original_size):
    xtl_n, ytl_n, xbr_n, ybr_n = predicted_values
    img_width, img_height = original_size

    # Rescale back to original image size
    xtl = xtl_n * img_width
    ytl = ytl_n * img_height
    xbr = xbr_n * img_width
    ybr = ybr_n * img_height

    return [xtl, ytl, xbr, ybr]


def batch_generator(X, Y, batch_size):
    total_samples = X.shape[0]
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        yield X[i:end_idx], Y[i:end_idx]


# PLOTS
def plot_images_with_bounding_boxes(all_data):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
    axes = axes.flatten()
    num_examples = len(all_data)
    for ax in axes:
        idx = random.randint(0, num_examples - 1)
        example = all_data[idx]

        # Showing image, the point (0,0) is placed at the top-left corner
        img = Image.open(example['X'])
        ax.imshow(img)
        
        # Red bounding box
        xtl, ytl, xbr, ybr, img_width, img_height = map(float, example['Y'])
        width = xbr - xtl
        height = ybr - ytl    
        rect = patches.Rectangle((xtl, ytl), width, height, linewidth=2, edgecolor='r', facecolor='none')
        
        # Add rectangle to axis
        ax.add_patch(rect)
        
        # Set title (image filename)
        ax.set_title(example['X'].split('/')[-1])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

