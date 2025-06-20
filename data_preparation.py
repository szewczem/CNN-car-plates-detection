from xml.dom.minidom import parse
import csv
import os
import random
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ==================== LOADING DATA FROM XML TO CSV ====================
# writing the plates location from .xml to .csv ('name', 'x_top_left', 'y_top_left', 'x_bottom_right', 'y_bottom_right')
def write_to_csv(plates):
    csvfile = open('./data/original/plates.csv', 'w', newline='')
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


# ==================== LOADING DATA FROM CSV ====================
# List of .csv and .jpg path for coressponding samples
sources = [
    ("./data/original/plates.csv", "./data/original/photos/"),
    ("./data/original/flipped_plates.csv", "./data/original/flipped_photos/"),
    ("./data/original/noise_plates.csv", "./data/original/noise_photos/"),
    ("./data/original/flipped_noise_plates.csv", "./data/original/flipped_noise_photos/")
]


# Create the list of dict [{'X: path, 'Y': name, xtl, ytl, xbr, ybr, width, height}] for all sources
def data_list_from_multiple_csv(sources):
    '''
    data -- list of tuples, (csv_path, image_dir)
    '''
    all_data = []
    for csv_path, img_dir in sources :
        with open(csv_path) as file:
            reader_file = csv.reader(file)
            next(reader_file)  # skip header
            for row in reader_file:
                img_name = row[0]
                img_path = os.path.join(img_dir, img_name)
                if os.path.exists(img_path):
                    all_data.append({"X": img_path, "Y": row[:]})
    return all_data


# ==================== IMAGE PREPROCESSING ====================
# Image resising, normalize bbox data (0-1 value), output [{'X': img_array(for color), 'Y': xtl_norm, ytl_norm, xbr_norm, ybr_norm}]
def img_set_size(all_data, new_width, new_height):
    all_data_resized = []
    number_of_examples = len(all_data)
    parts = 5
    processing_part = int(number_of_examples / parts)
    numer_of_part = 0

    for i, example in enumerate(all_data):
        if (i + 1) % processing_part == 0:
            numer_of_part += 1
            print(f'processing {numer_of_part}/{parts}')

        # Load and resize image
        img = cv.imread(example['X'])
        resized_img = cv.resize(img, (new_width, new_height))

        # Get filename
        name = example['Y'][0]

        # Normalize bbox values (0-1 value)
        xtl, ytl, xbr, ybr, img_width, img_height = map(float, example['Y'][1:])
        xtl_norm = xtl / img_width
        ytl_norm = ytl / img_height
        xbr_norm = xbr / img_width
        ybr_norm = ybr / img_height

        # Save resized example
        all_data_resized.append({'X': resized_img, 'Y': [xtl_norm, ytl_norm, xbr_norm, ybr_norm], 'true_size_size': [img_width, img_height], 'filename': name})

    return all_data_resized


# Image to grayscale, shuffle final list, output [{'X': img_array(for grayscale), 'Y': xtl_norm, ytl_norm, xbr_norm, ybr_norm}]
def data_list_processed(all_data_resized):
    all_data_processed = []

    for img in all_data_resized:
        img_gray = cv.cvtColor(img['X'], cv.COLOR_BGR2GRAY)
        all_data_processed.append({"X": img_gray, "Y": img['Y'], "true_size_size": img['true_size_size'], 'filename': img['filename']})

    # Shuffle the resized and converted to grayscale data list    
    random.seed(42)
    random.shuffle(all_data_processed)
    return all_data_processed


# Prepare the data list for CNN input, output X = [img_array(for grayscale)], Y = [xtl_norm, ytl_norm, xbr_norm, ybr_norm]
def normalize_data_input(all_data_processed):
    X = np.array([example['X'] for example in all_data_processed])
    Y = np.array([example['Y'] for example in all_data_processed])
    true_size_size = np.array([example['true_size_size'] for example in all_data_processed])
    filename = [example['filename'] for example in all_data_processed]

    # normalize pixel value (0-1 value)
    X = np.array(X, dtype=np.float32) / 255.0
    Y = np.array(Y, dtype=np.float32)
    true_size_size = np.array(true_size_size, dtype=np.int32)

    # 2D shape in grayscale, adding new dimension
    X = X[..., np.newaxis]

    print(f"Input feature (X): {X.shape}, target (Y): {Y.shape}")
    return X, Y, true_size_size, filename


# Prepare sets: training - 70%, test - 15%, validation - 15%
def data_split(X, Y, true_size_size, filename):
    split_ratio = 0.7
    split_idx = int(np.ceil(len(Y)*split_ratio))
  
    # Training set
    X_train = X[:split_idx,:,:,:]
    Y_train = Y[:split_idx,:]
    true_size_train = true_size_size[:split_idx]
    filename_train = filename[:split_idx]

    # Test and Validation set
    X_test_and_val = X[split_idx:,:,:,:]
    Y_test_and_val = Y[split_idx:,:]
    true_size_test_and_val = true_size_size[split_idx:]
    filename_test_and_val = filename[split_idx:]    

    # Test set
    half = int(len(Y_test_and_val)//2)
    X_test = X_test_and_val[:half,:,:,:]
    Y_test = Y_test_and_val[:half,:]
    true_size_test = true_size_test_and_val[:half]
    filename_test = filename_test_and_val[:half]

    # Validation set
    X_val = X_test_and_val[half:,:,:,:]    
    Y_val = Y_test_and_val[half:,:]
    true_size_val = true_size_test_and_val[half:]
    filename_val= filename_test_and_val[half:]

    print(f"Number of examples in train set: {len(X_train)}\nNumber of examples in test set: {len(X_test)}\nNumber of examples in validation set: {len(X_val)}")
    return X_train, Y_train, X_test, Y_test, X_val, Y_val, true_size_train, true_size_test, true_size_val, filename_train, filename_test, filename_val


def save_prepared_data(sources, new_width, new_height):
    print("Starting data preprocesing...")

    # CSV to data list
    all_data = data_list_from_multiple_csv(sources)

    # Resize and normalize bbox
    all_data_resized = img_set_size(all_data, new_width, new_height)

    # Convert to garyscale and shuffle
    all_data_processed = data_list_processed(all_data_resized)

    # Prepare input for CNN
    X, Y, true_size_size, filename = normalize_data_input(all_data_processed)

    # Split for training, test and valid dataset
    X_train, Y_train, X_test, Y_test, X_val, Y_val, true_size_train, true_size_test, true_size_val, filename_train, filename_test, filename_val = data_split(X, Y, true_size_size, filename)

    # Save prepared stets
    # training set
    np.save('./data/processed/X_train.npy', X_train)
    np.save('./data/processed/Y_train.npy', Y_train)
    np.save('./data/processed/true_size_train.npy', true_size_train)
    np.save('./data/processed/filename_train.npy', filename_train)

    # test set
    np.save('./data/processed/X_test.npy', X_test)
    np.save('./data/processed/Y_test.npy', Y_test)
    np.save('./data/processed/true_size_test.npy', true_size_test)
    np.save('./data/processed/filename_test.npy', filename_test)

    # validation set
    np.save('./data/processed/X_val.npy', X_val)
    np.save('./data/processed/Y_val.npy', Y_val)
    np.save('./data/processed/true_size_val.npy', true_size_val)
    np.save('./data/processed/filename_val.npy', filename_val)

    print("Data preparation and saving complete.")


def load_prepared_data():
    X_train = np.load('./data/processed/X_train.npy')
    Y_train = np.load('./data/processed/Y_train.npy')
    X_test = np.load('./data/processed/X_test.npy')
    Y_test = np.load('./data/processed/Y_test.npy')
    X_val = np.load('./data/processed/X_val.npy')
    Y_val = np.load('./data/processed/Y_val.npy')
    true_size_train = np.load('./data/processed/true_size_train.npy')
    true_size_test = np.load('./data/processed/true_size_test.npy')
    true_size_val = np.load('./data/processed/true_size_val.npy')
    filename_train = np.load('./data/processed/filename_train.npy')
    filename_test = np.load('./data/processed/filename_test.npy')
    filename_val = np.load('./data/processed/filename_val.npy')
    print("Data loaded.")
    return X_train, Y_train, X_test, Y_test, X_val, Y_val, true_size_train, true_size_test, true_size_val, filename_train, filename_test, filename_val


def load_data(sources, img_width, img_height):
    if not os.path.exists('./data/processed/X_train.npy'):
        save_prepared_data(sources, img_width, img_height)
        return load_prepared_data()
    else:
        return load_prepared_data()


def output_array_tolist(predicted_array):
    predicted_values = predicted_array.flatten().tolist()
    return predicted_values


# def rescale_bbox(predicted_values, true_size):
#     pred = np.array(predicted_values)
#     true = np.array(true_size)

#     xtl = [pred[:, 0] * true[:, 0]]
#     ytl = [pred[:, 1] * true[:, 1]]
#     xbr = [pred[:, 2] * true[:, 0]]
#     ybr = [pred[:, 3] * true[:, 1]]

#     rescaled_bboxs = np.stack([xtl, ytl, xbr, ybr], axis=1)
#     return rescaled_bboxs


def rescale_bbox(predicted_values, true_size):
    rescaled_bboxs = []
    for bbox, size in zip(predicted_values, true_size):
        xtl_n, ytl_n, xbr_n, ybr_n = bbox
        img_width, img_height = size

        # Rescale back to original image size
        xtl = xtl_n * img_width
        ytl = ytl_n * img_height
        xbr = xbr_n * img_width
        ybr = ybr_n * img_height

        rescaled_bboxs.append([float(xtl), float(ytl), float(xbr), float(ybr)])
    return rescaled_bboxs


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
        xtl, ytl, xbr, ybr, img_width, img_height = map(float, example['Y'][1:])
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

