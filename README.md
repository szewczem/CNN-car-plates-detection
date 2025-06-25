# Car License Plate Detection from Scratch 

![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/Numpy-2.0.2-informational?logo=numpy)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.1-blue?logo=tensorflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Model Execution](#model-execution)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Summary](#summary)
- [License](#license)

## Description
This project focuses on building a convolutional neural network (CNN) from scratch for car license plate detection. A Keras-based model was also implemented to compare its learning performance with the custom model built from scratch.  
The dataset used for this task is a small, publicly available collection from Kaggle: [Poland Vehicle License Plate Dataset](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset?resource=download). To enhance data variability, basic augmentation techniques were applied, including horizontal flipping, Gaussian noise combined with random brightness and contrast adjustments, and their combinations.  
Hyperparameter tuning was conducted on the Keras model, with the resulting parameters also applied to the custom model. A custom metric for mean Intersection over Union (IoU) was implemented to ensure a fair comparison. Training, validation, and evaluation results were visualized using a Jupyter Notebook.

## Features

- Custom CNN implementation (no TensorFlow, PyTorch, or scikit-learn)
- Manual forward & backward passes
- Bounding box regression (4 values per image)
- IoU - based accuracy tracking
- Augmented dataset support (flipping, noise, etc.)
- Model save/load capability
- Custom training, validation, and testing implementation
- Hyperparameters tuning for Keras model (grid search)
- Saving checkpoint corresponding to the epoch that achieved the best mean IoU (highest accuracy)
- Early stopping to prevent overfitting and unnecessary training

## Dataset

The dataset includes labeled images of Polish license plates. Each image has 4 bounding box values: `(xtl, ytl, xbr, ybr)`.
Bounding box annotations were converted from XML to CSV format using the provided preprocessing scripts.

**Sources (after data augmentation):**
- Original photos
- Horizontally flipped images
- Noisy versions of original and flipped photos

Annotations are stored in `.csv` files, and images are grouped into corresponding folders.

## Model Architecture

The default CNN architecture includes the following layers:

- Conv2D → ReLU → MaxPool
- Conv2D → ReLU → MaxPool
- Flatten
- Dense (output 4 values)

You can customize layer structure using the `custom_model.py` API.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/car-plates-detection.git
cd car-plates-detection
```

2. Create and activate the Conda environment:
```
conda env create -f environment.yaml
conda activate ml_cpd
```

3. Prepare the data directory:  
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset?resource=download), then create the following directory structure and place the dataset files accordingly:
```
car-plates-detection/
├── data/
│   └── original/    
│       ├── photos/             # Place the downloaded images here (.jpg files)
│       └── annotations.xml     # Place the downloaded annotation file (.xml file)
...
```

4. Run the data augmentation script:

> [!WARNING]
> Ensure that the annotation file and image folder are properly organized before running the augmentation script.

```
python increase_samples.py
```

After completing these steps, the environment, dataset, and project structure will be correctly configured, allowing full access to the functionality of the project.

**Project Structure:**
```
car-plates-detection/
├── data/
│   ├── original/
│   │   ├──  flipped_noise_photos/           # New: augmented images (flipped + noise)
│   │   ├── flipped_photos/                  # New: augmented images (flipped)
│   │   ├── noise_photos/                    # New: augmented images (noise)
│   │   ├── photos/                          # Original images (.jpg files)
│   │   ├── annotations.xml                  # Original annotation file (.xml)
│   │   ├── flipped_noise_plates.csv         # New: augmented labels
│   │   ├── flipped_plates.csv               # New: augmented labels
│   │   ├── noise_plates.csv                 # New: augmented labels
│   │   └── plates.csv                       # New: original labels
│   └── processed/                           # New: folder for processed data outputs
├── bbox_accuracy.py                         # Script to calculate bounding box accuracy
├── Custom_model.ipynb                       # Jupyter notebook for custom CNN model
├── custom_model.py                          # Custom CNN model implementation
├── data_preparation.py                      # Data preparation, saving, loading and preprocessing utilities
├── Data_visualization_and_cnn_flow.ipynb    # Notebook for visualizing data and training flow
├── environment.yaml                         # Conda environment definition
├── hyperparameters_tuning.py                # Script for hyperparameter search
├── increase_samples.py                      # Script for data augmentation
├── Keras_model.ipynb                        # Jupyter notebook for Keras CNN model
├── keras_model.py                           # Keras CNN model implementation
├── layers.py                                # Custom layers for CNN
├── main.py                                  # Main script to run custom model training/evaluation 
└── README.md                                # Project documentation
```

## Model Execution

Once the environment and data are prepared, you can run and compare both models using the provided notebooks: 
- `Custom_model.ipynb` — custom CNN implementation
- `Keras_model.ipynb` — equivalent model built using the Keras API

### Custom vs Keras Model

The **custom model**, where layers are implemented using basic NumPy operations, is significantly slower due to the lack of low-level optimization. Performance can be improved by applying **vectorization techniques** or leveraging optimized functions from libraries such as **SciPy**.

> [!WARNING]
> Training the custom model is significantly slower than the Keras model due to the lack of low-level optimizations. On CPU, full training may take several minutes per epoch.

- The architecture of the custom model can be modified directly inside the notebook.
- The **Keras model** architecture is defined in `keras_model.py` and should be edited there before running the notebook.
- Both notebooks follow a consistent structure where hyperparameters are defined before training, allowing for **easy tuning and experimentation**. This enables straightforward comparison.

> [!TIP]
> If training fails due to NaNs or exploding gradients, try reducing the learning rate or validating data integrity.

## Hyperparameter Tuning

The `hyperparameters_tuning.py` script performs a grid search to evaluate different training configurations. By default, it searches over:
- `learning_rate`
- `batch_size`
- `dense_layer_config` (number and size of Dense layers)

These are defined at the top of the `hyperparameters_tuning.py` file. You can extend or modify the search space by editing these lists.

> [!IMPORTANT]
> If you introduce architectural changes (e.g., new convolutional layers or activation functions), you must update the grid_search() function accordingly to ensure the search loop builds models compatible with the new design.

The tuning process uses `val_mean_iou` (mean Intersection over Union for validation set) as the primary evaluation metric, which measures the overlap between predicted and ground truth bounding boxes.

After execution:

- The best-performing model (based on `val_mean_iou`) is saved to `saved_models/best_keras_as_custom_model.keras`
- All configurations that achieve `val_mean_iou > valid_iou` are logged to `saved_models/top_models_result.txt` for review and comparison

To find optimal hyperparameters, use:
```
python hyperparameters_tuning.py
```

## Results

Both models output the predicted coordinates of license plate bounding boxes `(xtl, ytl, xbr, ybr)`. Throughout training, each epoch logs the duration, loss value, and accuracy based on the **Intersection over Union (IoU) metric**.  
Model parameters are saved in the `saved_models` directory, with the checkpoint corresponding to the epoch that achieved the best mean IoU (highest accuracy). Testing and visualization use these best-performing model parameters.  

To prevent overfitting and unnecessary training, an early stopping mechanism is applied: if no improvement is detected in the validation IoU (greater than `min_delta`) for a defined number of epochs `(patience)`, training stops automatically.

Performance is visualized in the corresponding Jupyter Notebooks, including:
- Plots of training and validation loss
- Plots of training and validation IoU
- Visual comparisons of predicted vs. true bounding boxes for several random images

## Summary

The custom and Keras models show comparable performance in predicting license plate positions. Visualizations of predicted bounding boxes, along with loss and accuracy plots, indicate that the models are able to learn how to locate plates. Accuracy can be improved by tuning hyperparameters, adding more layers, or changing the number and size of filters. Using larger input image sizes could further improve learning performance.

Due to long training times, the custom model was trained on a reduced subset of the dataset. This means its performance is limited by the size and diversity of the data. Even with data augmentation, the dataset remains relatively low in variation, which may limit the models ability to generalize to more diverse or complex real-world images.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
