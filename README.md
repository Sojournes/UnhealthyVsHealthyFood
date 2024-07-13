# Food Classifier

This project is a food image classifier built using TensorFlow and Keras. The classifier can identify images as either healthy or unhealthy food.

## Dataset

The dataset contains images of healthy and unhealthy food, stored in separate folders. The images are preprocessed to ensure they have the correct format and size.

## Requirements

To run this project, you'll need the following libraries:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- OpenCV

You can install the required libraries using the following command:

```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Images
![image](https://github.com/user-attachments/assets/fa7526c0-a2fc-4533-864a-1b6b0c12e21f) ![image](https://github.com/user-attachments/assets/22934829-33e7-4e70-8c5c-8d839e0abce5)

![image](https://github.com/user-attachments/assets/da6d2d2d-d384-4a9b-921d-151a632ea041) ![image](https://github.com/user-attachments/assets/56df48e8-c221-49b0-8d0a-25f3356b70a2)


## Project Structure

```
.
├── Data
│   ├── healthy
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── unhealthy
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
├── models
│   └── myImageClassifier.keras
├── food_classifier.py
└── README.md
```

## Data Preprocessing

The images are loaded, checked for valid extensions, and removed if invalid. The dataset is then loaded using `tf.keras.utils.image_dataset_from_directory`, normalized, and split into training, validation, and test sets.

## Model Architecture

The neural network model is a Convolutional Neural Network (CNN) with the following layers:

- Conv2D and MaxPooling2D layers for feature extraction
- Flatten layer to convert the 2D feature maps to 1D
- Dense layers for classification

The model architecture is summarized as follows:

```plaintext
Layer (type)                     Output Shape                Param #
=================================================================
conv2d (Conv2D)                  (None, 254, 254, 16)        448
max_pooling2d (MaxPooling2D)     (None, 127, 127, 16)        0
conv2d_1 (Conv2D)                (None, 125, 125, 32)        4,640
max_pooling2d_1 (MaxPooling2D)   (None, 62, 62, 32)          0
conv2d_2 (Conv2D)                (None, 60, 60, 16)          4,624
max_pooling2d_2 (MaxPooling2D)   (None, 30, 30, 16)          0
flatten (Flatten)                (None, 14400)               0
dense (Dense)                    (None, 256)                 3,686,656
dense_1 (Dense)                  (None, 1)                   257
=================================================================
Total params: 3,696,625
Trainable params: 3,696,625
Non-trainable params: 0
```

## Training

The model is compiled with the Adam optimizer and binary cross-entropy loss. It is trained for 20 epochs with a validation split of 20%. TensorBoard is used for monitoring the training process.

## Evaluation

The model's performance is evaluated using precision, recall, and accuracy metrics on the test set. Sample predictions are visualized to demonstrate the classifier's performance.

## Running the Code

To run the project, execute the `food_classifier.py` script:

```bash
python food_classifier.py
```

## Visualization

The training and validation loss and accuracy are plotted for analysis. The script also includes functions to visualize some sample images along with their predicted labels.

## Model Inference

Sample images can be tested with the trained model to predict whether the food is healthy or unhealthy. The model is saved and loaded for inference.

## Save the Model

The trained model is saved in the `models` directory as `myImageClassifier.keras`.

## Future Work

- Increase the dataset size for better accuracy.
- Experiment with different model architectures and hyperparameters.
- Implement data augmentation to improve model generalization.
