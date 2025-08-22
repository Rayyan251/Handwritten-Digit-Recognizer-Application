import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import glob
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

# Setting Paths

# Define the paths to the directories
training_directory_path = '/Users/fahadrashid/my_venv/ComputerVision/ModelDataForOCR/Train'  # Update with your training directory path
validation_directory_path = '/Users/fahadrashid/my_venv/ComputerVision/ModelDataForOCR/Validation'  # Update with your validation directory path

# -----------------------------------------------------------------------------------------------------
# This part is commented because I've already loaded the images and their labels in a pickle file
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------
# Loading Images and Seperating Labels for each image
# -----------------------------------------------------------------------------------------------------
# def load_images_and_labels_from_folders(directory_path):
#     images = []
#     labels = []
#     char_list = os.listdir(directory_path)  # List of character folders

#     for char in char_list:
#         char_folder_path = os.path.join(directory_path, char)
#         image_paths = glob.glob(os.path.join(char_folder_path, '*.jpg'))  # Change to '*.png' if your images are PNG files

#         for image_path in image_paths:
#             # Load image
#             img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 print(f"Failed to load image: {image_path}")
#                 continue
#             images.append(img)

#             # Use folder name as label
#             labels.append(char)

#     images = np.array(images)
#     #images = images / 255.0  # Normalize pixel values
#     images = np.expand_dims(images, axis=-1)  # Add a channel dimension

#     return images, labels

# # Load training and validation images and labels
# X_train, y_train = load_images_and_labels_from_folders(training_directory_path)
# X_validation, y_validation = load_images_and_labels_from_folders(validation_directory_path)

# -----------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------
# # Load the data from pickle files (for subsequent runs)
# ----------------------------------------------------------------------------------------------------
with open('C:\Python\Computer Vision\Handwritten Digit Detection Project\training_data.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)

with open('C:\Python\Computer Vision\Handwritten Digit Detection Project\validation_data.pkl', 'rb') as f:
    X_validation, y_validation = pickle.load(f)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_validation.shape}")

# ----------------------------------------------------------------------------------------------------
# Save the data to pickle files
# ----------------------------------------------------------------------------------------------------

# with open('training_data.pkl', 'wb') as f:
#     pickle.dump((X_train, y_train), f)

# with open('validation_data.pkl', 'wb') as f:
#     pickle.dump((X_validation, y_validation), f)




# Function to display a few images with their labels
def display_images(images, labels, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()

# Display a few training images with their labels
display_images(X_train[:5], y_train[:5])

# Function to display a few images with their labels and histograms
def display_images_with_histograms(images, labels, num_images=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        # Display image
        plt.subplot(num_images, 2, 2*i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')

        # Display histogram
        plt.subplot(num_images, 2, 2*i + 2)
        plt.hist(images[i].ravel(), bins=256, color='gray', alpha=0.7)
        plt.title(f'Histogram of {labels[i]}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Display a few training images with their histograms
display_images_with_histograms(X_train, y_train, num_images=5)

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_validation = X_validation.astype('float32') / 255.0

# Display a few training images with their histograms
display_images_with_histograms(X_train, y_train, num_images=5)

# ---------------------------------------------------------------------------------------------
# Using One-Hot Encoder
# ---------------------------------------------------------------------------------------------

# Create a LabelEncoder
label_encoder = LabelEncoder()
integer_encoded_y_train = label_encoder.fit_transform(y_train)
integer_encoded_y_validation = label_encoder.transform(y_validation)

# Create a OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)
y_train = onehot_encoder.fit_transform(integer_encoded_y_train.reshape(-1, 1))
y_validation = onehot_encoder.transform(integer_encoded_y_validation.reshape(-1, 1))

# ---------------------------------------------------------------------------------------------
# Creating the Model
# ---------------------------------------------------------------------------------------------

# Create the model using the Sequential API
model = keras.Sequential([
    # Convolutional Layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten the output of the convolutional layers
    layers.Flatten(),

    # Fully Connected Layers
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with softmax
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

# ---------------------------------------------------------------------------------------------
# Training the Model
# ---------------------------------------------------------------------------------------------

# # Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     epochs=10,  # Adjust the number of epochs as needed
#     validation_data=(X_val, y_val),
#     batch_size=32
# )

# # ---------------------------------------------------------------------------------------------
# # Evaluating the Model
# # ---------------------------------------------------------------------------------------------

# # Evaluate the model on the validation set
# loss, accuracy = model.evaluate(X_validation, y_validation, verbose=0)
# print(f'Validation Loss: {loss:.4f}')
# print(f'Validation Accuracy: {accuracy:.4f}')

# # Plot training history
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()

# ---------------------------------------------------------------------------------------------
# Loading The Model
# ---------------------------------------------------------------------------------------------

model_save_path = 'C:\Python\Computer Vision\Handwritten Digit Detection Project\my_ocr_model.h5'

# Load the model (with compile=False for older H5 files)
model = keras.models.load_model(model_save_path, compile=False) 

# ---------------------------------------------------------------------------------------------
# Predicting the Model
# ---------------------------------------------------------------------------------------------

# Predict on new images (e.g., from your test set)
predictions = model.predict(X_validation)

# Convert predicted probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Option 1: Convert y_validation to class labels
y_validation_labels = label_encoder.inverse_transform(np.argmax(y_validation, axis=1))
accuracy = accuracy_score(y_validation_labels, predicted_labels)

# Option 2: Use a OneHotEncoder that handles unseen categories
# onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # Or use 'use_default'
# predicted_labels_onehot = onehot_encoder.transform(predicted_labels.reshape(-1, 1))
# accuracy = accuracy_score(y_validation, predicted_labels_onehot)

print(f"Accuracy: {accuracy:.4f}")

# Display a few predictions and their actual labels
display_images(X_validation[:5], predicted_labels[:5])