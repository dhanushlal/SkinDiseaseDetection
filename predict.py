import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Load the trained model
model = load_model("skin_disease_model.h5")

# List of class labels
class_labels = ['acne', 'eczema', 'healthy', 'psoriasis', 'ringworm']

# Load and preprocess the image
image_path = "sample.jpg"  # Make sure this image exists in the same folder
if not os.path.exists(image_path):
    print(f"❌ Image '{image_path}' not found.")
    exit()

img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))  # same size used in training
img = img / 255.0  # normalize
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_class = class_labels[np.argmax(prediction)]

print(f"✅ Predicted class: {predicted_class}")