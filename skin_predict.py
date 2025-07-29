from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("model.h5")  # Make sure this file is in your project folder

# Class labels (update these based on your trained model)
class_labels = ['Acne', 'Eczema', 'Psoriasis', 'Ringworm']

# File upload dialog
Tk().withdraw()
file_path = askopenfilename(title="Upload a Skin Image")

# Load and preprocess the image
img = cv2.imread(file_path)
img = cv2.resize(img, (224, 224))
img = img_to_array(img)
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

# Prediction
pred = model.predict(img)
index = np.argmax(pred)
confidence = round(np.max(pred) * 100, 2)

# Show result
print("Predicted Disease:", class_labels[index])
print("Confidence:", confidence, "%")
