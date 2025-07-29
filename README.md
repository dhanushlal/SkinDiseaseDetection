# 🧬 SkinDiseaseDetection

A deep learning-based web app for **automatic skin disease classification** using image processing and CNN (Convolutional Neural Networks). Supports multiple skin conditions like **acne**, **eczema**, **psoriasis**, and **ringworm**.

> 🎓 Developed as a mini-project at JSS Science and Technology University.

---

## 🚀 Features

- 📷 Upload skin disease images via web interface
- 🧠 Trained CNN model (MobileNetV2) for fast and accurate detection
- ⚙️ Preprocessing with histogram equalization for clarity
- 📊 Displays predicted disease with confidence score
- 🌐 Easy-to-use GUI for doctors, students, and researchers

---

## 🛠️ Tech Stack

- **Python**, **TensorFlow/Keras**, **OpenCV**
- **Flask** (Web interface)
- **MobileNetV2** pretrained CNN model
- **Matplotlib** for accuracy/loss graphs

---

## 📁 Folder Structure


---

## 🧪 How It Works

1. 📁 Upload a skin disease image
2. 🧼 Image preprocessing (resize, histogram equalization)
3. 🧠 Prediction using trained CNN (MobileNetV2)
4. 📃 Output disease label and confidence score

---

## 🖼️ Screenshots

> Home Page  
![Upload Page](screenshots/home.png)

> Prediction Result  
![Result Page](screenshots/result.png)

---

## 📦 Installation & Run

```bash
# Clone the repository
git clone https://github.com/dhanushlal/SkinDiseaseDetection.git
cd SkinDiseaseDetection

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
