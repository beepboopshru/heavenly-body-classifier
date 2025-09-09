# 🌌 Celestial Body Classifier

A Python project that uses a **Convolutional Neural Network (CNN)** to classify telescope images of **galaxies** and **stars**.  
This project was developed as a hands-on learning exercise for an undergraduate-level introduction to **machine learning** and **computer vision**.  

The final trained model achieved **85.42% accuracy** on the test dataset.  

---

## 📖 Project Overview

This project follows a standard machine learning workflow:

1. **Data Acquisition & Exploration** – Downloads and analyzes an image dataset from Kaggle.  
2. **Data Preprocessing** – Resizes, normalizes, and splits the dataset into training, validation, and test sets using TensorFlow’s data pipelines.  
3. **Model Building** – Defines a sequential CNN architecture using `tensorflow.keras`.  
4. **Model Training** – Trains the CNN and saves the learned weights to a `.keras` file.  
5. **Evaluation & Prediction** – Evaluates accuracy on unseen test data and provides a script to classify new, single images.  

---

## ⚙️ Setup and Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/beepboopshru/heavenly-body-classifier.git
cd heavenly-body-classifier
````

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create environment
python -m venv .venv
```

Activate it:

* **Windows**

```bash
.venv\Scripts\activate
```

* **macOS/Linux**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install tensorflow matplotlib pillow
```

### 4. Download the Data

The dataset is **not included** in this repository.

Download from Kaggle: *Galaxy, Star, and Planet Classification*.

Unzip and place the image folders inside a parent folder named `Cutout Files` in the project root.

Expected directory structure:

```text
heavenly-body-classifier/
├── Cutout Files/
│   ├── galaxy/
│   │   ├── Galaxy_1.jpg
│   │   └── ...
│   └── star/
│       ├── Star_1.jpg
│       └── ...
├── train_model.py
├── predict_image.py
└── README.md
```

---

## 🚀 How to Run

### Train the Model

Run the training script. This will preprocess the data, build and train the CNN, and save the model as `celestial_classifier.keras`.

```bash
python train_model.py
```

### Make Predictions

Once the model is trained, use the prediction script.

> **Important:** Open `predict_image.py` and set the `image_to_predict` variable to the image path you want to classify.

```bash
python predict_image.py
```

The script will:

1. Output the model’s accuracy on the test set.
2. Print the prediction for the selected image.

---

## 🔮 Future Improvements

This project serves as a baseline. Potential improvements include:

* **Improve Accuracy** – Achieve higher than 85% by fine-tuning.
* **Tune Hyperparameters** – Increase training epochs (e.g., from 15 → 25 or 30).
* **Enhance Architecture** – Add more `Conv2D` + `MaxPooling2D` layers for deeper feature learning.
* **Data Augmentation** – Use random flips, rotations, etc., to improve generalization.
* **Add More Classes** – Extend to nebulae, quasars, planetary datasets.
* **Handle Data Imbalance** – Use `class_weight` in `model.fit` to balance galaxy vs star classes.
* **User Interface** – Build a simple GUI with Tkinter, PyQt, or a web UI with Streamlit/Flask to allow drag-and-drop image classification.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

