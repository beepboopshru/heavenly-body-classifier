🌌 Celestial Body Classifier

A Python project that uses a Convolutional Neural Network (CNN) to classify telescope images of galaxies and stars.This project was developed as a hands-on learning exercise for an undergraduate-level introduction to machine learning and computer vision.

✨ The final trained model achieved 85.42% accuracy on the test dataset.

📖 Project Overview

This project follows a standard machine learning workflow from start to finish:

Data Acquisition & ExplorationDownloads and analyzes an image dataset from Kaggle.

Data PreprocessingResizes, normalizes, and splits the data into training, validation, and test sets using TensorFlow’s data pipelines.

Model BuildingDefines a sequential CNN architecture using tensorflow.keras.

Model TrainingTrains the CNN on the prepared dataset and saves the learned weights to a .keras file.

Evaluation & PredictionEvaluates the model on unseen test data and provides a script to classify new single images.

⚙️ Setup and Installation

Follow these steps to get the project running on your local machine.

1. Clone the Repository

git clone https://github.com/beepboopshru/heavenly-body-classifier.git
cd heavenly-body-classifier

2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

Create the environment:

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

Activate on macOS/Linux:

source .venv/bin/activate

3. Install Dependencies

pip install tensorflow matplotlib pillow

4. Download the Data

The dataset is not included in this repository.

Download the dataset from Kaggle: Galaxy, Star, and Planet Classification

Unzip the downloaded file.

Place the image folders (galaxy, star) inside a parent folder named Cutout Files in the project root.

Final structure:

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

🚀 How to Run

1. Train the Model

This will preprocess the data, build and train the model, and save the result as celestial_classifier.keras.

python train_model.py

2. Make Predictions

Once the model is trained, you can use the prediction script.

⚠️ Open the predict_image.py file and edit the image_to_predict variable to point to the image you want to classify.

Run:

python predict_image.py

The script will:

Output the model’s overall accuracy on the test set.

Print the prediction for your chosen image.

🔮 Future Improvements

This project serves as a baseline. Here are ways it could be extended and improved:

Improve Model Accuracy: 85% is a good start, but can be improved.

Tune Hyperparameters: Experiment with more epochs (e.g., 25–30).

Enhance Architecture: Add more Conv2D and MaxPooling2D layers.

Data Augmentation: Apply flips, rotations, and transformations to increase variety.

Add More Classes: Extend to nebulae, quasars, planetary images, etc.

Handle Data Imbalance: Use class_weight in model.fit to balance galaxy vs. star samples.

Build a Simple UI: Use Tkinter, PyQt, Streamlit, or Flask for an interactive image upload and prediction interface.

📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

🌠 Acknowledgments

TensorFlow

Kaggle Dataset: Galaxy, Star, and Planet Classification

Inspiration from astronomy and astrophysics research
