Celestial Body Classifier
A Python project that uses a Convolutional Neural Network (CNN) to classify telescope images of galaxies and stars. This project was developed as a hands-on learning exercise for an undergraduate-level introduction to machine learning and computer vision.

The final trained model achieved 85.42% accuracy on the test dataset.

Project Overview
The project follows a standard machine learning workflow from start to finish:

Data Acquisition & Exploration: Downloads and analyzes an image dataset from Kaggle.

Data Preprocessing: Resizes, normalizes, and splits the data into training, validation, and test sets using TensorFlow's data pipelines.

Model Building: Defines a sequential CNN architecture using tensorflow.keras.

Model Training: Trains the CNN on the prepared dataset and saves the learned weights to a .keras file.

Evaluation & Prediction: Evaluates the model's accuracy on unseen test data and provides a script to classify new, single images.

Setup and Installation
Follow these steps to get the project running on your local machine.

1. Clone the Repository

git clone [https://github.com/beepboopshru/heavenly-body-classifier.git](https://github.com/beepboopshru/heavenly-body-classifier.git)
cd heavenly-body-classifier

2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

# Create the environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate

3. Install Dependencies
Install all the necessary Python libraries.

pip install tensorflow matplotlib pillow

4. Download the Data
The image data is not included in this repository.

Download the dataset from Kaggle: Galaxy, Star, and Planet Classification.

Unzip the downloaded file.

Place the image folders (galaxy, star) inside a parent folder named Cutout Files in the root of the project directory.

The final directory structure should look like this:

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

How to Run
1. Train the Model
Run the training script from your terminal. This will preprocess the data, build and train the model, and save the result as celestial_classifier.keras.

python train_model.py

2. Make Predictions
Once the model is trained, you can use the prediction script.

Important: Open the predict_image.py file and edit the image_to_predict variable to point to the image you want to classify.

python predict_image.py

The script will first output the model's overall accuracy on the test set and then print the prediction for your chosen image.

Future Improvements
This project serves as a great baseline. Here are several ways it could be extended and improved:

Improve Model Accuracy: The current 85% accuracy is a good start, but can be improved.

Tune Hyperparameters: Experiment with training for more EPOCHS (e.g., change from 15 to 25 or 30).

Enhance Architecture: Add more Conv2D and MaxPooling2D layers to the model in train_model.py to allow it to learn more complex features.

Data Augmentation: Implement data augmentation (e.g., random flips, rotations) to create more training examples from the existing data, which can help the model generalize better.

Add More Classes:

Find another dataset on Kaggle or other astronomy archives for different celestial bodies (e.g., Nebulae, Quasars, planetary images).

Modify the model to handle multi-class classification (e.g., change the final Dense layer to have num_classes units with a softmax activation and switch the loss function to SparseCategoricalCrossentropy).

Handle the Data Imbalance:

The current dataset has significantly more star images than galaxy images. This can bias the model.

Research and implement techniques like using the class_weight parameter during model training (model.fit) in TensorFlow to instruct the model to pay more attention to the under-represented 'galaxy' class.

Build a Simple UI:

Create a more user-friendly interface instead of requiring a code change to predict an image.

Use a Python GUI library like Tkinter (built-in), PyQt, or a simple web framework like Streamlit or Flask to build an application where a user can upload an image and see the prediction result in real-time.
