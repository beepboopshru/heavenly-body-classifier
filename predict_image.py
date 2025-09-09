import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = 'celestial_classifier.keras'
IMG_HEIGHT = 128
IMG_WIDTH = 128
# The class names must be in the same order as the training data folders (alphabetical)
CLASS_NAMES = ['galaxy', 'star'] 

def evaluate_model():
    """
    Loads the test dataset and evaluates the model's performance on it.
    """
    print("--- Evaluating Model Performance ---")
    
    # We need to recreate the test dataset exactly as we did during training
    # so the model gets data in the format it expects.
    # Note: We are only creating the test portion here.
    print("Loading and preparing test dataset...")
    
    # First, create the full validation set (20% of the data)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'Cutout Files',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32)
        
    # Then, take the first half of it as the test set
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)

    # Optimize it for performance
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Load the trained model
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(test_ds)
    
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    
def predict_single_image(image_path):
    """
    Loads a single image, preprocesses it, and predicts its class.
    
    Args:
        image_path (str): The path to the image file.
    """
    print(f"\n--- Predicting Image: {image_path} ---")
    
    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load the image from the file path and resize it
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    
    # Convert the image to a numpy array
    img_array = tf.keras.utils.img_to_array(img)
    
    # The model expects a "batch" of images, so we add an extra dimension
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Make the prediction
    predictions = model.predict(img_array)
    
    # The output of our sigmoid neuron is a single number (a logit).
    # We can use a threshold of 0. If it's > 0, it's more likely class 1 ('star').
    # If it's < 0, it's more likely class 0 ('galaxy').
    score = tf.nn.sigmoid(predictions[0]) # Apply sigmoid to get a probability
    
    predicted_class_index = 0 if score < 0.5 else 1
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = 100 * (1 - score[0]) if predicted_class_index == 0 else 100 * score[0]

    print(f"This image is most likely a {predicted_class_name} with {confidence:.2f}% confidence.")


if __name__ == '__main__':
    # First, check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please run the 'train_model.py' script first to train and save the model.")
    else:
        # Part 1: Evaluate the model's overall accuracy on the test set
        evaluate_model()

        # Part 2: Predict a single image
        # You need to provide a path to an image you want to classify.
        # Find an image in your 'Cutout Files/galaxy' or 'Cutout Files/star' folder
        # and paste the full path here.
        # Example for Windows: 'Cutout Files\\galaxy\\Galaxy_1.jpg'
        # Example for Mac/Linux: 'Cutout Files/galaxy/Galaxy_1.jpg'
        
        # --- CHANGE THE PATH BELOW TO AN IMAGE YOU WANT TO TEST ---
        image_to_predict = 'C:\\Users\\shrey\\Downloads\\m82-1.jpg'
        # --- -------------------------------------------------- ---

        if os.path.exists(image_to_predict):
            predict_single_image(image_to_predict)
        else:
            print(f"\nError: The image path '{image_to_predict}' does not exist.")
            print("Please update the 'image_to_predict' variable with a valid file path.")
