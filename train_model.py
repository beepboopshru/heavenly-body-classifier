import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

def preprocess_and_split_data(data_dir, img_height, img_width, batch_size):
    """
    Loads, preprocesses, and splits the image data into training, validation,
    and test sets.
    """
    print("--- Starting Data Preprocessing ---")
    
    # Create a full dataset and split it into training (80%) and validation (20%).
    print("Step 1: Creating Training and Validation datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    class_names = train_ds.class_names
    print(f"\nFound the following classes: {class_names}")
    
    # Create a separate test set from the validation set.
    print("\nStep 2: Creating the Test dataset from the Validation set...")
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)
    
    print(f"Number of training batches: {tf.data.experimental.cardinality(train_ds)}")
    print(f"Number of validation batches: {tf.data.experimental.cardinality(val_ds)}")
    print(f"Number of test batches: {tf.data.experimental.cardinality(test_ds)}")
    
    # Optimize datasets for performance.
    print("\nStep 3: Optimizing datasets for performance...")
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("\n--- Preprocessing Complete ---")
    
    return train_ds, val_ds, test_ds, class_names

def build_model(num_classes):
    """
    Builds the Convolutional Neural Network (CNN) architecture.
    """
    print("--- Building the CNN Model ---")
    
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1) 
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    print("--- Model Built Successfully ---")
    model.summary()
    return model

if __name__ == '__main__':
    # --- Configuration ---
    DATA_FOLDER = 'Cutout Files'
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BATCH_SIZE = 32
    EPOCHS = 15

    # Step 1: Preprocess the data
    train_ds, val_ds, test_ds, class_names = preprocess_and_split_data(
        data_dir=DATA_FOLDER,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE
    )

    # Step 2: Build the CNN model
    model = build_model(num_classes=len(class_names))

    # Step 3: Train the model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    print("--- Model Training Complete ---")

    # Step 4: Save the trained model
    model_save_path = 'celestial_classifier.keras'
    model.save(model_save_path)
    print(f"\nModel saved successfully to {model_save_path}")

