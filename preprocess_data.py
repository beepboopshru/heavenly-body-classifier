import tensorflow as tf
import os

def preprocess_and_split_data(data_dir, img_height, img_width, batch_size):
    """
    Loads, preprocesses, and splits the image data into training, validation,
    and test sets.

    Args:
        data_dir (str): The path to the directory containing the image folders.
        img_height (int): The target height for all images.
        img_width (int): The target width for all images.
        batch_size (int): The number of images to process in a batch.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    print("--- Starting Data Preprocessing ---")
    
    # Keras's image_dataset_from_directory is a powerful utility.
    # It reads images from a directory that is structured with subdirectories,
    # where each subdirectory is a different class.
    # It will also automatically resize the images for us.
    
    # First, we create a full dataset and split it into training (80%) and validation (20%).
    # We use a 'seed' to ensure the split is the same every time we run this.
    print("Step 1: Creating Training and Validation datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,  # 20% of the data will be for validation
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
    
    # Let's get the class names that were automatically inferred from the folder names.
    class_names = train_ds.class_names
    print(f"\nFound the following classes: {class_names}")
    
    # Now, we need a separate test set. A common practice is to take a portion
    # of the validation set to use for final testing. Let's take half of it.
    print("\nStep 2: Creating the Test dataset from the Validation set...")
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)
    
    print(f"Number of training batches: {tf.data.experimental.cardinality(train_ds)}")
    print(f"Number of validation batches: {tf.data.experimental.cardinality(val_ds)}")
    print(f"Number of test batches: {tf.data.experimental.cardinality(test_ds)}")
    
    # For performance, we'll use .cache() and .prefetch().
    # .cache() keeps the images in memory after they're loaded off disk during the first epoch.
    # .prefetch() overlaps data preprocessing and model execution while training.
    # This is a standard optimization practice.
    print("\nStep 3: Optimizing datasets for performance...")
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("\n--- Preprocessing Complete ---")
    
    return train_ds, val_ds, test_ds, class_names


if __name__ == '__main__':
    # --- Configuration ---
    DATA_FOLDER = 'Cutout Files'
    # We'll make the images smaller to train the model faster.
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    # Batch size is the number of images the model will see at once.
    BATCH_SIZE = 32

    # Check if the data folder exists
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder '{DATA_FOLDER}' not found.")
    else:
        # Run the preprocessing function
        train_dataset, val_dataset, test_dataset, class_names = preprocess_and_split_data(
            data_dir=DATA_FOLDER,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            batch_size=BATCH_SIZE
        )
        
        # To verify, let's print the shape of one batch of images and labels.
        print("\nVerifying the output...")
        for images, labels in train_dataset.take(1):
            print("Shape of one batch of images:", images.shape)
            print("Shape of one batch of labels:", labels.shape)
        print(f"The class labels are: {class_names}")
