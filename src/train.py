import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def setup_gpu():
    """Setup GPU for training if available."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Enable memory growth for GPU
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Found {len(physical_devices)} GPU(s). Using GPU for training.")
            return True
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    else:
        print("No GPU found. Using CPU for training.")
        return False

def create_model(num_classes, use_gpu=False):
    """Create a CNN model for plant disease classification."""
    # Use ResNet50 as base model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use a higher learning rate if GPU is available
    learning_rate = 0.001 if use_gpu else 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(args):
    """Train the plant disease detection model."""
    # Setup GPU
    use_gpu = setup_gpu()
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    validation_generator = val_datagen.flow_from_directory(
        args.data_dir,
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Model checkpoint callback
    checkpoint_path = os.path.join(args.output_dir, 'best_model.keras')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Add TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.output_dir, 'logs'),
        histogram_freq=1
    )
    
    # Create and train model
    model = create_model(len(train_generator.class_indices), use_gpu)
    
    # Train the model
    model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, tensorboard_callback],
        workers=args.workers,
        use_multiprocessing=True
    )
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, 'final_model.keras')
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train plant disease detection model')
    parser.add_argument('--data_dir', required=True, help='Directory containing training data')
    parser.add_argument('--output_dir', default='models', help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    train_model(args)

if __name__ == '__main__':
    main() 