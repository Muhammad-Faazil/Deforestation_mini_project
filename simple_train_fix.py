import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2
import os
import matplotlib.pyplot as plt

def load_and_check_data(data_dir):
    """Load data and check what we have"""
    images = []
    labels = []
    
    print("Checking data structure...")
    for root, dirs, files in os.walk(data_dir):
        print(f"Found directory: {root} with {len(files)} files")
    
    for category in ['deforestation', 'no_deforestation']:
        path = os.path.join(data_dir, category)
        label = 1 if category == 'deforestation' else 0
        
        if os.path.exists(path):
            print(f"Loading {category} from {path}")
            for filename in os.listdir(path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (128, 128))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(label)
    
    print(f"\n📊 Dataset Summary:")
    print(f"Total images: {len(images)}")
    print(f"Deforestation (1): {np.sum(np.array(labels) == 1)}")
    print(f"No Deforestation (0): {np.sum(np.array(labels) == 0)}")
    
    return np.array(images), np.array(labels)

def create_simple_model():
    """Create a simpler model that won't overfit as easily"""
    model = Sequential([
        # First block
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    return model

def main():
    # Load data
    data_dir = r"C:\Users\faazi\Desktop\Deforestation_CNN\data"
    X, y = load_and_check_data(data_dir)
    
    if len(X) == 0:
        print("❌ No images found! Please check your data directory structure.")
        return
    
    # Normalize
    X = X / 255.0
    
    # Split data - use stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 2)
    
    print(f"\n📈 Training set: {X_train.shape[0]} images")
    print(f"📈 Test set: {X_test.shape[0]} images")
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"🎯 Class weights: {class_weight_dict}")
    
    # Create model
    model = create_simple_model()
    
    # Compile with higher learning rate initially
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Higher learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Simple data augmentation (not too aggressive)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
    ]
    
    print("\n🚀 Starting training...")
    
    # Train without augmentation first to see if model can learn
    print("Phase 1: Training without augmentation...")
    history1 = model.fit(
        X_train, y_train_cat,
        batch_size=16,
        epochs=20,
        validation_data=(X_test, y_test_cat),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # If that works, continue with augmentation
    final_val_acc = history1.history['val_accuracy'][-1]
    if final_val_acc > 0.6:
        print(f"✅ Good progress! Validation accuracy: {final_val_acc:.4f}")
        print("Phase 2: Continuing with augmentation...")
        
        history2 = model.fit(
            datagen.flow(X_train, y_train_cat, batch_size=16),
            epochs=30,
            validation_data=(X_test, y_test_cat),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            initial_epoch=len(history1.epoch),
            verbose=1
        )
    else:
        print(f"❌ Poor performance. Let's try a different approach...")
        
        # Try a much simpler model
        print("Trying ultra-simple model...")
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            X_train, y_train_cat,
            batch_size=16,
            epochs=30,
            validation_data=(X_test, y_test_cat),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
    
    # Final evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Predictions analysis
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    confidences = np.max(y_pred, axis=1)
    
    print(f"\n📊 Confidence Analysis:")
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print(f"Confidence range: {np.min(confidences):.4f} - {np.max(confidences):.4f}")
    print(f"High confidence (>0.8): {np.mean(confidences > 0.8) * 100:.1f}%")
    print(f"Low confidence (<0.6): {np.mean(confidences < 0.6) * 100:.1f}%")
    
    # Check predictions per class
    unique, counts = np.unique(y_pred_classes, return_counts=True)
    print(f"\n📈 Predictions per class: {dict(zip(unique, counts))}")
    print(f"Actual test distribution: {np.unique(y_test, return_counts=True)}")
    
    # Save the model
    model.save('deforestation_model_fixed.keras')
    print("💾 Model saved as 'deforestation_model_fixed.keras'")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['accuracy'], label='Train Acc')
    plt.plot(history1.history['val_accuracy'], label='Val Acc')
    if 'history2' in locals():
        plt.plot(history2.history['accuracy'], label='Train Acc (Aug)')
        plt.plot(history2.history['val_accuracy'], label='Val Acc (Aug)')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('training_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()