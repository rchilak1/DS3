import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import datetime

# Caching data loading step so it won't be reloaded every time
def load_data():
    # Load the numpy arrays after downloading
    X_train = np.load('X_train.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_trainHot = np.load('y_trainHot.npy', allow_pickle=True)
    y_testHot = np.load('y_testHot.npy', allow_pickle=True)

    return X_train, X_test, y_trainHot, y_testHot

# Load the data
X_train, X_test, y_trainHot, y_testHot = load_data()
print("Data loaded successfully!")

# Custom function to create and compile the model
def create_model():
    # Load MobileNetV2 without the top layers and freeze its layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  # Freeze the base model

    # Add custom top layers for the classification task
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling
    x = Dense(1024, activation='selu')(x)  # Fully connected layer with selu activation
    x = Dropout(0.3)(x)  # Dropout to prevent overfitting
    predictions = Dense(10, activation='softmax')(x)  # Output layer for 10 classes

    # Define the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model with additional metrics: precision, recall
    optimizer = Adam(learning_rate=0.0001)  # Adam optimizer with specified learning rate
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(name='precision'), 
                           tf.keras.metrics.Recall(name='recall')])

    return model

# Custom callback to compute per-class accuracy
class EvaluationMetricsCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        predictions = np.argmax(self.model.predict(X_test), axis=1)
        true_labels = np.argmax(y_testHot, axis=1)

        # Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predictions)

        # Per-Class Accuracy
        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        for i, accuracy in enumerate(class_accuracies):
            print(f"Accuracy for class {i}: {accuracy * 100:.2f}%")

        # Display classification report
        print(f"\nClassification Report:\n")
        print(classification_report(true_labels, predictions))

# Train model with the specified parameters and save the best model as best_model.keras
def train_and_save_model(X_train, y_trainHot, X_test, y_testHot):
    # Create model
    model = create_model()

    # Set up TensorBoard logging dir
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Set up other callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    

    evaluation_callback = EvaluationMetricsCallback()

    # Train 
    model.fit(
        X_train, y_trainHot,
        validation_data=(X_test, y_testHot),
        epochs=10,  # You can adjust this as needed
        batch_size=128,  # Batch size as specified
        callbacks=[reduce_lr, checkpoint, tensorboard_callback, evaluation_callback],  # Add the callbacks here
        verbose=1
    )

    print("Training complete. Best model saved as 'best_model.keras'.")

# Train and save best model
train_and_save_model(X_train, y_trainHot, X_test, y_testHot)
