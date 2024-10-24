import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os

# Caching the data loading step so it won't be reloaded every time
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

    # Compile the model
    optimizer = Adam(learning_rate=0.0001)  # Adam optimizer with specified learning rate
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Train the model with the specified parameters and save the best model as best_model.h5
def train_and_save_model(X_train, y_trainHot, X_test, y_testHot):
    # Create the model
    model = create_model()

    # Set up callbacks: Reduce learning rate on plateau and save the best model
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')

    # Train the model
    model.fit(
        X_train, y_trainHot,
        validation_data=(X_test, y_testHot),
        epochs=10,  # You can adjust this as needed
        batch_size=128,  # Batch size as specified
        callbacks=[reduce_lr, checkpoint],
        verbose=1
    )

    print("Training complete. Best model saved as 'best_model.keras'.")

# Call the function to train the model and save the best model
train_and_save_model(X_train, y_trainHot, X_test, y_testHot)