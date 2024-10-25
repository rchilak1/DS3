import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import os
from datetime import datetime
import numpy as np
import pandas as pd

# Caching the data loading step so it won't be reloaded every time

def load_data():
    # Google Drive links for the files (replace with your own direct download links)
    X_train_link = 'https://drive.google.com/uc?id=1--m_zJGXNIzD4Q7Ozl9nimHWCJ8HAoE-'
    X_test_link = 'https://drive.google.com/uc?id=1-0MLEvsN-OeVHveNK5GXwfkw-t_18k45'
    y_trainHot_link = 'https://drive.google.com/uc?id=1-2FQR0BqDUqHhXSJc9k0iYn74-vJ7CMG'
    y_testHot_link = 'https://drive.google.com/uc?id=1-0iiFd75de7OPVb2BatpllwGD5x8Dm5u'

    # Download files if not already present
    if not os.path.exists('X_train.npy'):
        st.write("Downloading X_train.npy...")
        gdown.download(X_train_link, 'X_train.npy', quiet=False)
    if not os.path.exists('X_test.npy'):
        st.write("Downloading X_test.npy...")
        gdown.download(X_test_link, 'X_test.npy', quiet=False)
    if not os.path.exists('y_trainHot.npy'):
        st.write("Downloading y_trainHot.npy...")
        gdown.download(y_trainHot_link, 'y_trainHot.npy', quiet=False)
    if not os.path.exists('y_testHot.npy'):
        st.write("Downloading y_testHot.npy...")
        gdown.download(y_testHot_link, 'y_testHot.npy', quiet=False)

    # Load numpy arrays after downloading
    X_train = np.load('X_train.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_trainHot = np.load('y_trainHot.npy', allow_pickle=True)
    y_testHot = np.load('y_testHot.npy', allow_pickle=True)

    return X_train, X_test, y_trainHot, y_testHot


# Load data
X_train, X_test, y_trainHot, y_testHot = load_data()
print("Data loaded successfully!")

# Table to store parameter test results
results_table = []

# Custom function to create model
def create_model(activation='relu', dense_units=1024, dropout_rate=0.5):
    # Load MobileNetV2 without top layers and freeze layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False

    # Add custom top layers for classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  
    x = Dense(dense_units, activation=activation)(x)  # Fully connected layer with configurable activation
    x = Dropout(dropout_rate)(x)  # Dropout to prevent overfitting
    predictions = Dense(10, activation='softmax')(x)  # Output layer for 10 classes

   
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

# Function to compile and train model
def train_model(X_train, y_trainHot, X_test, y_testHot, learning_rate=0.0001, epochs=10, batch_size=32,
                activation='relu', dense_units=1024, dropout_rate=0.5, optimizer='adam', iteration=1):
    
    # Create model with given hyperparameters
    model = create_model(activation=activation, dense_units=dense_units, dropout_rate=dropout_rate)
    
    # Compile
    opt = Adam(learning_rate=learning_rate) if optimizer == 'adam' else tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Set up callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    # Train 
    history = model.fit(
        X_train, y_trainHot,
        validation_data=(X_test, y_testHot),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[reduce_lr],
        verbose=1
    )

    # Evaluate 
    train_acc = history.history['accuracy'][-1]  # Final training accuracy
    test_acc = model.evaluate(X_test, y_testHot, verbose=0)[1]  # Test accuracy
    
    # Log 
    results_table.append({
        'Iteration': iteration,
        'Number of Layers': 1,  # Fixed for MobileNetV2 in this case
        'Filter Size Layer 1': 'Global Average Pooling + Dense',  # MobileNetV2 uses preset filters
        'Activation Function': activation,
        'Dense Units': dense_units,
        'Dropout Rate': dropout_rate,
        'Learning Rate': learning_rate,
        'Batch Size': batch_size,
        'Optimizer': optimizer,
        'Train Accuracy': f"{train_acc*100:.2f}%",
        'Test Accuracy': f"{test_acc*100:.2f}%"
    })

    return model

# Expanded list of parameters for more combinations
param_combinations = [
    {'activation': 'relu', 'dense_units': 1024, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'batch_size': 32, 'optimizer': 'adam'},
    {'activation': 'tanh', 'dense_units': 512, 'dropout_rate': 0.3, 'learning_rate': 0.0001, 'batch_size': 64, 'optimizer': 'adam'},
    {'activation': 'sigmoid', 'dense_units': 256, 'dropout_rate': 0.4, 'learning_rate': 0.00001, 'batch_size': 32, 'optimizer': 'adam'},
    {'activation': 'relu', 'dense_units': 1024, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 16, 'optimizer': 'rmsprop'},
    {'activation': 'elu', 'dense_units': 512, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'batch_size': 64, 'optimizer': 'sgd'},
    {'activation': 'selu', 'dense_units': 1024, 'dropout_rate': 0.3, 'learning_rate': 0.0001, 'batch_size': 128, 'optimizer': 'adam'},
    {'activation': 'tanh', 'dense_units': 512, 'dropout_rate': 0.3, 'learning_rate': 0.0005, 'batch_size': 16, 'optimizer': 'adam'},
    {'activation': 'relu', 'dense_units': 256, 'dropout_rate': 0.4, 'learning_rate': 0.0005, 'batch_size': 128, 'optimizer': 'rmsprop'},
    {'activation': 'sigmoid', 'dense_units': 512, 'dropout_rate': 0.5, 'learning_rate': 0.001, 'batch_size': 64, 'optimizer': 'sgd'},
    {'activation': 'elu', 'dense_units': 1024, 'dropout_rate': 0.3, 'learning_rate': 0.0001, 'batch_size': 32, 'optimizer': 'adamax'}
]

# Train model with different hyperparameters
for iteration, params in enumerate(param_combinations, start=1):
    
    model = train_model(
        X_train, y_trainHot, X_test, y_testHot,
        learning_rate=params['learning_rate'], 
        epochs=10, 
        batch_size=params['batch_size'], 
        activation=params['activation'],  
        dense_units=params['dense_units'],   
        dropout_rate=params['dropout_rate'],  
        optimizer=params['optimizer'], 
        iteration=iteration
    )

# Convert results to DataFrame for tabular view
df_results = pd.DataFrame(results_table)

# Display result table
print(df_results)

# Save to a CSV file for reference
df_results.to_csv("parameter_tuning_results.csv", index=False)
