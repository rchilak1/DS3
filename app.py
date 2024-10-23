import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import os
import gdown

# Caching the data loading step so it won't be reloaded every time
@st.cache_data
def load_data():
    # Google Drive links for the files (replace with your own direct download links)
    X_train_link = 'https://drive.google.com/uc?id=1--m_zJGXNIzD4Q7Ozl9nimHWCJ8HAoE-'
    X_test_link = 'https://drive.google.com/uc?id=1-0MLEvsN-OeVHveNK5GXwfkw-t_18k45'
    y_trainHot_link = 'https://drive.google.com/uc?id=1-2FQR0BqDUqHhXSJc9k0iYn74-vJ7CMG'
    y_testHot_link = 'https://drive.google.com/uc?id=1-0iiFd75de7OPVb2BatpllwGD5x8Dm5u'

    # Download the files if not already present
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

    # Load the numpy arrays after downloading
    X_train = np.load('X_train.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_trainHot = np.load('y_trainHot.npy', allow_pickle=True)
    y_testHot = np.load('y_testHot.npy', allow_pickle=True)

    return X_train, X_test, y_trainHot, y_testHot

# Cache the model loading process to avoid reloading the model every time
@st.cache_resource
def load_model():
    # Load and set up the MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the data
X_train, X_test, y_trainHot, y_testHot = load_data()
st.write("Data loaded successfully!")

# Function to train the model
@st.cache_resource
def train_model():
    # Check if the model has already been trained and saved
    if os.path.exists('trained_model.h5'):
        model = tf.keras.models.load_model('trained_model.h5')
        st.write("Loaded trained model from disk.")
        return model

    # Load and compile the model
    model = load_model()

    # Initialize placeholders in Streamlit for dynamic updates
    log_placeholder = st.empty()

    # Store all epoch logs to display them cumulatively
    log_history = []

    # Custom callback to dynamically update logs in Streamlit
    class StreamlitProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Append the new epoch logs
            log_history.append(f"Epoch {epoch + 1}/{self.params['epochs']}\n"
                               f"Train Loss: {logs['loss']:.4f}, Train Accuracy: {logs['accuracy']:.4f}\n"
                               f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}\n")

            # Update the log placeholder with all epoch logs
            log_placeholder.text("\n".join(log_history))

    # Set up callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    # Train the model with the Streamlit callback
    model.fit(
        X_train, y_trainHot,
        validation_data=(X_test, y_testHot),
        epochs=10,
        batch_size=32,
        callbacks=[StreamlitProgress(), reduce_lr],
        verbose=0  # Turn off verbose so Streamlit handles it
    )

    # Save the entire trained model
    model.save('trained_model.h5')
    st.write("Model trained and saved to disk.")

    return model


# Get the trained model
model = train_model()

# After training is complete, show final accuracy
final_acc = model.evaluate(X_test, y_testHot, verbose=0)[1]
st.write(f"Final Accuracy on test data: {final_acc*100:.2f}%")

# Cache predictions so they are not recomputed each time the slider moves
@st.cache_data
def get_predictions(model, X_test):
    return model.predict(X_test)

# Get predictions and cache them
y_pred = get_predictions(model, X_test)

# Limit to 25 images for visualization
num_images_to_visualize = 25
random_indices = np.random.choice(X_test.shape[0], num_images_to_visualize, replace=False)
X_test_subset = X_test[random_indices]
y_testHot_subset = y_testHot[random_indices]
y_pred_subset = y_pred[random_indices]

# Create a slider for selecting the index of the test image from the limited subset
st.write(f"Visualizing {num_images_to_visualize} random test images and their predictions:")
index = st.slider("Select test image index:", min_value=0, max_value=num_images_to_visualize - 1, value=0)

# Display the selected test image and the predicted label
st.image(X_test_subset[index], caption=f"Test Image at Index {index}", use_column_width=True)

# Get the predicted label and the true label for the selected image
predicted_label = np.argmax(y_pred_subset[index])
true_label = np.argmax(y_testHot_subset[index])

# Display the predicted label and the true label
st.write(f"Predicted Label: {predicted_label}")
st.write(f"True Label: {true_label}")
