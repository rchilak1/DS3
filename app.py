import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

# Add custom CSS to increase the font size of the slider label and value
st.markdown(
    """
    <style>
    .stSlider label {
        font-size: 200pixel; /* Increase the size for the slider label */
    }
    .st-af .css-1dp5vir {
        font-size: 200pixel; /* Increase the size for the slider value */
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
        st.subheader("Downloading X_train.npy...")
        gdown.download(X_train_link, 'X_train.npy', quiet=False)
    if not os.path.exists('X_test.npy'):
        st.subheader("Downloading X_test.npy...")
        gdown.download(X_test_link, 'X_test.npy', quiet=False)
    if not os.path.exists('y_trainHot.npy'):
        st.subheader("Downloading y_trainHot.npy...")
        gdown.download(y_trainHot_link, 'y_trainHot.npy', quiet=False)
    if not os.path.exists('y_testHot.npy'):
        st.subheader("Downloading y_testHot.npy...")
        gdown.download(y_testHot_link, 'y_testHot.npy', quiet=False)

    # Load the numpy arrays after downloading
    X_train = np.load('X_train.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_trainHot = np.load('y_trainHot.npy', allow_pickle=True)
    y_testHot = np.load('y_testHot.npy', allow_pickle=True)

    return X_train, X_test, y_trainHot, y_testHot

# Dictionary for mapping numeric labels to common names
monkey_labels = {
    0: 'mantled_howler',
    1: 'patas_monkey',
    2: 'bald_uakari',
    3: 'japanese_macaque',
    4: 'pygmy_marmoset',
    5: 'white_headed_capuchin',
    6: 'silvery_marmoset',
    7: 'common_squirrel_monkey',
    8: 'black_headed_night_monkey',
    9: 'nilgiri_langur'
}



# Cache the model loading process to avoid reloading the model every time
@st.cache_resource
def load_model():
    # Load the pre-trained best model (best_model.keras)
    if os.path.exists('best_model.keras'):
        model = tf.keras.models.load_model('best_model.keras')
        st.subheader("Loaded best model from cloud. History plot shown below")
    else:
        st.error("Pre-trained model 'best_model.keras' not found. Please upload the file.")
        return None
    return model

# Load the data
X_train, X_test, y_trainHot, y_testHot = load_data()
st.subheader("Dataset loaded successfully!")
st.subheader(' ')

# Load the trained model for inference
model = load_model()

st.image('epochInfo.png')

if model:
    # Evaluate the model's performance on the entire test data
    final_acc = model.evaluate(X_test, y_testHot, verbose=0)[1]
    st.subheader(f"Final Accuracy on test data: {final_acc*100:.2f}%")
    st.subheader(' ')

    # Cache predictions so they are not recomputed each time the slider moves
    @st.cache_data
    def get_predictions(model, X_test):
        return model.predict(X_test)

    # Get predictions for the entire test set
    y_pred = get_predictions(model, X_test)

    # **Subset selection for display purposes**:
    # Limit to 25 images for visualization (subset of test set)
    num_images_to_visualize = 25
    random_indices = np.random.choice(X_test.shape[0], num_images_to_visualize, replace=False)
    X_test_subset = X_test[random_indices]
    y_testHot_subset = y_testHot[random_indices]
    y_pred_subset = y_pred[random_indices]

    # Create a slider for selecting the index of the test image from the limited subset
    st.subheader(f"Visualizing {num_images_to_visualize} random test images and their predictions:")
    index = st.slider("Select test image index:", min_value=0, max_value=num_images_to_visualize - 1, value=0)

    # Display the selected test image and the predicted label
    st.image(X_test_subset[index], caption=f"Test Image at Index {index}", use_column_width=True)

    # Get the predicted label and the true label for the selected image
    predicted_label = np.argmax(y_pred_subset[index])
    true_label = np.argmax(y_testHot_subset[index])

    # Display the predicted and true label as common names
    predicted_common_name = monkey_labels[predicted_label]
    true_common_name = monkey_labels[true_label]

    # Display the predicted common name and the true common name
    st.subheader(f"Predicted Label: {predicted_common_name}")
    st.subheader(f"True Label: {true_common_name}")


st.subheader(' ')
st.subheader('Parameter Testing Log')
st.image('parameters.png')
