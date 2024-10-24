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

# Function to load the test data using tf.data.Dataset
@st.cache_resource
def load_test_dataset(batch_size=32):
    # Download links for X_test and y_testHot
    X_test_link = 'https://drive.google.com/uc?id=1-0MLEvsN-OeVHveNK5GXwfkw-t_18k45'
    y_testHot_link = 'https://drive.google.com/uc?id=1-0iiFd75de7OPVb2BatpllwGD5x8Dm5u'

    # Download the files if not already present
    if not os.path.exists('X_test.npy'):
        st.subheader("Downloading X_test.npy...")
        gdown.download(X_test_link, 'X_test.npy', quiet=False)
    if not os.path.exists('y_testHot.npy'):
        st.subheader("Downloading y_testHot.npy...")
        gdown.download(y_testHot_link, 'y_testHot.npy', quiet=False)

    # Load the numpy arrays after downloading
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_testHot = np.load('y_testHot.npy', allow_pickle=True)

    # Create a tf.data.Dataset from the test data
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_testHot))
    test_dataset = test_dataset.batch(batch_size)

    return test_dataset, X_test.shape[0]  # Return total number of test samples

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

# Function to compute overall test accuracy
@st.cache_data
def compute_test_accuracy(model, _test_dataset):
    # Evaluate the model on the test dataset
    results = model.evaluate(test_dataset, verbose=0)
    accuracy = results[1]  # Assuming 'accuracy' is the second metric
    return accuracy

# Load the test dataset
test_dataset, num_test_samples = load_test_dataset()
st.subheader("Test dataset loaded successfully!")
st.subheader(' ')

# Load the trained model for inference
model = load_model()

st.image('epochInfo.png')

if model:
    # Compute and display the overall test accuracy
    final_acc = compute_test_accuracy(model, test_dataset)
    st.subheader(f"Final Accuracy on test data: {final_acc*100:.2f}%")
    st.subheader(' ')

    # **Subset selection for display purposes**:
    # Limit to 25 images for visualization (subset of test set)
    num_images_to_visualize = 25
    # Seed the random number generator for consistent results
    np.random.seed(42)
    random_indices = np.random.choice(num_test_samples, num_images_to_visualize, replace=False)

    # Function to load the visualization subset
    @st.cache_resource
    def load_visualization_subset():
        # Load only the subset of test images and labels
        X_test_subset = np.load('X_test.npy', mmap_mode='r')[random_indices]
        y_testHot_subset = np.load('y_testHot.npy', mmap_mode='r')[random_indices]
        return X_test_subset, y_testHot_subset

    X_test_subset, y_testHot_subset = load_visualization_subset()

    # Cache predictions on the subset
    @st.cache_data
    def get_predictions_subset(model, X_subset):
        return model.predict(X_subset)

    # Get predictions for the subset
    y_pred_subset = get_predictions_subset(model, X_test_subset)

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
