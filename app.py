import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

st.title("CS4372 Assignment 3")
st.subheader(' ')



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
        st.subheader(":grey[Loaded best model from cloud, inferenced on test data]")
        st.divider()
        if model:
            # Compute and display the overall test accuracy
            final_acc = compute_test_accuracy(model, test_dataset)
            st.subheader(f"Final Accuracy on test data: {final_acc*100:.2f}%")
           
    else:
        st.error("Pre-trained model 'best_model.keras' not found. Please upload the file.")
        return None
    return model

# Function to compute overall test accuracy
@st.cache_data
def compute_test_accuracy(_model, _test_dataset):
    # Evaluate the model on the test dataset
    results = model.evaluate(test_dataset, verbose=0)
    accuracy = results[1]  # Assuming 'accuracy' is the second metric
    return accuracy

# Load the test dataset
test_dataset, num_test_samples = load_test_dataset()
st.subheader(":gray[Test dataset loaded successfully!]")
st.subheader(' ')

# Load the trained model for inference
model = load_model()

st.subheader(' ')
st.subheader(':red[Click below for detailed _plots_, _metrics_, and _report_]')

with st.expander('Click Me!'):
    st.subheader('Epoch Accuracy Plot')
    st.image('epochAccuracy.png')
    st.text(' ')

    st.subheader('Epoch Loss Plot')
    st.image('epochLoss.png')
    st.text(' ')
    
    st.subheader('The above metrics display some interesting findings. As expected, the loss decreases each epoch while the accuracy increases, demonstrating an inverse relationship. Both the training and validation accuracy seem to flatten out around the 9th and 10th epochs, but perhaps more may see a slight increase. I will experiment further in the future. Similarly, the loss seems to flatten out around the 9th/10th epoch. Here the slope seems much closer to 0 though, depicting a heavier stagnation')
    st.text(' ')
    
    st.subheader('Epoch Precision Plot')
    st.image('epochPrecision.png')
    st.text(' ')

    st.subheader('Epoch Recall Plot')
    st.image('epochRecall.png')
    st.text(' ')
    
    st.subheader('Although the precision plot shows relatively low improvement after the 3rd epoch (for both sets), it does not seem horrible since it is around 90% consistently. On the other hand, the recall plot shows steady improvement throughout the training/validation cycles. It too however begins to stagnate at the 9th/10th epoch. ')
    st.text(' ')
    
    st.subheader('Per Class Accuracy')
    st.image('classAccuracy.png')
    st.text(' ')
    
    st.subheader('The per class accuracy has a high range, with the lowest being 76.92% (Nilgiri Langur) and the highest being 100% (Common Squirrel Monkey and Black Headed Monkey). I believe this is because the Nilgiri Langur closely resembles the Japanese Macaque, which also has relatively poor accuracy (80%). On the other hand, the Common Squirrel Monkey and Black Headed Monkey are much more distinct, hence the high value. ')
    st.text(' ')
    
    st.subheader('Bias Histogram')
    st.image('biasHistogram.png')
    st.text(' ')
    st.subheader('The bias histogram is as expected. Steadily decreases over epochs, demonstrating increased generalizability.')
    

    st.subheader('Parameter Testing Log')
    st.image('parameters.png')
    st.text(' ')
    st.subheader('The parameter log displays the different parameters I experimented with. MobileNetV2 requires the filter layer 1 to be global average pooling + dense, so I was not able to change it like in the instruction example. I compensated by adjusting the following:')
    st.subheader('Activation function - relu, tanh, sigmoid, elu')
    st.subheader('Number of dense units - 256, 512, 1024')
    st.subheader('Dropout Rate - 0.2, 0.3, 0.4, 0.5')
    st.subheader('Learning Rate - 0.001 through 0.00005')
    st.subheader('Batch size: 16,32,64,128')
    st.subheader('Optimizer: adam, rmsprop, sgd, adamax')
    st.subheader('Although it was close, below was the best combo, so it was used to generate best_model.keras (more info in ReadMe and GitHub')
    st.subheader('Selu, 1024, 0.3, 0.0001, 128, adam')  
    
    

    
   
   
   
    



if model:

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
    def get_predictions_subset(_model, X_subset):
        return model.predict(X_subset)

    # Get predictions for the subset
    y_pred_subset = get_predictions_subset(model, X_test_subset)

    st.divider()

    # Create a slider for selecting the index of the test image from the limited subset
    st.subheader(f"{num_images_to_visualize} random test images with predicted and true labels:")
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
    st.subheader(f":blue[Predicted Label:] {predicted_common_name}")
    st.subheader(f":blue[True Label:] {true_common_name}")



