import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from PIL import Image
import numpy as np

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss_value = -alpha_t * tf.math.pow(1 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss_value)
    return focal_loss_fixed

get_custom_objects().update({"focal_loss_fixed": focal_loss()})

# Download the model file from Hugging Face
model_id = "mikejrodd/esca_grapeleaf_classifier"
model_filename = "grapeleaf_classifier.keras"

model_file = hf_hub_download(repo_id=model_id, filename=model_filename)

model = load_model(model_file, custom_objects={"focal_loss_fixed": focal_loss()}, compile=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])

def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB to ensure 3 channels
    image = image.resize((150, 150))  # Resize to the input size expected by your model
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

st.sidebar.title("About Esca Disease")
st.sidebar.write("""
Esca is a serious fungal disease that affects grapevines, causing significant damage to vineyards. Early detection of Esca can help in managing and controlling its spread, ensuring healthier vineyards and better grape yields.

### How to Interpret the Results:
- **Yes, this leaf shows signs of an Esca infection**: 
  - **Precision**: 0.79. This means that when the model predicts Esca, it is correct 79% of the time. Thus, there is a 79% chance that your vine truly has Esca if the model indicates so.
  - **Accuracy**: The overall model accuracy is 92%, so the model correctly predicts both Esca and healthy leaves 92% of the time.

- **This leaf does not show signs of an Esca infection**:
  - **Precision**: 0.99. This means that when the model predicts healthy, it is correct 99% of the time. Thus, there is a 99% chance that your vine is truly free of Esca if the model indicates so.
  - **Accuracy**: The overall model accuracy is 92%, so the model correctly predicts both Esca and healthy leaves 92% of the time.
""")

st.title("Esca Disease Detection in Grapevine Leaves")
st.write("Upload a grape leaf image (JPG or PNG) to detect if it shows signs of Esca infection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Detect Esca"):
        img = preprocess_image(image)
        result = model.predict(img)

        label = 'esca' if result[0][0] < 0.5 else 'healthy'

        if label == 'esca':
            st.markdown("""
            <div style="text-align: center;">
                <h2 style="color: red;">Yes, this leaf shows signs of an Esca infection.</h2>
                <p>Please view <a href="https://ipm.ucanr.edu/agriculture/grape/esca-black-measles/#gsc.tab=0">this page</a> for more information.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center;">
                <h2 style="color: green;">This leaf does not show signs of an Esca infection.</h2>
                <p>Be aware that this model does not actively detect blight or rot, so your vine may still exhibit symptoms of other diseases.</p>
            </div>
            """, unsafe_allow_html=True)
