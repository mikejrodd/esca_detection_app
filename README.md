# Esca Disease Detection in Grapevine Leaves

This application uses a pre-trained deep learning model to detect the presence of Esca disease in grapevine leaves based on leaf images. [Esca](https://ipm.ucanr.edu/agriculture/grape/esca-black-measles/#gsc.tab=0) is a serious fungal disease that affects grapevines, causing significant damage to vineyards. Early detection of Esca can help manage and control its spread, ensuring healthier vineyards and better grape yields.

## Requirements

- Python 3.6 or later
- Streamlit
- TensorFlow
- Hugging Face Transformers
- Pillow (PIL)
- NumPy

You can install the required packages using the following command:
```
pip install streamlit tensorflow huggingface-hub pillow numpy
```

## Usage

1. Clone the repository or download the code file.
2. Navigate to the project directory.
3. Run the following command to start the Streamlit app:
```
streamlit run esca_app.py
```

4. The application will open in your default web browser.
5. Upload a grape leaf image (JPG or PNG) using the file uploader.
6. Click the "Detect Esca" button to initiate the disease detection process.
7. The application will display the prediction result, indicating whether the leaf shows signs of Esca infection or not.

## How It Works

1. The application loads a pre-trained Keras model from Hugging Face (`mikejrodd/esca_grapeleaf_classifier`).
2. The uploaded image is preprocessed (resized, normalized, and converted to a NumPy array) to match the input requirements of the model.
3. The preprocessed image is passed through the model, which outputs a prediction score.
4. The prediction score is used to classify the leaf as either "esca" or "healthy." Healthy is defined by this classifier as "not showing signs of Esca infection," meaning other infections sucgh as blight or rot will be classified as healthy/non-Esca.
5. The result is displayed on the application interface, along with additional information and guidance.

## Model Details

The pre-trained model used in this application was trained on a dataset of [grapevine leaf images](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original), both healthy and infected with Esca and otherdisease. The model achieved the following performance metrics:

- **Precision (Esca)**: 0.79. When the model predicts Esca, it is correct 79% of the time.
- **Precision (Healthy)**: 0.99. When the model predicts healthy, it is correct 99% of the time.
- **Overall Accuracy**: 92%. The model correctly predicts both Esca and healthy leaves 92% of the time.

The model uses a custom focal loss function to handle the class imbalance in the training data.

## UI

<img width="1170" alt="Screenshot 2024-06-23 at 12 47 52 PM" src="https://github.com/mikejrodd/esca_detection_app/assets/137613726/1c308a5c-5a08-4bce-9052-bbcdaf619cc8">


<p align="center">
    <img width="682" alt="Screenshot 2024-06-23 at 12 48 39 PM" src="https://github.com/mikejrodd/esca_detection_app/assets/137613726/368cc824-e238-458d-91bf-6af613669e92">
</p>


## Contributing

Contributions to improve the application or the underlying model are welcome. If you encounter any issues or have suggestions for enhancements, please open an issue or submit a pull request on the project's GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The pre-trained model was created by [mikejrodd](https://huggingface.co/mikejrodd) and is available on the Hugging Face Hub.
- The application uses the Streamlit framework for building the interactive user interface.
- The TensorFlow and Keras libraries are used for loading and running the pre-trained model.
- The Hugging Face Transformers library is used for downloading the pre-trained model from the Hugging Face Hub.
- The Pillow (PIL) library is used for image processing and manipulation.
- The NumPy library is used for numerical operations on the image data.
