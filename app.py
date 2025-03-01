from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load Models
custom_model = tf.keras.models.load_model('model/model.h5')
pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    model_choice = request.form['model']

    if file.filename == '':
        return "No selected file"

    image = Image.open(file).resize((224, 224))
    image = np.array(image) / 255.0

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = np.expand_dims(image, axis=0)

    if model_choice == 'custom':
        image = tf.image.resize(image, (150, 150))  # Resize image to match custom model input
        prediction = custom_model.predict(image)
    else:
        prediction = pretrained_model.predict(image)

    result = "Healthy" if prediction[0][0] > 0.5 else "Unhealthy"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)