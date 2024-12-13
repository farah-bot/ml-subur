import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

image_model_path = 'rice-leaf-disease-detection.keras'
image_model = load_model(image_model_path)
class_names = ['Bacterial leaf blight', 'Brown spot', 'Health', 'Leaf smut']
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

tabular_model = tf.keras.models.load_model('fine_tuned_dnn_model.keras')
scaler = joblib.load('trained_scaler.joblib')

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)
    predictions = image_model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    os.remove(file_path)

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence
    })

@app.route('/predict-tabular', methods=['POST'])
def predict_tabular():
    try:
        data = request.json
        new_data = pd.DataFrame({
            'Land Area': [data['Land Area']],
            'Rainfall': [data['Rainfall']],
            'Humidity': [data['Humidity']],
            'Average Temperature': [data['Average Temperature']]
        })

        new_data_for_prediction = new_data[['Land Area', 'Rainfall', 'Humidity', 'Average Temperature']]
        new_data_scaled = scaler.transform(new_data_for_prediction)

        prediction = tabular_model.predict(new_data_scaled)
        prediction_divided = "{:.2f}".format(float(prediction[0][0]) / 3)

        return jsonify({
            'status': 'success',
            'prediction': prediction_divided
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Server is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
