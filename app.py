# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
from PIL import Image
# from werkzeug.utils import secure_filename
# from werkzeug.urls import url_quote
import numpy as np
import cv2
import logging
import os
from OCRscript import ImprovedComprehensiveImageAnalyzer
# from improved_comprehensive_image_analyzer import ImprovedComprehensiveImageAnalyzer
import base64
import cv2
import numpy as np

def convert_image_to_base64(image):
    """Convert an OpenCV image (ndarray) to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image)  # Encode image as JPEG
    image_base64 = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string
    return image_base64

def ensure_json_serializable(data):
    """
    Recursively check a dictionary or list and convert non-serializable
    objects like NumPy arrays to serializable formats (e.g., base64 for images).
    """
    if isinstance(data, dict):
        # If data is a dictionary, check each key-value pair
        for key, value in data.items():
            data[key] = ensure_json_serializable(value)
    elif isinstance(data, list):
        # If data is a list, check each item
        data = [ensure_json_serializable(item) for item in data]
    elif isinstance(data, np.ndarray):
        # If data is a NumPy array (likely an image), convert to base64
        return convert_image_to_base64(data)
    return data

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize your analyzer
known_brands = [
        "Cannot Detect", "Fortune", "Dabur", "Colgate", "Maggi", "Surf Excel", "Amul",
        "Parle", "Sunfeast", "Good Day", "Marie Gold", "Lays", "Kurkure", "Bingo", "Haldirams",
        "Tata", "Britannia", "Haldiram's", "Mother Dairy", "Patanjali", "Nestlé",
        "ITC", "Hindustan Unilever", "Godrej", "Bisleri", "Cadbury", "Vicco", "Nestle",
        "Frooti", "Kissan", "MTR", "Lijjat", "Nirma", "Boroline", "Everest",
        "MDH", "Bournvita", "Hajmola", "Lifebuoy", "Clinic Plus", "Parachute",
        "Fevicol", "Pidilite", "Santoor", "Vim", "Saffola", "Trust",
        "Mondelez", "Ananda", "AASHIRVAAD",  "Uncle Chips", "Madhur", "Uttam",
        "Tata Tea", "Lipton", "Red Label", "Ariel", "Tide", "Dove", "Lux", "Pantene", "Head & Shoulders"
    ]
product_categories = {
    'dairy': ['Ghee'],
    'grains': ['Atta', 'Rice'],
    'sweeteners': ['Sugar'],
    'snacks': ['Biscuit', 'Namkeen'],
    'beverages': ['Tea', 'Cold Drinks'],
    'oils': ['Sunflower Oil', 'Mustard Oil', 'Groundnut Oil', 'Olive Oil'],
    'personal_care': ['Soap', 'Face Wash', 'Shampoo', 'Toothpaste', 'Toothbrush', 'Shaving Cream', 'Hair Oil'],
    'cleaning': ['Detergent'],
    'lentils': ['Dal', 'Toor Dal', 'Moong Dal', 'Chana Dal'],
    'dry_fruits_nuts': ['Dry Fruits', 'Almonds', 'Cashew Nuts', 'Dates & Raisins'],
    'pasta': ['Noodles', 'Pasta'],
    'confectionery': ['Chocolates', 'Sweets & Mithai'],
}

product_types = {
    'biscuits': ['biscuit', 'cookie', 'cracker', 'wafer'],
    'chips': ['chips', 'crisps', 'wafers', 'nachos'],
    'flour': ['atta', 'flour', 'multigrain'],
    'ghee': ['ghee', 'clarified butter'],
    'rice': ['basmati', 'long grain', 'brown rice', 'white rice'],
    'sugar': ['granulated', 'powdered', 'brown sugar'],
    'tea': ['black tea', 'green tea', 'herbal tea'],
    'cold_drinks': ['soda', 'cola', 'lemonade', 'fruit juice'],
    'oils': ['cooking oil', 'vegetable oil', 'olive oil', 'mustard oil'],
    'personal_care': ['soap', 'shampoo', 'toothpaste', 'face wash', 'shaving cream'],
    'detergents': ['powder detergent', 'liquid detergent', 'fabric softener'],
    'lentils': ['dal', 'pulses', 'legumes'],
    'dry_fruits_nuts': ['almonds', 'cashews', 'raisins', 'dates'],
    'noodles': ['instant noodles', 'pasta', 'spaghetti'],
    'chocolates': ['milk chocolate', 'dark chocolate', 'white chocolate'],
}

# analyzer = ImprovedComprehensiveImageAnalyzer(known_brands, product_categories, product_types, azure_key, azure_endpoint)
# Initialize the analyzer with your configurations
analyzer = ImprovedComprehensiveImageAnalyzer(
    known_brands=known_brands,
    product_categories=product_categories,
    product_types = product_types,
    azure_key= azure_key,
    azure_endpoint= azure_endpoint
)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        logging.info(f"Request Content-Type: {request.content_type}")
        logging.info(f"Request files: {request.files}")
        # Check if the image is sent as a base64 string in JSON
        if request.content_type == 'application/json' and 'image' in request.json:
            base64_image = request.json['image']
            logging.info(f"Received image file: {file.filename}")
            image_data = base64.b64decode(base64_image.split(',')[1])
            img = Image.open(io.BytesIO(image_data))
        # Check if the image is sent as a file (multipart/form-data)
        elif 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream)
        else:
            return jsonify({'error': 'No image provided'}), 400

        # Convert PIL Image to OpenCV format
        logging.info("Converting image to OpenCV format")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Analyze the image
        logging.info("Analyzing image")
        result = analyzer.analyze_image(img_cv)
        
        result_serializable = ensure_json_serializable(result)
        
        return jsonify(result_serializable)
        # return jsonify(result)

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Error processing image'}), 500

@app.route('/', methods=['GET'])
def test():
    return jsonify({'message': 'API is working'}), 200

def run_app():
    try:
        # Set up ngrok
        # public_url = ngrok.connect(5000)
        # print(f"Public URL: {public_url}")

        # Run the app on all available interfaces
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Error starting the Flask app: {str(e)}")


if __name__ == '__main__':
    run_app()
