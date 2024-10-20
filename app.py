# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import logging
import os
from OCRscript import ImprovedComprehensiveImageAnalyzer
# from improved_comprehensive_image_analyzer import ImprovedComprehensiveImageAnalyzer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize your analyzer
azure_endpoint = "https://ocr-testing-10.cognitiveservices.azure.com/"
azure_key = "d3893ea029aa4d52863a23d9bbce11d8"
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
        # Check if the image is sent as base64
        if 'image' in request.json:
            base64_image = request.json['image']
            image_data = base64.b64decode(base64_image.split(',')[1])
            img = Image.open(io.BytesIO(image_data))
        # Check if the image is sent as a file
        elif 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream)
        else:
            return jsonify({'error': 'No image provided'}), 400

        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Analyze the image
        result = analyzer.analyze_image(img_cv)

        return jsonify(result)

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
        app.run(host='0.0.0.0', port=6000)
    except Exception as e:
        logging.error(f"Error starting the Flask app: {str(e)}")


if __name__ == '__main__':
    run_app()
