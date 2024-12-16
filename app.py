import os
import threading
import json
import psutil
import traceback
import time

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename

from paddleocr import PaddleOCR
from easyocr import Reader
import cv2
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fuzzywuzzy import process

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Text extraction class
class TextExtractor:
    def __init__(self, confidence_threshold=0.5):
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        self.easyocr_reader = Reader(['en'])
        self.confidence_threshold = confidence_threshold

    def extract_text(self, image):
        try:
            if isinstance(image, bytes):
                # Convert bytes to numpy array for OpenCV
                nparr = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not read image")
            
            paddle_results = self.paddle_ocr.ocr(image, cls=True) or []
            paddle_text = [
                text[1][0]
                for line in paddle_results for text in line
                if len(text) > 1 and text[1][1] > self.confidence_threshold
            ]
            easyocr_results = self.easyocr_reader.readtext(image) or []
            easyocr_text = [
                text[1]
                for text in easyocr_results
                if text[2] > self.confidence_threshold
            ]
            return list(set(paddle_text + easyocr_text))
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return []

# LLM Setup with caching
class LLMCache:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            cls._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", trust_remote_code=True)
            cls._model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen1.5-1.8B-Chat",
                torch_dtype=torch.float16,  
                trust_remote_code=True
            )
            cls._model.to(cls._device)
        return cls._instance
    
    def generate_product_details(self, input_text):
        try:
            inputs = self._tokenizer(input_text, return_tensors="pt").to(self._device)
            outputs = self._model.generate(
                **inputs, 
                max_length=3700, 
                num_return_sequences=1, 
                temperature=0.7
            )
            result_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Attempt to parse the result as JSON
            try:
                parsed_details = json.loads(result_text)
                return parsed_details
            except (json.JSONDecodeError, ValueError):
                return {"raw_text": result_text}
        except Exception as e:
            print(f"LLM Generation Error: {e}")
            return {"error": str(e)}

# Health Check and Testing Route
@app.route('/health', methods=['GET'])
def health_check():
    """
    Comprehensive health check endpoint to verify system components
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "system_checks": {
            "ocr_engines": {
                "paddleocr": False,
                "easyocr": False
            },
            "llm_model": {
                "loaded": False,
                "device": str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            },
            "system_resources": {}
        },
        "test_results": {}
    }

    try:
        # Check OCR Engines
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # PaddleOCR Test
        try:
            paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            paddle_results = paddle_ocr.ocr(test_image, cls=True)
            health_status["system_checks"]["ocr_engines"]["paddleocr"] = True
        except Exception as e:
            health_status["system_checks"]["ocr_engines"]["paddleocr_error"] = str(e)
        
        # EasyOCR Test
        try:
            easyocr_reader = Reader(['en'])
            easyocr_results = easyocr_reader.readtext(test_image)
            health_status["system_checks"]["ocr_engines"]["easyocr"] = True
        except Exception as e:
            health_status["system_checks"]["ocr_engines"]["easyocr_error"] = str(e)
        
        # LLM Model Check
        try:
            llm_cache = LLMCache()
            test_prompt = "Perform a system check."
            test_generation = llm_cache.generate_product_details(test_prompt)
            health_status["system_checks"]["llm_model"]["loaded"] = True
            health_status["system_checks"]["llm_model"]["test_generation"] = bool(test_generation)
        except Exception as e:
            health_status["system_checks"]["llm_model"]["error"] = str(e)
        
        # System Resources
        try:
            health_status["system_checks"]["system_resources"] = {
                "cpu_count": os.cpu_count(),
                "available_memory": f"{psutil.virtual_memory().available / (1024*1024*1024):.2f} GB",
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            }
        except Exception as e:
            health_status["system_checks"]["system_resources_error"] = str(e)
        
        # Simulated Test Scenarios
        test_scenarios = [
            {
                "name": "Basic Text Extraction",
                "input": "Brand: TestBrand\nWeight: 500g\nMRP: ₹99.99",
                "expected": ["Brand", "Weight", "MRP"]
            }
        ]
        
        health_status["test_results"]["scenarios"] = []
        
        for scenario in test_scenarios:
            result = {
                "name": scenario["name"],
                "passed": False
            }
            
            try:
                test_input = scenario["input"]
                test_llm_input = f"""Extract product information:
Text: {test_input}
Extract details."""
                
                llm_cache = LLMCache()
                generation_result = llm_cache.generate_product_details(test_llm_input)
                
                # Basic validation
                result["passed"] = all(
                    any(keyword in str(generation_result).lower() for keyword in scenario["expected"])
                )
                result["output"] = generation_result
            except Exception as e:
                result["error"] = str(e)
            
            health_status["test_results"]["scenarios"].append(result)
        
        # Overall Status
        health_status["status"] = "healthy" if all([
            health_status["system_checks"]["ocr_engines"]["paddleocr"],
            health_status["system_checks"]["ocr_engines"]["easyocr"],
            health_status["system_checks"]["llm_model"]["loaded"]
        ]) else "degraded"
        
        return jsonify(health_status), 200
    
    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)
        return jsonify(health_status), 500

# Define endpoint for file upload
@app.route('/extract_text', methods=['POST'])
def extract_text_endpoint():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    file = request.files['file']
    
    # If no file is selected, browser also submit an empty part without filename
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    
    # Read the image file
    file_bytes = file.read()
    
    try:
        # Extract text
        extractor = TextExtractor()
        extracted_text = extractor.extract_text(file_bytes)
        
        if extracted_text:
            # Generate product details using LLM
            llm_cache = LLMCache()
            input_for_llm = f""" Extract detailed product information from the following text:

Text: {' '.join(extracted_text)}
Please extract the following details:
- Brand
- Expiry Date (if exact date is not given see for best before )
- MRP (it can be structured as MRP10.00 where 10 is mrp )
- Net Weight
- Manufacturer
- Storage Conditions
Provide a structured response strictly in json format 
about the above details.
 
"""
            product_details = llm_cache.generate_product_details(input_for_llm)
            return jsonify({
                "extracted_text": extracted_text,
                "product_details": product_details
            }), 200
        else:
            return jsonify({"message": "No text extracted from the image."}), 200
    
    except Exception as e:
        return jsonify({"message": f"Error processing image: {str(e)}"}), 500

# Simple HTML upload form for testing
@app.route('/', methods=['GET'])
def upload_form():
    return '''
    <!doctype html>
    <title>Upload Product Image</title>
    <h1>Upload Product Image for Text Extraction</h1>
    <form method=post enctype=multipart/form-data action="/extract_text">
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <h2>System Health</h2>
    <p><a href="/health">Check System Health</a></p>
    '''

# Run the server
if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5500, debug=False, use_reloader=False)).start()