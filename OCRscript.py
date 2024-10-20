import cv2
import numpy as np
import os
import re
from collections import Counter, defaultdict
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from transformers import AutoTokenizer, AutoModel
import logging

# def install_packages():
#     packages = ["paddlepaddle", "paddleocr", "fuzzywuzzy", "python-Levenshtein", "easyocr", "dateparser", "transformers", "albumentations"]
#     for package in packages:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install_packages()
from fuzzywuzzy import fuzz
import easyocr
from paddleocr import PaddleOCR
import dateparser

import base64

def convert_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)  # Encode image as JPEG
    image_base64 = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string
    return image_base64

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

class TextExtractor:
    def __init__(self, azure_key, azure_endpoint):
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.easyocr_reader = easyocr.Reader(['en'])
        self.paddle_conf_threshold = 0.5
        self.easyocr_conf_threshold = 0.5
        self.min_word_length = 2
        self.max_word_length = 30
        self.azure_client = ImageAnalysisClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_key)
        )

    def clean_text(self, text):
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()
        cleaned = ' '.join(word for word in cleaned.split()
                           if self.min_word_length <= len(word) <= self.max_word_length)
        return cleaned

    def extract_text_azure(self, image):
        _, img_encoded = cv2.imencode('.jpg', image)
        image_bytes = img_encoded.tobytes()

        try:
            result = self.azure_client.analyze(
                image_data=image_bytes,
                visual_features=[VisualFeatures.READ, VisualFeatures.TAGS]
           )

            extracted_text = []

            if result.read is not None:
                for block in result.read.blocks:
                    for line in block.lines:
                        cleaned_text = self.clean_text(line.text)
                        if cleaned_text:
                            bbox = line.bounding_polygon
                            if len(bbox) >= 2:
                                height = bbox[2].y - bbox[0].y
                                extracted_text.append((cleaned_text, height))

            if result.tags:
                for tag in result.tags:
                    if isinstance(tag, dict) and 'confidence' in tag and tag['confidence'] > 0.8:
                        extracted_text.append((tag['name'], None))
                    elif isinstance(tag, str):
                        extracted_text.append((tag, None))

            return extracted_text

        except HttpResponseError as e:
            print(f"Status code: {e.status_code}")
            print(f"Reason: {e.reason}")
            print(f"Message: {e.error.message}")
            return []

    def extract_text_azure_stream(self, image_stream):
        try:
           result = self.azure_client.analyze(
               image_data=image_stream,
               visual_features=[VisualFeatures.READ]
           )

           extracted_text = []

           if result.read is not None:
               for block in result.read.blocks:
                   cleaned_text = self.clean_text(block.text)
                   if cleaned_text:
                       bbox = block.bounding_polygon
                       if len(bbox) >= 2:
                           height = bbox[2].y - bbox[0].y
                           extracted_text.append((cleaned_text, height))

           return extracted_text

        except Exception as e:
            print(f"Error in Azure API call: {e}")
            return []

    def extract_text_paddle(self, image):
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        result = self.paddle_ocr.ocr(image, cls=True)
        extracted_text = []
        if result is not None:
            for line in result:
                if isinstance(line, list):
                    for word_info in line:
                        if isinstance(word_info, tuple) and len(word_info) > 1:
                            bbox = word_info[0]
                            text, confidence = word_info[1]
                            if confidence > self.paddle_conf_threshold:
                                cleaned_text = self.clean_text(text)
                                if cleaned_text:
                                    height = self._calculate_text_height(bbox)
                                    extracted_text.append((cleaned_text, height))
        return extracted_text

    def extract_text_easyocr(self, image):
        result = self.easyocr_reader.readtext(image)
        extracted_text = []
        for detection in result:
            bbox, text, conf = detection
            if conf > self.easyocr_conf_threshold:
                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    height = self._calculate_text_height(bbox)
                    extracted_text.append((cleaned_text, height))
        return extracted_text

    def _calculate_text_height(self, bbox):
        if isinstance(bbox, list) and len(bbox) == 4:
            return int(abs(bbox[3][1] - bbox[0][1]))
        return 0

    def extract_all_text(self, image):
        all_text = []
        all_text.extend(self.extract_text_azure(image) or [])
        all_text.extend(self.extract_text_paddle(image) or [])
        all_text.extend(self.extract_text_easyocr(image) or [])

        if not all_text:
            print("No text detected in the image.")
            return {}

        size_groups = defaultdict(set)
        for text, height in all_text:
            if text.strip():
                size_groups[height].add(text)

        return {size: list(texts) for size, texts in size_groups.items()}

class ProductInfoExtractor:
    def __init__(self, known_brands, product_categories, product_types):
        self.known_brands = known_brands
        self.product_categories = product_categories
        self.product_types = product_types

    def extract_info(self, all_text):
        brand, brand_score, category = self.find_brand_and_category(all_text)
        product_type = self.find_product_type(all_text, category)
        package_details = self.find_package_details(all_text)
        expiry_date = self.find_expiry_date(all_text)
        mrp = self.find_mrp(all_text)
        size_weight = self.find_size_weight(all_text)
        manufacturer = self.find_manufacturer(all_text)
        address = self.find_address(all_text)
        license_no = self.find_license_no(all_text)
        best_before = self.find_best_before(all_text)
        customer_care = self.find_customer_care(all_text)
        email = self.find_email(all_text)
        nutritional_values = self.find_nutritional_values(all_text)
        ingredients = self.find_ingredients(all_text)

        return {
            'brand': brand,
            'brand_confidence': brand_score / 100.0,
            'category': category,
            'product_type': product_type,
            'package_details': package_details,
            'expiry_date': expiry_date,
            'mrp': mrp,
            'size_weight': size_weight,
            'manufacturer': manufacturer,
            'address': address,
            'license_no': license_no,
            'best_before': best_before,
            'customer_care': customer_care,
            'email': email,
            'nutritional_values': nutritional_values,
            'ingredients': ingredients,
            'all_text': ' '.join(all_text)
        }

    def find_brand_and_category(self, all_text):
        best_match = None
        best_score = 0
        best_category = None

        combined_text = ' '.join(all_text).lower()

        for brand in self.known_brands:
            score = fuzz.partial_ratio(combined_text, brand.lower())
            if score > best_score and score > 80:
                best_score = score
                best_match = brand
                best_category = self.find_category_for_brand(brand)

        if best_match is None:
            for text in all_text:
                for brand in self.known_brands:
                    if brand.lower() in text.lower():
                        best_match = brand
                        best_category = self.find_category_for_brand(brand)
                        best_score = 100
                        break
                if best_match:
                    break

        return best_match, best_score, best_category

    def find_category_for_brand(self, brand):
        for category, brands in self.product_categories.items():
            if brand in brands:
                return category
        return None

    def find_product_type(self, all_text, category):
        if category and category in self.product_types:
            for text in all_text:
                for product_type in self.product_types[category]:
                    if product_type.lower() in text.lower():
                        return product_type
        return None

    def find_package_details(self, all_text):
        keywords = ['pack', 'packet', 'box', 'container']
        for text in all_text:
            if any(keyword in text.lower() for keyword in keywords):
                return text
        return None

    def find_expiry_date(self, all_text):
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec).*\d{2,4}',
            r'exp.*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'best before.*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        ]
        for text in all_text:
            for pattern in date_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    date_str = match.group(0)
                    try:
                        return dateparser.parse(date_str).strftime('%Y-%m-%d')
                    except:
                        pass
        return None

    def find_mrp(self, all_text):
        mrp_pattern = [
            r'M\.?R\.?P\.?\s*(?:Rs\.?|₹)?\s*(\d+(?:\.\d{2})?)\s*(?:/-)?(?:\s*\((?:Incl\.?|Including)\s+(?:of\s+)?all\s+taxes\))?',
            r'MRP\s*Rs\.?\s*(\d+(?:\.\d{2})?)',
            r'(?:Price|MRP)[:.]?\s*(?:Rs\.?|₹)?\s*(\d+(?:\.\d{2})?)'
        ]
        for text in all_text:
            for pattern in mrp_pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
        return None

    def find_size_weight(self, all_text):
        size_patterns = [
            r'(\d+(\.\d+)?\s*(ml|l|g|kg|gm))',
            r'(\d+(\.\d+)?\s*(ml|l|g|kg|gm)).*net\s*wt',
            r'net\s*wt.*(\d+(\.\d+)?\s*(ml|l|g|kg|gm))',
            r'NET\s*(?:WEIGHT|WT\.?|QUANTITY)[:.]?\s*(\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l))',
            r'(?:net\s*(?:weight|wt\.?|quantity)|(?:pack|pack\s*size))[:.]?\s*(\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l))',
            r'(\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l))(?:\s*NET)?'
        ]
        for text in all_text:
            for pattern in size_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
        return None

    def find_manufacturer(self, all_text):
        manufacturer_pattern = [
            r'(?:Mfd\.?|Manufactured|Marketed)\s*(?:for\s*&\s*)?(?:by|for)[:.]?\s*(.*?(?:Limited|Ltd|PVT\.?\s*LTD\.?))\.?',
            r'(?:Marketed|Manufactured)\s*by[:.]?\s*(.*?(?:Limited|Ltd|PVT\.?\s*LTD\.?))\.?',
            r'(.*?(?:Limited|Ltd|PVT\.?\s*LTD\.?))\s*(?:H\.?O\.?|Head\s*Office)[:.]?'
        ]
        for text in all_text:
            for pattern in manufacturer_pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        return None

    def find_address(self, all_text):
        address_pattern = [
            r'(?:Unit No\.?|Plot No\.?)[^,\n]*,[^,\n]*(?:,\s*[^,\n]+)*',
            r'(?:H\.?O\.?|Head\s*Office)[:.]?\s*(.*?(?:\d{6}|\d{3}\s*\d{3}))',
            r'(?:Address|Registered\s*Office)[:.]?\s*(.*?(?:\d{6}|\d{3}\s*\d{3}))'
        ]
        for text in all_text:
            for pattern in address_pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip() if match.groups() else match.group(0).strip()
        return None

    def find_license_no(self, all_text):
      license_pattern = [
            r'(?:Lic\.?|License)\s*No\.?[:.]?\s*(\d+(?:[-/]\d+)*)',
            r'fssai\s*(?:No\.?|License)[:.]?\s*(\d+)',
            r'(?:FSSAI|Lic\.?)\s*(?:No\.?|License)[:.]?\s*(\d+(?:[-/]\d+)*)'
        ]
      licenses = []
      for text in all_text:
            for pattern in license_pattern:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    licenses.append(match.group(1))
      return ', '.join(licenses) if licenses else None

    def find_best_before(self, all_text):
        best_before_pattern = [
            r'Best\s*Before[:.]?\s*(\d+\s*(?:months|days)(?:\s*from\s*(?:packaging|manufacture))?)',
            r'(?:Use\s*By|Best\s*Before)[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(?:Expiry|Best\s*Before)[:.]?\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-\s]\d{2,4})'
        ]
        for text in all_text:
            for pattern in best_before_pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
        return None

    def find_customer_care(self, all_text):
        customer_care_pattern = [
            r'(?:call|contact|customer\s*care)[:.]?\s*(\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4})',
            r'(?:Toll\s*Free|Customer\s*Care)[:.]?\s*(\d{4}[-\s]?\d{3}[-\s]?\d{3})',
            r'(?:Customer\s*Care|For\s*Feedback)[:.]?\s*([+]?\d{1,4}[-\s]?\d{3,4}[-\s]?\d{3,4}(?:[-\s]?\d{3,4})?)'
        ]
        for text in all_text:
            for pattern in customer_care_pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
        return None

    def find_email(self, all_text):
        email_pattern = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'(?:Email|E-mail)[:.]?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
            r'(?:contact|info|customer|care)@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
        ]
        for text in all_text:
            for pattern in email_pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1) if match.groups() else match.group(0)
        return None

    def find_nutritional_values(self, all_text):
        nutrition_pattern = (
            r'(?:Nutrition[a]?l?\s*(?:Information|Values?|Facts)[^:]*:?\s*)'
            r'((?:(?:Energy|Calories|Protein|Carbohydrate|Fat|Fibre|Sodium|Sugar).*?'
            r'(?:\d+(?:\.\d+)?\s*(?:g|mg|kcal|kJ)).*?\n?)+)'
        )
        text = ' '.join(all_text)
        match = re.search(nutrition_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            nutrition_text = match.group(1)
            nutrition_dict = {}
            for line in nutrition_text.split('\n'):
                parts = re.split(r'\s{2,}|\t|(?<=\D)(?=\d)', line.strip())
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value = ' '.join(parts[1:]).strip()
                    nutrition_dict[key] = value
            return nutrition_dict
        return None

    def find_ingredients(self, all_text):
        ingredients_pattern = [
            r'Ingredients?[:.]?\s*(.*?)(?=(?:\n\n|\Z))',
            r'(?:Ingredients?|Contents?)[:.]?\s*((?:(?!\b(?:CONTAINS|ALLERGEN)\b).)*)(?=\b(?:CONTAINS|ALLERGEN)\b|$)',
            r'Ingredients?[:.]?\s*((?:(?!(?:\bNutrition|\bManufactured|\bBest Before|\bStorage|\bMRP|\bNet\b)).)*)',
            r'(?:Ingredients|INGREDIENTS)[:.]?\s*((?:(?!\b(?:CONTAINS|ALLERGEN|Nutrition|Manufactured|Best Before)\b).)*)(?=\b(?:CONTAINS|ALLERGEN|Nutrition|Manufactured|Best Before)\b|$)'
        ]
        text = ' '.join(all_text)
        for pattern in ingredients_pattern:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ingredients = match.group(1).strip()
                return [ing.strip() for ing in re.split(r',|\(|\)', ingredients) if ing.strip()]
        return None

class ImageProcessor:
    @staticmethod
    def preprocess_image(image):
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        sharpened = cv2.filter2D(denoised, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
        return [gray, denoised, sharpened]

    @staticmethod
    def outline_color_changes(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlined_image = image.copy()
        cv2.drawContours(outlined_image, contours, -1, (0, 255, 0), 2)
        return outlined_image, contours

    @staticmethod
    def extract_dominant_colors(image, n_colors=5):
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        counts = Counter(labels.flatten())
        ordered_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        ordered_colors = [centers[i[0]] for i in ordered_colors[:n_colors]]
        return ordered_colors

class ImprovedComprehensiveImageAnalyzer:
    def __init__(self, known_brands, product_categories, product_types, azure_key, azure_endpoint):
        self.text_extractor = TextExtractor(azure_key, azure_endpoint)
        self.product_info_extractor = ProductInfoExtractor(known_brands, product_categories, product_types)
        self.azure_client = ImageAnalysisClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_key)
        )
        self.tinybert_tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        self.tinybert_model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")


    def analyze_image(self, image):
        preprocessed_images = ImageProcessor.preprocess_image(image)
        all_text = set()
        azure_result = None

        for img_version in preprocessed_images:
            size_groups = self.text_extractor.extract_all_text(img_version)
            for texts in size_groups.values():
                all_text.update(texts)
        logging.info(f"Extracted text: {all_text}")
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        logging.info("Analyzing image with Azure AI vision")
        azure_result = self.azure_client.analyze(
            image_data=image_bytes,
            visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
            gender_neutral_caption=True
        )

        if azure_result.read:
            for block in azure_result.read.blocks:
                for line in block.lines:
                    all_text.add(line.text)
        
        logging.info("Extracting product information from azure results")
        product_info = self.structure_text_with_models(list(all_text))
        product_info['all_text'] = list(all_text)

        if azure_result.caption:
            product_info['Package Description'] = azure_result.caption.text
            product_info['Confidence'] = azure_result.caption.confidence
        
        logging.info("Extracting dominant colors using Image Processor Module")
        outlined_image, _ = ImageProcessor.outline_color_changes(image)
        dominant_colors = ImageProcessor.extract_dominant_colors(image)

        return {
            'product_info': product_info,
            'outlined_image': convert_image_to_base64(outlined_image),
            'dominant_colors': dominant_colors,
        }

    def structure_text_with_models(self, text_list):
        product_info = {
            'brand': None,
            'brand_confidence': 0,
            'expiry_date': None,
            'mrp': None,
            'customer_care': None,
            'nutritional_values': {},
            'ingredients': None,
            'net_weight': None,
            'manufacturer': None,
            'storage_conditions': None,
            'best_before': None,
            'license_no': None,
            'all_text': ' '.join(text_list)
        }

        logging.info("Extracting entities and classifying text using TinyBERT")
        for text in text_list:

            category = self.classify_text(text)
            entities = self.extract_entities(text)

            for key, value in entities.items():
                if key in product_info and product_info[key] is None:
                    product_info[key] = value

            # Use existing methods for specific fields if not found by TinyBERT
            if product_info['expiry_date'] is None:
                product_info['expiry_date'] = self.product_info_extractor.find_expiry_date([text])
            if product_info['mrp'] is None:
                product_info['mrp'] = self.product_info_extractor.find_mrp([text])
            if product_info['customer_care'] is None:
                product_info['customer_care'] = self.product_info_extractor.find_customer_care([text])
            if product_info['ingredients'] is None:
                product_info['ingredients'] = self.product_info_extractor.find_ingredients([text])
            if product_info['net_weight'] is None:
                product_info['net_weight'] = self.product_info_extractor.find_size_weight([text])
            if product_info['manufacturer'] is None:
                product_info['manufacturer'] = self.product_info_extractor.find_manufacturer([text])
            if product_info['storage_conditions'] is None:
                product_info['storage_conditions'] = self.extract_storage_conditions(text)
            if product_info['best_before'] is None:
                product_info['best_before'] = self.product_info_extractor.find_best_before([text])
            if product_info['license_no'] is None:
                product_info['license_no'] = self.product_info_extractor.find_license_no([text])

            # Extract nutritional values
            nutrition_info = self.product_info_extractor.find_nutritional_values([text])
            if nutrition_info:
                product_info['nutritional_values'].update(nutrition_info)

        # Find brand and calculate confidence
        brand, brand_score, _ = self.product_info_extractor.find_brand_and_category(text_list)
        if brand:
            product_info['brand'] = brand
            product_info['brand_confidence'] = brand_score / 100.0

        return product_info


    def classify_text(self, text):
        inputs = self.tinybert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.tinybert_model(**inputs)
        # This is a placeholder. You would need to add a classification layer and fine-tune the model for your specific categories
        # For now, we'll just return a default category
        return 'product_info'

    def extract_entities(self, text):
        inputs = self.tinybert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.tinybert_model(**inputs)

        # Placeholder for entity extraction logic
        # This should be replaced with actual entity extraction based on the model outputs
        entities = {}

        # Brand extraction
        brand_match = re.search(r'\b(?:brand|company):\s*(\w+)', text, re.IGNORECASE)
        if brand_match:
            entities['brand'] = brand_match.group(1)

        # Product name extraction
        product_match = re.search(r'\b(?:product|item):\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
        if product_match:
            entities['product_name'] = product_match.group(1).strip()

        # Weight/Size extraction
        weight_patterns = [
            r'(\d+(\.\d+)?\s*(ml|l|g|kg|gm))',
            r'(\d+(\.\d+)?\s*(ml|l|g|kg|gm)).*net\s*wt',
            r'net\s*wt.*(\d+(\.\d+)?\s*(ml|l|g|kg|gm))',
            r'NET\s*(?:WEIGHT|WT\.?|QUANTITY)[:.]?\s*(\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l))',
            r'(?:net\s*(?:weight|wt\.?|quantity)|(?:pack|pack\s*size))[:.]?\s*(\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l))',
            r'(\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l))(?:\s*NET)?'
        ]
        weight_pattern = '|'.join(weight_patterns)
        weight_match = re.search(weight_pattern, text, re.IGNORECASE)
        if weight_match:
            entities['weight'] = weight_match.group(1)

        # Price extraction
        price_patterns = [
            r'M\.?R\.?P\.?\s*(?:Rs\.?|₹)?\s*(\d+(?:\.\d{2})?)\s*(?:/-)?(?:\s*\((?:Incl\.?|Including)\s+(?:of\s+)?all\s+taxes\))?',
            r'MRP\s*Rs\.?\s*(\d+(?:\.\d{2})?)',
            r'(?:Price|MRP)[:.]?\s*(?:Rs\.?|₹)?\s*(\d+(?:\.\d{2})?)'
        ]
        price_pattern = '|'.join(price_patterns)
        price_match = re.search(price_pattern, text, re.IGNORECASE)

        # Expiry date extraction
        expiry_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec).*\d{2,4}',
            r'exp.*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'best before.*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        ]
        expiry_pattern = '|'.join(expiry_patterns)
        expiry_match = re.search(expiry_pattern, text, re.IGNORECASE)
        if expiry_match:
            entities['expiry_date'] = expiry_match.group(1)

        # Customer care extraction
        customer_care_patterns = [
            r'(?:call|contact|customer\s*care)[:.]?\s*(\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4})',
            r'(?:Toll\s*Free|Customer\s*Care)[:.]?\s*(\d{4}[-\s]?\d{3}[-\s]?\d{3})',
            r'(?:Customer\s*Care|For\s*Feedback)[:.]?\s*([+]?\d{1,4}[-\s]?\d{3,4}[-\s]?\d{3,4}(?:[-\s]?\d{3,4})?)'
        ]
        customer_care_pattern = '|'.join(customer_care_patterns)
        customer_care_match = re.search(customer_care_pattern, text, re.IGNORECASE)
        if customer_care_match:
            entities['customer_care'] = customer_care_match.group(1)

        best_before_match = re.search(r'Best\s*Before.*?(\d+\s*months?)', text, re.IGNORECASE)
        if best_before_match:
            entities['best_before'] = best_before_match.group(1)

        license_match = re.search(r'Lic\.?\s*No\.?\s*([\d-]+)', text, re.IGNORECASE)
        if license_match:
            entities['license_no'] = license_match.group(1)

        return entities

    def extract_storage_conditions(self, text):
        storage_match = re.search(r'Storage\s*Conditions?:?\s*(.+)', text, re.IGNORECASE)
        if storage_match:
            return storage_match.group(1)
        return None

def process_images(folder_path, known_brands, product_categories, product_types, azure_key, azure_endpoint):
    analyzer = ImprovedComprehensiveImageAnalyzer(known_brands, product_categories, product_types, azure_key, azure_endpoint)
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            analysis_result = analyzer.analyze_image(image)
            analysis_result['filename'] = filename
            analysis_result['original_image'] = convert_image_to_base64(image)  # Convert image to base64
            results.append(analysis_result)

    return results
