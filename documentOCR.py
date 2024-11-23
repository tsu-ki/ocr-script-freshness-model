import cv2
import numpy as np
import os
import re
from collections import defaultdict
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import pytesseract
import pdf2image
import spacy
import pandas as pd
import dateparser
import base64

def convert_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

class DocumentProcessor:
    def __init__(self, azure_key, azure_endpoint):
        self.azure_client = ImageAnalysisClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_key)
        )
        # Load SpaCy model for NER
        self.nlp = spacy.load("en_core_web_sm")
        self.tesseract_config = r'--oem 3 --psm 6'

    def process_document(self, file_path):
        """Main entry point for document processing"""
        if file_path.lower().endswith('.pdf'):
            return self.process_pdf(file_path)
        else:
            image = cv2.imread(file_path)
            return self.process_image(image)

    def process_pdf(self, pdf_path):
        """Process PDF documents"""
        pages = pdf2image.convert_from_path(pdf_path)
        results = []
        
        for i, page in enumerate(pages):
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            result = self.process_image(opencv_image)
            result['page_number'] = i + 1
            results.append(result)
            
        return {
            'document_type': 'PDF',
            'total_pages': len(pages),
            'pages': results
        }

    def process_image(self, image):
        """Process single image"""
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Extract text using multiple methods
        text_results = self.extract_text(preprocessed)
        
        # Extract structured information
        structured_info = self.extract_structured_info(text_results)
        
        # Detect tables
        tables = self.detect_tables(preprocessed)
        
        # Detect signatures
        signatures = self.detect_signatures(preprocessed)
        
        return {
            'structured_info': structured_info,
            'tables': tables,
            'signatures': signatures,
            'raw_text': text_results,
            'image_base64': convert_to_base64(image)
        }

    def preprocess_image(self, image):
        """Enhance image for better text extraction"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Deskew if needed
        angle = self.get_skew_angle(denoised)
        if abs(angle) > 0.5:
            denoised = self.rotate_image(denoised, angle)
            
        return denoised

    def extract_text(self, image):
        """Extract text using multiple OCR methods"""
        # Tesseract OCR
        tesseract_text = pytesseract.image_to_string(
            image, config=self.tesseract_config
        )
        
        # Azure OCR
        azure_text = self.extract_text_azure(image)
        
        # Combine results
        combined_text = self.merge_text_results([tesseract_text, azure_text])
        
        return combined_text

    def extract_structured_info(self, text):
        """Extract structured information using NLP"""
        doc = self.nlp(text)
        
        # Extract entities
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
            
        # Extract dates
        dates = self.extract_dates(text)
        
        # Extract emails
        emails = self.extract_emails(text)
        
        # Extract phone numbers
        phones = self.extract_phone_numbers(text)
        
        # Extract addresses
        addresses = self.extract_addresses(text)
        
        return {
            'entities': dict(entities),
            'dates': dates,
            'emails': emails,
            'phones': phones,
            'addresses': addresses
        }

    def detect_tables(self, image):
        """Detect and extract tables from the image"""
        # Find table-like structures
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100, 
            minLineLength=100, maxLineGap=10
        )
        
        if lines is None:
            return []
            
        # Group lines into potential tables
        tables = self.group_lines_into_tables(lines)
        
        # Extract content from each table
        extracted_tables = []
        for table_coords in tables:
            table_image = self.crop_table(image, table_coords)
            table_data = pytesseract.image_to_data(
                table_image, output_type=pytesseract.Output.DATAFRAME
            )
            extracted_tables.append(self.structure_table_data(table_data))
            
        return extracted_tables

    def detect_signatures(self, image):
        """Detect potential signatures in the document"""
        # Apply morphological operations to isolate signature-like components
        kernel = np.ones((5,5), np.uint8)
        morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        signatures = []
        for contour in contours:
            if self.is_signature_like(contour):
                x, y, w, h = cv2.boundingRect(contour)
                signatures.append({
                    'coordinates': (x, y, w, h),
                    'confidence': self.calculate_signature_confidence(
                        image[y:y+h, x:x+w]
                    )
                })
                
        return signatures

    # Helper methods
    def extract_text_azure(self, image):
        """Extract text using Azure's OCR"""
        try:
            _, img_encoded = cv2.imencode('.jpg', image)
            result = self.azure_client.analyze(
                image_data=img_encoded.tobytes(),
                visual_features=[VisualFeatures.READ]
            )
            
            if result.read:
                return ' '.join(
                    line.text for block in result.read.blocks 
                    for line in block.lines
                )
        except HttpResponseError as e:
            print(f"Azure API error: {e}")
        return ""

    def merge_text_results(self, text_list):
        """Merge and clean text from multiple sources"""
        combined = ' '.join(filter(None, text_list))
        # Clean up common OCR artifacts
        cleaned = re.sub(r'\s+', ' ', combined)
        cleaned = re.sub(r'[^\w\s@.,;:()\-\'\"]+', '', cleaned)
        return cleaned.strip()

    def extract_dates(self, text):
        """Extract dates from text"""
        potential_dates = []
        for match in re.finditer(
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            text
        ):
            date_str = match.group()
            parsed_date = dateparser.parse(date_str)
            if parsed_date:
                potential_dates.append({
                    'original': date_str,
                    'parsed': parsed_date.strftime('%Y-%m-%d')
                })
        return potential_dates
    
    def extract_emails(self, text):
        """Extract email addresses from text"""
        return re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)

    def extract_phone_numbers(self, text):
        """Extract phone numbers from text"""
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+?\d{1,3}[-.\s]?\d{9,10}',
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}'
        ]
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        return list(set(phones))  # Remove duplicates

    def extract_addresses(self, text):
        """Extract potential addresses using NLP and pattern matching"""
        doc = self.nlp(text)
        addresses = []
        
        # Look for address patterns
        address_pattern = r'\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\s+[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}'
        matches = re.finditer(address_pattern, text, re.IGNORECASE)
        
        for match in matches:
            addresses.append(match.group())
            
        # Also check for GPE (Geo-Political Entity) named entities
        address_components = []
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:
                address_components.append(ent.text)
                
        if address_components:
            addresses.extend(address_components)
            
        return list(set(addresses))

    def get_skew_angle(self, image):
        """Detect skew angle of the document"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None:
            return 0
            
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            if angle < 45:
                angles.append(angle)
            elif angle > 135:
                angles.append(angle - 180)
                
        if not angles:
            return 0
            
        return np.median(angles)

    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def group_lines_into_tables(self, lines):
        """Group detected lines into potential tables"""
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # Horizontal line
                horizontal_lines.append((x1, y1, x2, y2))
            elif abs(x2 - x1) < 10:  # Vertical line
                vertical_lines.append((x1, y1, x2, y2))
                
        # Group intersecting lines into table boundaries
        tables = []
        for h1 in horizontal_lines:
            for h2 in horizontal_lines:
                if h1 != h2:
                    for v1 in vertical_lines:
                        for v2 in vertical_lines:
                            if v1 != v2:
                                table = self.check_table_bounds(h1, h2, v1, v2)
                                if table:
                                    tables.append(table)
                                    
        return self.merge_overlapping_tables(tables)

    def check_table_bounds(self, h1, h2, v1, v2):
        """Check if four lines form a valid table boundary"""
        # Extract coordinates
        h1_x1, h1_y1, h1_x2, h1_y2 = h1
        h2_x1, h2_y1, h2_x2, h2_y2 = h2
        v1_x1, v1_y1, v1_x2, v1_y2 = v1
        v2_x1, v2_y1, v2_x2, v2_y2 = v2
        
        # Check if lines form a rectangle
        if (abs(h1_y1 - h2_y1) > 20 and  # Different y-coordinates
            abs(v1_x1 - v2_x1) > 20 and  # Different x-coordinates
            self.lines_intersect(h1, v1) and
            self.lines_intersect(h1, v2) and
            self.lines_intersect(h2, v1) and
            self.lines_intersect(h2, v2)):
            
            return (min(v1_x1, v2_x1), min(h1_y1, h2_y1),
                   max(v1_x2, v2_x2), max(h1_y2, h2_y2))
        return None

    def lines_intersect(self, line1, line2):
        """Check if two lines intersect"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denominator = ((x2 - x1) * (y4 - y3)) - ((y2 - y1) * (x4 - x3))
        if denominator == 0:
            return False
            
        t = (((x3 - x1) * (y4 - y3)) - ((y3 - y1) * (x4 - x3))) / denominator
        u = (((x3 - x1) * (y2 - y1)) - ((y3 - y1) * (x2 - x1))) / denominator
        
        return (0 <= t <= 1) and (0 <= u <= 1)

    def merge_overlapping_tables(self, tables):
        """Merge overlapping table boundaries"""
        if not tables:
            return []
            
        merged = []
        tables.sort()
        current = list(tables[0])
        
        for table in tables[1:]:
            if table[0] <= current[2]:  # Overlapping
                current[2] = max(current[2], table[2])
                current[3] = max(current[3], table[3])
            else:
                merged.append(tuple(current))
                current = list(table)
                
        merged.append(tuple(current))
        return merged

    def structure_table_data(self, table_df):
        """Convert raw OCR table data into structured format"""
        # Filter out low-confidence text
        table_df = table_df[table_df['conf'] > 30]
        
        # Group by lines
        lines = defaultdict(list)
        for _, row in table_df.iterrows():
            if row['text'].strip():
                lines[row['block_num']].append({
                    'text': row['text'],
                    'left': row['left'],
                    'top': row['top'],
                    'width': row['width'],
                    'height': row['height']
                })
                
        # Sort cells within each line by position
        structured_table = []
        for block_num in sorted(lines.keys()):
            line = sorted(lines[block_num], key=lambda x: x['left'])
            structured_table.append([cell['text'] for cell in line])
            
        return structured_table

    def is_signature_like(self, contour):
        """Check if a contour has signature-like characteristics"""
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        extent = float(area) / (w * h) if w * h != 0 else 0
        
        # Define signature characteristics
        return (area > 1000 and  # Not too small
                perimeter > 200 and  # Reasonable perimeter
                0.2 < aspect_ratio < 5 and  # Not too narrow or wide
                extent < 0.6)  # Not too solid

    def calculate_signature_confidence(self, signature_image):
        """Calculate confidence score for signature detection"""
        # Convert to binary
        _, binary = cv2.threshold(signature_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate features
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
            
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # Calculate complexity and density
        complexity = perimeter * perimeter / area if area > 0 else 0
        density = area / (signature_image.shape[0] * signature_image.shape[1])
        
        # Score based on typical signature characteristics
        score = min(1.0, max(0.0,
            0.3 * (1.0 if 10 < complexity < 100 else 0.0) +
            0.3 * (1.0 if 0.1 < density < 0.3 else 0.0) +
            0.4 * (1.0 if area > 1000 else area / 1000.0)
        ))
        
        return score

def process_document_batch(input_path, output_path, azure_key, azure_endpoint):
    """Process a batch of documents from input directory"""
    processor = DocumentProcessor(azure_key, azure_endpoint)
    results = []
    
    # Process all files in the input directory
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        if filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.tiff')):
            try:
                result = processor.process_document(file_path)
                result['filename'] = filename
                results.append(result)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                
    # Save results
    if results:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, 'extraction_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
    return results