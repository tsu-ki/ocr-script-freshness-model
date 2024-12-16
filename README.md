# Critiscan's OCR Text Extraction Model
![image](https://github.com/user-attachments/assets/42c10f9f-065d-4fd4-a6db-8e02e5ce13fa)

## Flipkart Grid Hackathon Robotics Track Submission - Team sendittokrishnaa

## _Overview_

This repository contains the OCR-based text extraction model designed to read and interpret text from images. The model employs techniques for optical character recognition (OCR) and integrates multiple libraries to enhance accuracy and versatility. It has been optimized for extracting key product details such as brand, expiry date, and pricing information.
- [Link to Website Repository](https://github.com/aanushkaguptaa/critiscan)
- [Link to Fruit Quality Assessment Model](https://github.com/tsu-ki/Freshness-model)
- [Link to Item Counting and Brand Detection](https://github.com/tsu-ki/FMCGDetectron)

## _Key Features_

- **Multi-OCR Integration**: Combines PaddleOCR and EasyOCR for robust text extraction.
- **Advanced Text Parsing**: Uses a pre-trained large language model (LLM) to interpret and structure extracted text.
- **Real-time Processing**: Optimized for quick and efficient processing of images.
- **Flask API**: Provides a RESTful endpoint for text extraction and information parsing.

---

## _Technical Architecture_

### **1. Core Components**

- **OCR Frameworks**:
    - PaddleOCR for angle-sensitive text detection.
    - EasyOCR for multilingual text extraction.
- **Language Model Integration**:
    - Pre-trained LLM (Qwen1.5-1.8B-Chat) for structured information extraction.

### **2. Technical Specifications**

| Component            | Specification      |
| -------------------- | ------------------ |
| OCR Libraries        | PaddleOCR, EasyOCR |
| Language Model       | Qwen1.5-1.8B-Chat  |
| Processing Framework | Flask              |
| Deployment           | Docker + AWS EC2   |

### **3. Key Processing Capabilities**

- Simultaneous use of multiple OCR libraries for high accuracy.
- Structured output in JSON format with details such as brand, MRP, expiry date, etc.
- Error handling for invalid inputs and edge cases.

---

## _Performance Metrics_

- **Accuracy**: Combines multiple OCR outputs to minimize errors.
- **Output Format**: Provides structured JSON output for ease of integration.
- **Scalability**: Tested on cloud platforms for reliable performance.

---

## _Getting Started_

### **Prerequisites**

- Python 3.8+
- CUDA-capable GPU (recommended)
- Docker (optional)

### **Installation**

```
# Clone the repository
git clone https://github.com/tsu-ki/ocr-script

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

---

## _API Usage_

**Endpoint**: `/extract_text`

- **Method**: `GET`
- **Parameters**:
    - `image_path`: Path to the image file.

**Example Usage**:
```
http://0.0.0.0:8080/extract_text?image_path=/path/to/image.jpg
```

**Sample JSON Output**:
```
{
  "product_details": {
    "Brand": "Example Brand",
    "Expiry Date": "Best Before 12/2025",
    "MRP": "10.00",
    "Net Weight": "500g",
    "Manufacturer": "Example Manufacturer",
    "Storage Conditions": "Keep in a cool, dry place"
  }
}
```

---

## _Deployment_

- **Web Framework**: Flask
- **Containerization**: Docker
- **Cloud Platform**: AWS EC2

---

## _Future Development Roadmap_

- Enhance multilingual text extraction capabilities.
- Expand text parsing to include additional product details.
- Integrate more OCR libraries for specialized use cases.

---

## _References_

- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [Qwen1.5-1.8B-Chat Model](https://github.com/QwenLM/Qwen1.5-Chat)
- [FMCGDetectron Repository](https://github.com/tsu-ki/FMCGDetectron)
