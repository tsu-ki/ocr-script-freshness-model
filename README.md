# Critiscan's OCR Product Detection Model
![image](https://github.com/user-attachments/assets/42c10f9f-065d-4fd4-a6db-8e02e5ce13fa)

## Flipkart Robotics Challenge Hackathon Submission

## _Overview_

A sophisticated multi-model detection system designed to extract detailed information from product images using advanced computer vision and machine learning techniques. This integrated detection solution combines multiple state-of-the-art models to provide comprehensive product analysis.
- [Link to Website Repository](https://github.com/aanushkaguptaa/critiscan)
- [Link to Fruit Quality Assessment Model](https://github.com/tsu-ki/Freshness-model)
## _Key Features_

- **Multi-Model Integration**: Combines YOLO, Detectron2, and custom deep learning models
- **Object Detection**: Identifies and counts multiple objects in a single frame
- **Brand Recognition**: Intelligent brand detection using custom trained models
- **Real-time Processing**: Optimized for live video stream analysis
- **Comprehensive Logging**: Detailed error tracking and performance monitoring
---
## _Technical Architecture_

#### **1. Model Components**

- **Object Detection**:
    - YOLO v8 for rapid object identification
    - Detectron2 Instance Segmentation for detailed object analysis
- **Brand Recognition**: Custom Keras-based classification model
- **Multimodal Analysis**: Qwen2 VL multimodal model for advanced image understanding

#### **2. Technical Specifications**

| Component             | Specification         |
| --------------------- | --------------------- |
| Object Detection      | YOLO v8               |
| Instance Segmentation | Detectron2 Mask R-CNN |
| Brand Detection       | Custom Keras Model    |
| Multimodal Model      | Qwen2 VL-2B-Instruct  |
| Processing Framework  | Flask                 |
| Deployment            | Docker + AWS EC2      |

#### **3. Key Processing Capabilities**

- Real-time video frame processing
- Adaptive frame skipping for performance optimization
- Multi-source object detection
- Automatic brand recognition
- Excel export of detection results

---
## _Performance Metrics_

- **Adaptive Frame Processing**: Processes every 2nd frame
- **Resolution**: Optimized to 640x480 for efficient processing
- **Object Detection Sources**:
    - YOLO Object Detection
    - Detectron2 Instance Segmentation
- **Logging**: Comprehensive error tracking and system monitoring
---
## _Getting Started_

#### **Prerequisites**

- Python 3.8+
- CUDA-capable GPU (recommended)
- Docker (optional)
#### **Installation**

```
`# Clone the repository
        git clone https://github.com/tsu-ki/ocr-script-freshness-model
# Install dependencies
        pip install -r requirements.txt
# Run the application
        python app.py`
```
---
## _Deployment_

- **Web Framework**: Flask
- **Containerization**: Docker
- **Cloud Platform**: AWS EC2
- **Streaming**: Real-time video feed with object detection

## _Future Development Roadmap_

- Expand object recognition capabilities
- Improve brand detection accuracy
- Optimize processing speed
- Develop more sophisticated multimodal analysis
---
## _References_

- [Detectron2 Documentation](https://detectron2.readthedocs.io/en/latest/)
- [Qwen2-VL Repo](https://github.com/QwenLM/Qwen2-VL)
