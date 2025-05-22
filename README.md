# Real-Time Object Detection and Recognition

This project detects and recognizes objects in real-time using deep learning (MobileNet SSD + TensorFlow). Designed to assist visually impaired users and improve video surveillance.

## Features
- Real-time object detection using MobileNet SSD
- Video input from webcam or file
- Object tracking using OpenCV
- Web interface using Django

## Technologies Used
- Python, OpenCV, TensorFlow, Django
- Deep Learning (MobileNet SSD)

## Requirements
- opencv-python
- tensorflow
- flask
- imutils
- numpy
- Pillow

## Installation
```bash
git clone https://github.com/yourusername/real-time-object-detection.git
cd real-time-object-detection
pip install -r requirements.txt
```

## Run
```
python src/main.py         # For command-line
python app/app.py          # For web app
```

## Project Structure

- src/: Backend logic (detection, tracking)

- app/: Web application interface

- models/: Pre-trained models
