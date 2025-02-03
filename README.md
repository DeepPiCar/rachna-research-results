# DeepPi Car Electrobit Project

# Object and Face Detection with YOLOv8 and OpenCV

This project demonstrates real-time object and face detection using the YOLOv8 object detection model and OpenCV's face detection capabilities.  It captures video from your webcam, detects objects using YOLOv8, detects faces using Haar cascades, and displays the results with bounding boxes and labels.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Code Explanation](#code-explanation)



## Installation

1. **Clone the repository (optional):**  If you're using Git, you can clone the repository:

   ```bash
   git clone [https://github.com/](https://github.com/)[your_username]/[your_repository_name].git  # Replace with your repo URL
   cd [your_repository_name]

## Create a virtual environment
- python3 -m venv .venv  # Or python -m venv .venv depending on your setup
- source .venv/bin/activate  # On Windows: .venv\Scripts\activate

## Run the script
- python main.py

## Webcam Capture: 
- The script will automatically use your default webcam (usually camera index 0).

## Real-time Detection: 
- The script will display a window showing the video feed with detected objects and faces outlined. Object labels and confidence scores are displayed for detected objects, and "Face" is displayed for detected faces. The number of objects and faces detected are also shown.

# Quit:
- Press the 'q' key to exit the application.
![alt text](../test.png)

## Dependencies
- ultralytics: For using the YOLOv8 object detection model.
- opencv-python: For image processing, face detection (Haar cascades), and video capture.
- numpy: (Implicitly a dependency of opencv-python) For numerical operations on images.