# ANPR & ATCC System

## Overview

This project is an Automatic Number Plate Recognition (ANPR) and Automated Traffic Control & Compliance (ATCC) system leveraging YOLOv8 and YOLOv9 for real-time vehicle monitoring. It includes additional features such as:

- Helmet Detection
- Triple Riding Detection
- Accident Detection
- Emergency Vehicle Detection

The system is developed using Flask and Python for the backend, while React.js is used for the frontend.

## Features

- **ANPR (Automatic Number Plate Recognition):** Detects and extracts vehicle license plates using YOLOv8/YOLOv9 models.
- **ATCC (Automated Traffic Control & Compliance):** Monitors traffic violations and ensures compliance with regulations.
- **Helmet Detection:** Identifies riders without helmets on two-wheelers.
- **Triple Riding Detection:** Detects cases where more than two riders are on a two-wheeler.
- **Accident Detection:** Recognizes accident scenarios using deep learning models.
- **Emergency Vehicle Detection:** Detects emergency vehicles and prioritizes them.

## Technologies Used

- **Backend:** Flask, Python
- **Frontend:** React.js
- **Machine Learning Models:** YOLOv8, YOLOv9, PyTesseract-OCR, Computer Vision
- **Database:** MySQL

  
## Installation

### Prerequisites

Ensure you have the following installed:

- Python (>=3.11)
- Node.js & npm (for frontend)
- Flask
- React.js
- OpenCV, NumPy, TensorFlow/PyTorch
- YOLOv8 & YOLOv9 (Ultralytics)
- Pytesseract-OCR
