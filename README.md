# Face Recognition System

This project implements a face recognition system using MTCNN for face detection and Facenet for generating face embeddings. It also integrates an SQLite database for storing and retrieving embeddings for user verification.

## Features
1. Capture face images using a webcam.
2. Detect faces using MTCNN.
3. Extract the face region and generate face embeddings using Facenet.
4. Store the embeddings in an SQLite database for later verification.
5. Perform real-time face recognition and verification.

## Steps

### 1. Import the required libraries
Ensure that you have installed the necessary dependencies like `MTCNN`, `Facenet`, `SQLite`, and others.

```python
import cv2
import numpy as np
import sqlite3
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
