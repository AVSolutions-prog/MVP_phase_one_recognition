#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
print(cv2.__version__)


# In[3]:


pip install mtcnn


# In[4]:


import tensorflow
print(tensorflow.__version__)


# In[6]:


pip install keras-facenet


# In[39]:


import cv2
import numpy as np
import sqlite3
from keras_facenet import FaceNet
from mtcnn import MTCNN


# In[40]:


detection_method = MTCNN()
embed_extract = FaceNet()


# In[41]:


def capture_img():
    video_cap = cv2.VideoCapture(0)
    ret,frame = video_cap.read()
    video_cap.release()
    return frame


# In[42]:


def detect_img(img):
    face = detection_method.detect_faces(img)
    return face


# In[43]:


def ext_face(img,box):
    x,y,ht,wt = box
    face = img[y:y+ht,x:x+wt]
    face = cv2.resize(face,(160,160))
    face = face.astype('float32')/255
    return face
    


# In[44]:


def face_embed(face_img):
    face_list = [face_img]
    embeddings = embed_extract.embeddings(face_list)
    return embed[0]


# In[45]:


def embed_store(name, embedding):
    conn = sqlite3.connect('face_recognition.db')  
    c = conn.cursor() 
    c.execute('''CREATE TABLE IF NOT EXISTS faces (name TEXT, encoding BLOB)''') 
    c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, embedding.tobytes()))  
    conn.commit() 
    conn.close() 


# In[51]:


def load_embed():
    conn = sqlite3.connect('face_recognition.db')  
    c = conn.cursor()
    c.execute("Select name,encoding from faces") 
    rows = c.fetchall()
    verified_face_name = []
    verified_face_encoding = []
    for row in rows:
        name = row[0]
        encoding = row[1]
        encoding = np.frombuffer(encoding,dtype = np.float32)
        verified_face_name.append(name)
        verified_face_encoding.append(encoding)
    conn.close()
    return verified_face_name,verified_face_encoding
    
    


# In[52]:


def calc_dist(embed1,embed2):
    return np.linalg.norm(embed1 - embed2)


# In[18]:


def capture_store():
    image = capture_img()  # Capture an image from the webcam
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    faces = detect_img(img)  # Detect faces using MTCNN

    if len(faces) == 0:
        print("No face detected. Try again.")
        return

    # Process the first face detected (you can modify this to handle multiple faces)
    face = faces[0]
    box = face['box']  # Get bounding box of the detected face

    # Extract the face region from the image
    face_image = ext_face(img, box)

    # Generate the face embedding using FaceNet
    embedding = face_embed(face_img)

    # Prompt the user for a name
    name = input("Enter the name for the face: ")

    # Store the face embedding in the database
    embed_store(name, embedding)
    print(f"Face embedding for {name} has been stored in the database.")


# #### All the peices together for the face detection and real-time verification system

# 1)	Import the required libraries:
# 2)	Initialize the pretrained models
# 3)	Capture the face image using the webcam (function 1)
# 4)	Detect the face using the MTCNN (function 2)
# 5)	Form a bounding box around the detected face and extract the face region (function 3)
# 6)	Generate face embeddings with Facenet from the extracted face region (function 4)
# 7)	Function to store the generated face embedding in the SQLite database(function 5)
# 8)	Load the embeddings from the database for verification (function 6)
# 9)	Calculate euclidean distance between stored and real time embedding(function 7)
# 10)	Capture and store a new face with their name during registration (function 8)
# 11)	Verify the user in real-time (function 9)
# 

# In[38]:


import cv2
import numpy as np
import sqlite3
from keras_facenet import FaceNet
from mtcnn import MTCNN

detection_method = MTCNN()
embed_extract = FaceNet()

def capture_img():
    video_cap = cv2.VideoCapture(0)
    ret,frame = video_cap.read()
    video_cap.release()
    return frame

def detect_img(img):
    face = detection_method.detect_faces(img)
    return face

def ext_face(img,box):
    x,y,wt,ht = box
    face = img[y:y+ht,x:x+wt]
    face = cv2.resize(face,(160,160))
    face = face.astype('float32')/255
    return face


def face_embed(face_img):
    face_list = [face_img]
    embeddings = embed_extract.embeddings(face_list)
    return embeddings[0]

def embed_store(name, embedding):
    conn = sqlite3.connect('face_recognition.db')  
    c = conn.cursor() 
    c.execute('''CREATE TABLE IF NOT EXISTS faces (name TEXT, encoding BLOB)''') 
    c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, embedding.tobytes()))  
    conn.commit() 
    conn.close() 
    
def load_embed():
    conn = sqlite3.connect('face_recognition.db')  
    c = conn.cursor()
    c.execute("Select name,encoding from faces") 
    rows = c.fetchall()
    verified_face_name = []
    verified_face_encoding = []
    for row in rows:
        name = row[0]
        encoding = row[1]
        encoding = np.frombuffer(encoding,dtype = np.float32)
        verified_face_name.append(name)
        verified_face_encoding.append(encoding)
    conn.close()
    return verified_face_name,verified_face_encoding

def calc_dist(embed1,embed2):
    return np.linalg.norm(embed1 - embed2)

def capture_store():
    image = capture_img()  # Capture an image from the webcam
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    faces = detect_img(rgb_image)  # Detect faces using MTCNN

    if len(faces) == 0:
        print("No face detected. Try again.")
        return

   
    face = faces[0]
    box = face['box']  # Get bounding box of the detected face

    # Extract the face region from the image
    face_image = ext_face(rgb_image, box)

    # Generate the face embedding using FaceNet
    embedding = face_embed(face_image)

    
    name = input("Enter the name for the face: ")

    # Store the face embedding in the database
    embed_store(name, embedding)
    print(f"Face embedding for {name} has been stored in the database.")
    
    
    
def verify_face():
    image = capture_img()  # Capture an image from the webcam
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    faces = detect_img(rgb_image)  # Detect faces using MTCNN

    if len(faces) == 0:
        print("No face detected. Try again.")
        return

    
    face = faces[0]
    box = face['box']  # Get bounding box of the detected face

    # Extract the face region from the image
    face_image = ext_face(rgb_image, box)

    # Generate the face embedding using FaceNet
    embedding = face_embed(face_image)
    
    threshold = 0.5
    
    verified_face_name , verified_face_embedding = load_embed()
    
    for i , verified_user_embedding in enumerate(verified_face_embedding):
        dist = calc_dist(embedding,verified_user_embedding)
        if dist < threshold:
            print(f"Match found: {verified_face_name[i]} (distance: {dist:.4f})")
            return
    print("No Match found")

def main():
    print("Capture and store the face on registration..")
    capture_store()
    print("\n Verifying the face on real-time..")
    verify_face()
    
main()


