#immporting libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

emotions = ['anger', 'Disgust', 'fear', 'happy', 'Neutreal', 'sad', 'Surprise']
model = tf.keras.models.load_model('modelv2.keras')
#using Cascade to detect faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
         
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (48, 48))
        image_array = np.array(resized_face) 
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=-1)
        
        emotion_prediction = model.predict(image_array)
        emotion_prediction = emotions[np.argmax(emotion_prediction)]
        
        cv2.putText(frame, emotion_prediction , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow('emotion', frame)
        cv2.imshow('detectetd face', resized_face)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()