import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import fileinput
import sys 
import pyttsx3
import cv2 

from os import path
from datetime import datetime, timedelta

from playsound import playsound

def filecheck():
    five_seconds_ago = datetime.now() - timedelta(seconds=2)
    filetime = datetime.fromtimestamp(path.getmtime(r"new_eye.png"))

    if filetime > five_seconds_ago:
        return True

model = tf.keras.models.load_model(r"C:\Users\Pranav Kukreja\Desktop\Personal Project\NewModel.h5")

class_names = ['0', '1' , '2' , '3' , '4' ] 

img_height = 224 
img_width = 224

engine = pyttsx3.init()

engine.setProperty("rate" , 130)

cam = cv2.VideoCapture(0)

cv2.namedWindow("Blind-Sighted Scanner")

img_counter = 0 

engine.say("Your Eyes are Ready to be scanned. ")
engine.runAndWait()

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame , 1)
    if not ret:
        break
    cv2.imshow("Blind-Sighted Scanner", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        break

    if cv2.waitKey(33) == ord(' '): #Space Button Pressed for Snapshot 
        img_name = "new_eye.png"
        cv2.imwrite(img_name, frame)
        playsound(r'C:\Users\Pranav Kukreja\Desktop\Hackathon\ShutterSound (mp3cut.net).wav')

    random_eye = r"new_eye.png"

    if filecheck():
        img = keras.preprocessing.image.load_img(
        random_eye, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_severe = (class_names[np.argmax(score)])

        engine.say( "This is " + predicted_severe)

        engine.runAndWait()


cam.release()

cv2.destroyAllWindows()