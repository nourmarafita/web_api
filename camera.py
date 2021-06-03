import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import cv2
from keras.models import model_from_json
from keras.preprocessing import image
import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

model = model_from_json(open("fer.json", "r").read())
# load weights
model.load_weights('fer.h5')

# load dataset
# Importing Data from CSV file
fer2013 = pd.read_csv("fer2013.csv")
labels = fer2013.iloc[:, [0]].values
pixels = fer2013['pixels']

# Facial Expressions
Ekspresi = {0: "Marah", 1: "Jijik", 2: "Takut", 3: "Senang", 4: "Sedih", 5: "Terkejut", 6: "Netral"}
from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels, len(Ekspresi))

# converting pixels to Gray Scale images of 48X48
image = np.array([np.fromstring(pixel, dtype=int, sep=" ") for pixel in pixels])
image = image / 255.0
image = image.reshape(image.shape[0], 48, 48, 1).astype('float32')

train_features,test_features,train_labels,test_labels = train_test_split(image,labels,test_size=0.2,random_state=0)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

score = model.evaluate(train_features,train_labels, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.__current_frame = None

    def __del__(self):
        self.video.release()

    def make_prediction(self, pred):
        pred = cv2.resize(pred, (48, 48))
        pred = pred / 255.0
        pred = np.array(pred).reshape(-1, 48, 48, 1)
        predict = np.argmax(model.predict(pred), axis=1)
        return predict[0]

    def get_frame(self):
        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        while True:
            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in face:
                sub_face = gray[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                res = self.make_prediction(sub_face)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, str(Ekspresi[res]), (x, y - 5), font, 0.5, (205, 200, 50), 1, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()
        flag, encodedImage = cv2.imencode('.jpg', frame)
        self.__current_frame = bytearray(encodedImage)
        return encodedImage.tobytes()


